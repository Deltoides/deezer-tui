use std::io::{self, Read, Seek, SeekFrom};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use tracing::{debug, info, warn};

use crate::api::models::{AudioQuality, DeezerError, TrackData};
use crate::api::DeezerClient;
use crate::decrypt::{self, BLOCK_SIZE};

/// Bytes buffered before playback starts (~2 s of FLAC, ~16 s of MP3 128).
const PREBUFFER_BYTES: usize = 256 * 1024;

/// Whole-body limit for a streamed download. The CDN client's 30 s timeout is
/// sized for full downloads of small files; a FLAC track on a slow link can
/// legitimately take longer while it is already playing.
const STREAM_TIMEOUT: Duration = Duration::from_secs(15 * 60);

/// A connection that sends nothing for this long is treated as dropped and
/// resumed, instead of leaving playback frozen until `STREAM_TIMEOUT`.
const STALL_TIMEOUT: Duration = Duration::from_secs(20);

/// Resume attempts after a dropped or stalled connection before giving up.
const MAX_RETRIES: u32 = 3;

/// Create a shared HTTP client suitable for CDN downloads (no cookies needed).
pub fn new_cdn_client() -> Result<reqwest::Client, DeezerError> {
    reqwest::Client::builder()
        .user_agent("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36")
        .timeout(std::time::Duration::from_secs(30))
        .connect_timeout(std::time::Duration::from_secs(10))
        .build()
        .map_err(|e| DeezerError::Http(e.to_string()))
}

/// Download and decrypt a full track, returning raw audio bytes (MP3 or FLAC).
pub async fn fetch_track(
    client: &DeezerClient,
    track: &TrackData,
    quality: AudioQuality,
    master_key: &[u8; 16],
) -> Result<(Vec<u8>, AudioQuality), DeezerError> {
    // Get streaming URL (with quality fallback)
    let (url, actual_quality) = client.get_stream_url(track, quality).await?;

    let data = download_and_decrypt(&url, &track.track_id, master_key, client.http()).await?;

    Ok((data, actual_quality))
}

/// Download encrypted audio from a CDN URL and decrypt it.
/// This function does NOT need a `DeezerClient` reference, so it can run
/// without holding the client lock.
/// Pass a shared `reqwest::Client` to reuse connections across downloads.
pub async fn download_and_decrypt(
    url: &str,
    track_id: &str,
    master_key: &[u8; 16],
    http: &reqwest::Client,
) -> Result<Vec<u8>, DeezerError> {
    info!(track_id = %track_id, "Downloading track");

    let resp = http
        .get(url)
        .send()
        .await
        .map_err(|e| DeezerError::Http(e.to_string()))?;

    if !resp.status().is_success() {
        return Err(DeezerError::Http(format!(
            "CDN returned status {}",
            resp.status()
        )));
    }

    let mut data = resp
        .bytes()
        .await
        .map_err(|e| DeezerError::Http(e.to_string()))?
        .to_vec();

    debug!(
        bytes = data.len(),
        track_id = %track_id,
        "Downloaded encrypted audio"
    );

    // Derive per-track key and decrypt
    let track_key = decrypt::derive_track_key(track_id, master_key);
    decrypt::decrypt_stream(&mut data, &track_key)?;

    debug!(track_id = %track_id, "Decrypted audio");

    Ok(data)
}

/// Download unencrypted audio, for podcast episodes served by the show's own
/// host rather than Deezer's CDN.
pub async fn download_direct(url: &str, http: &reqwest::Client) -> Result<Vec<u8>, DeezerError> {
    info!("Downloading direct stream");

    let resp = http
        .get(url)
        .send()
        .await
        .map_err(|e| DeezerError::Http(e.to_string()))?;

    if !resp.status().is_success() {
        return Err(DeezerError::Http(format!(
            "Stream host returned status {}",
            resp.status()
        )));
    }

    let data = resp
        .bytes()
        .await
        .map_err(|e| DeezerError::Http(e.to_string()))?
        .to_vec();

    debug!(bytes = data.len(), "Downloaded direct stream");

    Ok(data)
}

/// Decrypted audio shared between the download task (writer) and the decoder
/// (reader). Grows as blocks arrive; readers block until their bytes are there.
struct StreamBuffer {
    state: Mutex<StreamState>,
    grew: Condvar,
    /// Full file size from `Content-Length`, needed to answer `SeekFrom::End`.
    total_len: Option<u64>,
}

#[derive(Default)]
struct StreamState {
    data: Vec<u8>,
    done: bool,
    /// Set when the track is skipped or stopped: the download stops and any
    /// read blocked on missing bytes returns. Kept under the lock so a waiting
    /// reader cannot miss the wakeup.
    cancelled: bool,
    /// Why the download gave up before the end (retries exhausted).
    error: Option<String>,
}

impl StreamBuffer {
    fn new(total_len: Option<u64>) -> Self {
        Self {
            state: Mutex::new(StreamState::default()),
            grew: Condvar::new(),
            total_len,
        }
    }

    /// Append decrypted bytes, wake readers, return the new length.
    fn append(&self, bytes: &[u8]) -> usize {
        let len = {
            let mut st = self.state.lock().unwrap();
            st.data.extend_from_slice(bytes);
            st.data.len()
        };
        self.grew.notify_all();
        len
    }

    /// Mark the download over, successfully or not; return the final length.
    fn finish(&self, error: Option<String>) -> usize {
        let len = {
            let mut st = self.state.lock().unwrap();
            st.done = true;
            st.error = error;
            st.data.len()
        };
        self.grew.notify_all();
        len
    }

    fn cancel(&self) {
        self.state.lock().unwrap().cancelled = true;
        self.grew.notify_all();
    }

    fn is_cancelled(&self) -> bool {
        self.state.lock().unwrap().cancelled
    }
}

/// Control over a streaming track, kept by the engine while the decoder owns
/// the `StreamReader`.
#[derive(Clone)]
pub struct StreamHandle(Arc<StreamBuffer>);

impl StreamHandle {
    /// Stop the download and release a read blocked on missing bytes, so a
    /// skipped track frees the audio thread even on a stalled network.
    pub fn cancel(&self) {
        self.0.cancel();
    }

    /// Error that cut the download short, once it has given up.
    pub fn error(&self) -> Option<String> {
        self.0.state.lock().unwrap().error.clone()
    }
    /// Share of the file downloaded so far, or `None` once the download is
    /// over. `Some(0.0)` when the server sent no `Content-Length`.
    pub fn progress(&self) -> Option<f64> {
        let st = self.0.state.lock().unwrap();
        if st.done {
            return None;
        }
        Some(match self.0.total_len {
            Some(total) if total > 0 => (st.data.len() as f64 / total as f64).min(1.0),
            _ => 0.0,
        })
    }
}

/// `Read + Seek` view over a track that is still downloading, so playback can
/// start after a short prebuffer instead of after the whole file.
pub struct StreamReader {
    buf: Arc<StreamBuffer>,
    pos: u64,
}

impl StreamReader {
    pub fn handle(&self) -> StreamHandle {
        StreamHandle(Arc::clone(&self.buf))
    }
}

fn cancelled_error() -> io::Error {
    io::Error::new(io::ErrorKind::ConnectionAborted, "stream cancelled")
}

impl Read for StreamReader {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if out.is_empty() {
            return Ok(0);
        }
        let mut st = self.buf.state.lock().unwrap();
        loop {
            if st.cancelled {
                return Err(cancelled_error());
            }
            let len = st.data.len() as u64;
            if self.pos < len {
                let start = self.pos as usize;
                let n = out.len().min(st.data.len() - start);
                out[..n].copy_from_slice(&st.data[start..start + n]);
                self.pos += n as u64;
                return Ok(n);
            }
            if st.done {
                // A failed download must not look like a complete track.
                return match &st.error {
                    Some(e) => Err(io::Error::other(e.clone())),
                    None => Ok(0),
                };
            }
            st = self.buf.grew.wait(st).unwrap();
        }
    }
}

impl Seek for StreamReader {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let target = match pos {
            SeekFrom::Start(p) => Some(p),
            SeekFrom::Current(d) => self.pos.checked_add_signed(d),
            SeekFrom::End(d) => {
                let total = match self.buf.total_len {
                    Some(t) => t,
                    None => {
                        // No Content-Length: the end is only known once downloaded.
                        let mut st = self.buf.state.lock().unwrap();
                        while !st.done && !st.cancelled {
                            st = self.buf.grew.wait(st).unwrap();
                        }
                        if st.cancelled {
                            return Err(cancelled_error());
                        }
                        st.data.len() as u64
                    }
                };
                total.checked_add_signed(d)
            }
        };
        // Seeking past what has arrived is fine: the next read waits for it.
        self.pos = target.ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "seek before start of stream")
        })?;
        Ok(self.pos)
    }
}

impl Drop for StreamReader {
    fn drop(&mut self) {
        self.buf.cancel();
    }
}

/// GET the encrypted file from `offset` on. Also returns how many leading
/// bytes to discard, for a CDN that ignores `Range` and answers 200.
async fn open_stream(
    http: &reqwest::Client,
    url: &str,
    offset: usize,
) -> Result<(reqwest::Response, u64), DeezerError> {
    let mut req = http.get(url).timeout(STREAM_TIMEOUT);
    if offset > 0 {
        req = req.header(reqwest::header::RANGE, format!("bytes={offset}-"));
    }
    let resp = req
        .send()
        .await
        .map_err(|e| DeezerError::Http(e.to_string()))?;

    if !resp.status().is_success() {
        return Err(DeezerError::Http(format!(
            "CDN returned status {}",
            resp.status()
        )));
    }
    let skip = if offset > 0 && resp.status() != reqwest::StatusCode::PARTIAL_CONTENT {
        offset as u64
    } else {
        0
    };
    Ok((resp, skip))
}

/// Download loop: decrypt whole blocks into `buf` as they arrive, resuming
/// with a `Range` request when the connection drops or stalls.
async fn pump(
    buf: &StreamBuffer,
    mut resp: reqwest::Response,
    url: &str,
    http: &reqwest::Client,
    track_key: &[u8; 16],
    ready_tx: &mut Option<tokio::sync::oneshot::Sender<Result<(), DeezerError>>>,
) -> Result<(), DeezerError> {
    let mut pending: Vec<u8> = Vec::with_capacity(4 * BLOCK_SIZE);
    // Bytes in `buf`: always whole blocks until the final append.
    let mut written = 0usize;
    let mut skip = 0u64;
    let mut retries = 0u32;

    loop {
        if buf.is_cancelled() {
            return Ok(());
        }
        let chunk = match tokio::time::timeout(STALL_TIMEOUT, resp.chunk()).await {
            Ok(Ok(Some(chunk))) => chunk,
            Ok(Ok(None)) => {
                // Trailing partial block is never encrypted.
                buf.append(&pending);
                return Ok(());
            }
            failure => {
                let mut err = match failure {
                    Ok(Err(e)) => DeezerError::Http(e.to_string()),
                    _ => DeezerError::Http(format!("no data for {}s", STALL_TIMEOUT.as_secs())),
                };
                // Restart from the last whole block written.
                pending.clear();
                loop {
                    retries += 1;
                    if retries > MAX_RETRIES {
                        return Err(err);
                    }
                    warn!(err = %err, offset = written, attempt = retries, "Stream interrupted, resuming");
                    tokio::time::sleep(Duration::from_secs(retries.into())).await;
                    if buf.is_cancelled() {
                        return Ok(());
                    }
                    match open_stream(http, url, written).await {
                        Ok((r, s)) => {
                            resp = r;
                            skip = s;
                            break;
                        }
                        Err(e) => err = e,
                    }
                }
                continue;
            }
        };
        retries = 0;

        let mut chunk = &chunk[..];
        if skip > 0 {
            let n = chunk.len().min(skip as usize);
            chunk = &chunk[n..];
            skip -= n as u64;
        }
        pending.extend_from_slice(chunk);

        let whole = pending.len() / BLOCK_SIZE * BLOCK_SIZE;
        if whole == 0 {
            continue;
        }
        let mut blocks: Vec<u8> = pending.drain(..whole).collect();
        let first_block = written / BLOCK_SIZE;
        for (i, block) in blocks.chunks_exact_mut(BLOCK_SIZE).enumerate() {
            decrypt::decrypt_block(block, first_block + i, track_key)?;
        }
        written = buf.append(&blocks);
        if written >= PREBUFFER_BYTES {
            if let Some(tx) = ready_tx.take() {
                let _ = tx.send(Ok(()));
            }
        }
    }
}

/// Start downloading an encrypted track and return a reader as soon as
/// `PREBUFFER_BYTES` are decrypted. The rest keeps downloading in a background
/// task, decrypting each 2048-byte block as it arrives (Deezer's stripe cipher
/// restarts its CBC chain on every block, so blocks decrypt independently).
pub async fn stream_and_decrypt(
    url: &str,
    track_id: &str,
    master_key: &[u8; 16],
    http: &reqwest::Client,
) -> Result<StreamReader, DeezerError> {
    info!(track_id = %track_id, "Streaming track");

    let (resp, _) = open_stream(http, url, 0).await?;
    let buf = Arc::new(StreamBuffer::new(resp.content_length()));
    let track_key = decrypt::derive_track_key(track_id, master_key);
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel::<Result<(), DeezerError>>();

    let writer = Arc::clone(&buf);
    let track_id = track_id.to_string();
    let url = url.to_string();
    let http = http.clone();
    tokio::spawn(async move {
        let mut ready_tx = Some(ready_tx);
        let result = pump(&writer, resp, &url, &http, &track_key, &mut ready_tx).await;

        let total = writer.finish(result.as_ref().err().map(ToString::to_string));
        match &result {
            Ok(()) if writer.is_cancelled() => {
                debug!(track_id = %track_id, bytes = total, "Stream cancelled")
            }
            Ok(()) => debug!(track_id = %track_id, bytes = total, "Stream complete"),
            Err(e) => warn!(track_id = %track_id, bytes = total, err = %e, "Stream failed"),
        }
        if let Some(tx) = ready_tx.take() {
            // Short file, or failed before the prebuffer filled.
            let _ = tx.send(if total > 0 { Ok(()) } else { result });
        }
    });

    match ready_rx.await {
        Ok(Ok(())) => Ok(StreamReader { buf, pos: 0 }),
        Ok(Err(e)) => Err(e),
        Err(_) => Err(DeezerError::Http("stream task ended unexpectedly".into())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    fn reader(total_len: Option<u64>) -> StreamReader {
        StreamReader {
            buf: Arc::new(StreamBuffer::new(total_len)),
            pos: 0,
        }
    }

    #[test]
    fn read_waits_for_data_then_ends() {
        let mut r = reader(None);
        let buf = Arc::clone(&r.buf);
        let writer = thread::spawn(move || {
            thread::sleep(Duration::from_millis(50));
            buf.append(b"hello");
            buf.finish(None);
        });
        let mut out = Vec::new();
        r.read_to_end(&mut out).unwrap();
        writer.join().unwrap();
        assert_eq!(out, b"hello");
    }

    #[test]
    fn failed_download_is_an_error_not_eof() {
        let mut r = reader(None);
        r.buf.append(b"abc");
        r.buf.finish(Some("connection reset".into()));
        let mut out = [0u8; 8];
        assert_eq!(r.read(&mut out).unwrap(), 3);
        assert!(r.read(&mut out).is_err());
        assert_eq!(r.handle().error().as_deref(), Some("connection reset"));
    }

    #[test]
    fn cancel_releases_blocked_read() {
        let mut r = reader(None);
        let handle = r.handle();
        let canceller = thread::spawn(move || {
            thread::sleep(Duration::from_millis(50));
            handle.cancel();
        });
        let err = r.read(&mut [0u8; 8]).unwrap_err();
        canceller.join().unwrap();
        assert_eq!(err.kind(), io::ErrorKind::ConnectionAborted);
    }

    #[test]
    fn seek_end_uses_content_length() {
        let mut r = reader(Some(1000));
        assert_eq!(r.seek(SeekFrom::End(-10)).unwrap(), 990);
        assert_eq!(r.seek(SeekFrom::Current(-90)).unwrap(), 900);
        assert!(r.seek(SeekFrom::Current(-1000)).is_err());
    }

    #[test]
    fn progress_tracks_download_until_done() {
        let r = reader(Some(1000));
        r.buf.append(&[0u8; 250]);
        assert_eq!(r.handle().progress(), Some(0.25));
        r.buf.finish(None);
        assert_eq!(r.handle().progress(), None);
    }

    #[test]
    fn dropping_reader_cancels_download() {
        let r = reader(None);
        let handle = r.handle();
        drop(r);
        assert!(handle.0.is_cancelled());
    }
}
