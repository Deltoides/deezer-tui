use std::io::{self, Cursor, Read, Seek, SeekFrom};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use rodio::{Decoder, OutputStream, OutputStreamHandle, Sink, Source};
use tracing::{debug, info};

use crate::api::models::{AudioQuality, DeezerError, TrackData};
use crate::player::eq::{EqHandle, Equalizer};
use crate::player::flac::{self, FlacDecoder};
use crate::player::state::{PlaybackStatus, PlayerState};
use crate::player::stream::{StreamHandle, StreamReader};

/// Audio handed to the engine: a fully downloaded file (offline tracks,
/// podcast episodes) or a track still streaming from the CDN.
pub enum AudioInput {
    Bytes(Cursor<Vec<u8>>),
    Stream(StreamReader),
}

impl From<Vec<u8>> for AudioInput {
    fn from(data: Vec<u8>) -> Self {
        Self::Bytes(Cursor::new(data))
    }
}

impl From<StreamReader> for AudioInput {
    fn from(reader: StreamReader) -> Self {
        Self::Stream(reader)
    }
}

impl AudioInput {
    /// Size for logs: the full length for downloaded audio, `None` while streaming.
    pub fn known_len(&self) -> Option<usize> {
        match self {
            Self::Bytes(c) => Some(c.get_ref().len()),
            Self::Stream(_) => None,
        }
    }

    /// Bytes readable without waiting: everything for a downloaded file, what
    /// has arrived so far for a stream.
    pub fn available_len(&self) -> u64 {
        match self {
            Self::Bytes(c) => c.get_ref().len() as u64,
            Self::Stream(s) => s.available_len(),
        }
    }

    fn stream_handle(&self) -> Option<StreamHandle> {
        match self {
            Self::Bytes(_) => None,
            Self::Stream(s) => Some(s.handle()),
        }
    }
}

impl Read for AudioInput {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match self {
            Self::Bytes(c) => c.read(buf),
            Self::Stream(s) => s.read(buf),
        }
    }
}

impl Seek for AudioInput {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        match self {
            Self::Bytes(c) => c.seek(pos),
            Self::Stream(s) => s.seek(pos),
        }
    }
}

pub struct PlayerEngine {
    state: Arc<Mutex<PlayerState>>,
    _stream: OutputStream,
    stream_handle: OutputStreamHandle,
    sink: Sink,
    eq: EqHandle,
    /// Download behind the current track, when it is streamed from the CDN.
    stream: Option<StreamHandle>,
}

impl PlayerEngine {
    /// Builds the engine on top of an existing shared state handle, so callers that
    /// already handed out `Arc` clones (e.g. MPRIS) keep seeing live updates instead
    /// of a snapshot frozen before the engine existed. `eq` is likewise shared:
    /// settings applied to it are heard live on the playing track.
    pub fn new(
        _master_key: [u8; 16],
        state: Arc<Mutex<PlayerState>>,
        eq: EqHandle,
    ) -> Result<Self, DeezerError> {
        let (stream, stream_handle) =
            OutputStream::try_default().map_err(|e| DeezerError::Playback(e.to_string()))?;

        let sink =
            Sink::try_new(&stream_handle).map_err(|e| DeezerError::Playback(e.to_string()))?;

        Ok(Self {
            state,
            _stream: stream,
            stream_handle,
            sink,
            eq,
            stream: None,
        })
    }

    pub fn state(&self) -> Arc<Mutex<PlayerState>> {
        Arc::clone(&self.state)
    }

    /// Play decrypted audio: fully downloaded, or still streaming in.
    /// Called on the main thread with audio from a background fetch.
    pub fn play_decoded(
        &mut self,
        audio: impl Into<AudioInput>,
        track: &TrackData,
        quality: AudioQuality,
    ) -> Result<(), DeezerError> {
        let mut audio = audio.into();
        let stream = audio.stream_handle();
        // rodio's own decoder cannot seek in FLAC (see `flac`); MP3 keeps it.
        let source: Box<dyn Source<Item = i16> + Send> = if flac::is_flac(&mut audio) {
            Box::new(FlacDecoder::new(audio)?)
        } else {
            Box::new(
                Decoder::new(audio)
                    .map_err(|e| DeezerError::Playback(format!("Failed to decode audio: {e}")))?,
            )
        };
        let source = Equalizer::new(source, self.eq.clone());

        // Preserve current volume before recreating the sink
        let current_volume = self.state.lock().unwrap().volume;

        // The previous track may be blocked on the network inside the audio
        // thread: cancel its stream so stopping the sink takes effect now.
        if let Some(old) = std::mem::replace(&mut self.stream, stream) {
            old.cancel();
        }

        // Clear the current sink and create a fresh one (Sink can't be reused after stop)
        self.sink.stop();
        self.sink =
            Sink::try_new(&self.stream_handle).map_err(|e| DeezerError::Playback(e.to_string()))?;
        self.sink.set_volume(current_volume);
        self.sink.append(source);
        self.sink.play();

        info!(
            title = %track.title,
            artist = %track.artist,
            quality = quality.as_api_format(),
            "Now playing"
        );

        {
            let mut state = self.state.lock().unwrap();
            state.status = PlaybackStatus::Playing;
            state.current_track = Some(track.clone());
            state.duration_secs = track.duration_secs();
            state.position_secs = 0;
            state.quality = quality;
        }

        Ok(())
    }

    pub fn pause(&self) {
        self.sink.pause();
        let mut state = self.state.lock().unwrap();
        state.status = PlaybackStatus::Paused;
        debug!("Paused");
    }

    pub fn resume(&self) {
        self.sink.play();
        let mut state = self.state.lock().unwrap();
        state.status = PlaybackStatus::Playing;
        debug!("Resumed");
    }

    pub fn toggle_pause(&self) {
        let status = self.state.lock().unwrap().status;
        match status {
            PlaybackStatus::Playing => self.pause(),
            PlaybackStatus::Paused => self.resume(),
            _ => {}
        }
    }

    pub fn stop(&mut self) {
        if let Some(stream) = self.stream.take() {
            stream.cancel();
        }
        self.sink.stop();
        // Recreate sink for future use
        if let Ok(new_sink) = Sink::try_new(&self.stream_handle) {
            self.sink = new_sink;
        }
        let mut state = self.state.lock().unwrap();
        state.status = PlaybackStatus::Stopped;
        state.current_track = None;
        state.position_secs = 0;
        state.duration_secs = 0;
        debug!("Stopped");
    }

    pub fn set_volume(&self, volume: f32) {
        let volume = volume.clamp(0.0, 1.0);
        self.sink.set_volume(volume);
        self.state.lock().unwrap().volume = volume;
    }

    pub fn volume(&self) -> f32 {
        self.state.lock().unwrap().volume
    }

    pub fn try_seek(&self, pos: Duration) -> Result<(), DeezerError> {
        self.sink
            .try_seek(pos)
            .map_err(|e| DeezerError::Playback(format!("Seek failed: {e}")))
    }

    pub fn is_finished(&self) -> bool {
        self.sink.empty()
    }

    /// Position in the track as actually played (follows seeks, and stays
    /// put while the stream waits on the network).
    pub fn position(&self) -> Duration {
        self.sink.get_pos()
    }

    /// Share of the current track downloaded so far, or `None` when it is all
    /// in memory (finished stream, offline file, podcast download).
    pub fn stream_progress(&self) -> Option<f64> {
        self.stream.as_ref()?.progress()
    }

    /// Why the current track's download gave up (network lost beyond the
    /// resume attempts), so the track ended early. Returned once.
    pub fn take_stream_error(&mut self) -> Option<String> {
        let err = self.stream.as_ref()?.error()?;
        self.stream = None;
        Some(err)
    }
}
