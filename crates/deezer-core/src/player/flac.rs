//! FLAC decoding that can seek.
//!
//! rodio 0.19 hands symphonia a source with no length, and symphonia's FLAC
//! demuxer refuses to seek without one (`SeekError(Unseekable)`). This decoder
//! is rodio's symphonia decoder, built on a source that reports its length.
//!
//! For a track still streaming, the reported length is what has downloaded so
//! far: the demuxer's binary search then stays within bytes that are already
//! there instead of waiting on the network for the middle of the file.

use std::io::{Read, Seek, SeekFrom};
use std::time::Duration;

use rodio::source::SeekError;
use rodio::Source;
use symphonia::core::audio::{AudioBufferRef, SampleBuffer, SignalSpec};
use symphonia::core::codecs::{Decoder, DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error;
use symphonia::core::formats::{FormatOptions, FormatReader, SeekMode, SeekTo, SeekedTo};
use symphonia::core::io::{MediaSource, MediaSourceStream};
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use symphonia::core::units::{self, Time};

use crate::api::models::DeezerError;
use crate::player::engine::AudioInput;

/// Consecutive undecodable packets tolerated before giving up (as rodio).
const MAX_DECODE_RETRIES: usize = 3;

impl MediaSource for AudioInput {
    fn is_seekable(&self) -> bool {
        true
    }

    fn byte_len(&self) -> Option<u64> {
        Some(self.available_len())
    }
}

/// Whether `audio` starts with the FLAC signature. Leaves it at the start.
pub fn is_flac(audio: &mut AudioInput) -> bool {
    let mut magic = [0u8; 4];
    let found = audio.read_exact(&mut magic).is_ok() && &magic == b"fLaC";
    audio.seek(SeekFrom::Start(0)).is_ok() && found
}

pub struct FlacDecoder {
    decoder: Box<dyn Decoder>,
    format: Box<dyn FormatReader>,
    total_duration: Option<Time>,
    buffer: SampleBuffer<i16>,
    spec: SignalSpec,
    current_frame_offset: usize,
}

impl FlacDecoder {
    pub fn new(audio: AudioInput) -> Result<Self, DeezerError> {
        Self::init(audio).map_err(|e| DeezerError::Playback(format!("Failed to decode FLAC: {e}")))
    }

    fn init(audio: AudioInput) -> Result<Self, Error> {
        let mss = MediaSourceStream::new(Box::new(audio), Default::default());
        let mut hint = Hint::new();
        hint.with_extension("flac");
        let format_opts = FormatOptions {
            enable_gapless: true,
            ..Default::default()
        };
        let mut format = symphonia::default::get_probe()
            .format(&hint, mss, &format_opts, &MetadataOptions::default())?
            .format;

        let track = format
            .tracks()
            .iter()
            .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
            .ok_or(Error::Unsupported("No track with supported codec"))?;
        let track_id = track.id;
        let total_duration = track
            .codec_params
            .time_base
            .zip(track.codec_params.n_frames)
            .map(|(base, frames)| base.calc_time(frames));
        let mut decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &DecoderOptions::default())?;

        let mut decode_errors = 0;
        let decoded = loop {
            let packet = match format.next_packet() {
                Ok(packet) => packet,
                Err(Error::IoError(_)) => break decoder.last_decoded(),
                Err(e) => return Err(e),
            };
            if packet.track_id() != track_id {
                continue;
            }
            match decoder.decode(&packet) {
                Ok(decoded) => break decoded,
                Err(Error::DecodeError(_)) if decode_errors < MAX_DECODE_RETRIES => {
                    decode_errors += 1;
                }
                Err(e) => return Err(e),
            }
        };
        let spec = *decoded.spec();
        let buffer = Self::get_buffer(decoded, &spec);
        Ok(Self {
            decoder,
            format,
            total_duration,
            buffer,
            spec,
            current_frame_offset: 0,
        })
    }

    fn get_buffer(decoded: AudioBufferRef, spec: &SignalSpec) -> SampleBuffer<i16> {
        let duration = units::Duration::from(decoded.capacity() as u64);
        let mut buffer = SampleBuffer::<i16>::new(duration, *spec);
        buffer.copy_interleaved_ref(decoded);
        buffer
    }

    /// Decode from the packet holding the seek target and skip to the exact
    /// sample. The frame offset must be adjusted by the caller afterwards.
    fn refine_position(&mut self, seek_res: SeekedTo) -> Result<(), Error> {
        let mut samples_to_pass = seek_res.required_ts - seek_res.actual_ts;
        let packet = loop {
            let candidate = self.format.next_packet()?;
            if candidate.dur() > samples_to_pass {
                break candidate;
            }
            samples_to_pass -= candidate.dur();
        };

        let mut decoded = self.decoder.decode(&packet);
        for _ in 0..MAX_DECODE_RETRIES {
            if decoded.is_err() {
                let packet = self.format.next_packet()?;
                decoded = self.decoder.decode(&packet);
            }
        }
        let decoded = decoded?;
        self.spec = *decoded.spec();
        self.buffer = Self::get_buffer(decoded, &self.spec);
        self.current_frame_offset = samples_to_pass as usize * self.channels() as usize;
        Ok(())
    }
}

impl Source for FlacDecoder {
    fn current_frame_len(&self) -> Option<usize> {
        Some(self.buffer.samples().len())
    }

    fn channels(&self) -> u16 {
        self.spec.channels.count() as u16
    }

    fn sample_rate(&self) -> u32 {
        self.spec.rate
    }

    fn total_duration(&self) -> Option<Duration> {
        self.total_duration.map(|Time { seconds, frac }| {
            Duration::from_secs(seconds) + Duration::from_secs_f64(frac)
        })
    }

    fn try_seek(&mut self, pos: Duration) -> Result<(), SeekError> {
        // Seeking right onto the end fails in some demuxers: stop just before.
        let pos = match self.total_duration() {
            Some(total) if pos + Duration::from_millis(1) >= total => {
                total.saturating_sub(Duration::from_millis(1))
            }
            _ => pos,
        };
        // Keep the next sample on the same channel.
        let to_skip = self.current_frame_offset % self.channels() as usize;

        let seek_res = self
            .format
            .seek(
                SeekMode::Accurate,
                SeekTo::Time {
                    time: pos.as_secs_f64().into(),
                    track_id: None,
                },
            )
            .map_err(|e| SeekError::Other(Box::new(e)))?;
        self.refine_position(seek_res)
            .map_err(|e| SeekError::Other(Box::new(e)))?;
        self.current_frame_offset += to_skip;
        Ok(())
    }
}

impl Iterator for FlacDecoder {
    type Item = i16;

    fn next(&mut self) -> Option<i16> {
        if self.current_frame_offset >= self.buffer.len() {
            let packet = self.format.next_packet().ok()?;
            let mut decoded = self.decoder.decode(&packet);
            for _ in 0..MAX_DECODE_RETRIES {
                if decoded.is_err() {
                    let packet = self.format.next_packet().ok()?;
                    decoded = self.decoder.decode(&packet);
                }
            }
            let decoded = decoded.ok()?;
            self.spec = *decoded.spec();
            self.buffer = Self::get_buffer(decoded, &self.spec);
            self.current_frame_offset = 0;
        }

        let sample = *self.buffer.samples().get(self.current_frame_offset)?;
        self.current_frame_offset += 1;
        Some(sample)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_flac_and_rewinds() {
        let mut audio: AudioInput = b"fLaC\0\0\0\x22".to_vec().into();
        assert!(is_flac(&mut audio));
        let mut first = [0u8; 4];
        audio.read_exact(&mut first).unwrap();
        assert_eq!(&first, b"fLaC");
    }

    #[test]
    fn mp3_is_not_flac() {
        let mut audio: AudioInput = b"ID3\x04\0\0\0\0".to_vec().into();
        assert!(!is_flac(&mut audio));
        assert_eq!(audio.stream_position().unwrap(), 0);
    }
}
