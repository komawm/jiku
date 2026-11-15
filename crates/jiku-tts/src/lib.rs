#![forbid(unsafe_code)]

//! Optional speech synthesis providers, lip-sync envelope extractors, and
//! D-Bus desktop notification helpers for Jiku.
//!
//! Disabled by default unless explicitly enabled in `config.toml` (`[tts] enabled = true`).

use std::{
    f64::consts::PI,
    future::Future,
    io::{self, Write as _},
    path::Path,
    pin::Pin,
    process::{Child, Command, Stdio},
    str::FromStr,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    thread,
    time::Duration,
};

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Default local VOICEVOX Engine endpoint.
pub const DEFAULT_VOICEVOX_ENDPOINT: &str = "http://127.0.0.1:50021";

/// Default local Irodori-TTS HTTP inference endpoint.
pub const DEFAULT_IRODORI_ENDPOINT: &str = "http://127.0.0.1:8080/v1/audio/speech";

/// Errors returned by [`jiku-tts`](crate) providers and envelope extractors.
#[derive(Debug, Error)]
pub enum TtsError {
    /// Failed parsing VOICEVOX `AudioQuery` phoneme payload.
    #[error("invalid VOICEVOX audio_query payload: {0}")]
    InvalidAudioQuery(String),

    /// Invalid PCM byte buffer length (expected even 16-bit LE samples).
    #[error("invalid 16-bit PCM byte buffer length: {0}")]
    InvalidPcmLength(usize),

    /// Invalid or truncated RIFF/WAVE container.
    #[error("invalid WAV container: {0}")]
    InvalidWav(String),

    /// Unsupported WAV sample encoding, channel count, or bit depth.
    #[error(
        "unsupported WAV audio format: format={format}, bits_per_sample={bits_per_sample}, channels={channels}"
    )]
    UnsupportedWavFormat {
        /// RIFF WAVE audio format tag (`1` = PCM, `3` = IEEE float).
        format: u16,
        /// Bits per sample.
        bits_per_sample: u16,
        /// Channel count.
        channels: u16,
    },

    /// External TTS synthesizer or playback process failed.
    #[error("TTS process `{program}` failed: {detail}")]
    ProcessFailed {
        /// Executable name or path.
        program: String,
        /// Failure details.
        detail: String,
    },

    /// Subprocess or pipe I/O failure.
    #[error("TTS I/O error: {0}")]
    Io(#[from] io::Error),

    /// HTTP client or remote TTS server error.
    #[error("TTS HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),

    /// Remote TTS endpoint returned a non-success HTTP status.
    #[error("TTS endpoint returned HTTP {status}: {body}")]
    RemoteStatus {
        /// HTTP status code.
        status: u16,
        /// Response error text.
        body: String,
    },
}

/// Companion emotion state emitted alongside `AvatarCue` events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AvatarEmotion {
    /// Calm resting state.
    #[default]
    Neutral,
    /// Ambient idle state.
    Idle,
    /// Waiting on planner or decision batch.
    Thinking,
    /// Executing Koma workspace action.
    Acting,
    /// Playing synthesized or live speech.
    Speaking,
    /// Paused awaiting human verification (`Mod + G` / 2FA).
    WaitingHuman,
    /// Completed goal successfully.
    Success,
    /// Raised critical alert or protected-surface prompt.
    Alert,
}

impl FromStr for AvatarEmotion {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "neutral" => Ok(Self::Neutral),
            "idle" => Ok(Self::Idle),
            "thinking" => Ok(Self::Thinking),
            "acting" => Ok(Self::Acting),
            "speaking" => Ok(Self::Speaking),
            "waiting_human" => Ok(Self::WaitingHuman),
            "success" => Ok(Self::Success),
            "alert" => Ok(Self::Alert),
            _ => Err(()),
        }
    }
}

/// Deterministic mora/vowel timing cue extracted from VOICEVOX `AudioQuery`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MoraViseme {
    /// Vowel token (`"a"`, `"i"`, `"u"`, `"e"`, `"o"`, `"N"`, or `"sil"`).
    pub vowel: String,
    /// Start offset in milliseconds from utterance onset.
    pub start_ms: u32,
    /// Duration of this viseme segment in milliseconds.
    pub duration_ms: u32,
}

/// Real-time amplitude & frequency-band envelope extracted from `24 kHz` 16-bit PCM.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AudioEnvelopeFrame {
    /// Normalized RMS energy in `[0.0, 1.0]`.
    pub rms: f32,
    /// Normalized low-frequency amplitude proxy in `[0.0, 1.0]`.
    pub low_band: f32,
    /// Normalized mid-frequency amplitude proxy in `[0.0, 1.0]`.
    pub mid_band: f32,
    /// Normalized high-frequency amplitude proxy in `[0.0, 1.0]`.
    pub high_band: f32,
}

/// Structured cue broadcast over `jiku.sock` (`avatar.cue`) for decoupled UI subscribers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AvatarCue {
    /// Semantic emotion state.
    pub emotion: AvatarEmotion,
    /// Optional expression index (`0..=254`).
    pub expression_id: Option<u8>,
    /// Optional motion clip index (`0..=254`).
    pub motion_id: Option<u8>,
    /// Normalized gaze target `(u, v)` on `[0.0, 1.0]^2`.
    pub gaze_uv: Option<(f32, f32)>,
    /// Mora/viseme timing sequence when synthesized via VOICEVOX.
    pub mora_envelope: Vec<MoraViseme>,
    /// Instantaneous PCM RMS/band envelope when streaming audio.
    pub pcm_envelope: Option<AudioEnvelopeFrame>,
    /// True when server VAD triggered barge-in reset.
    pub interrupted: bool,
}

impl Default for AvatarCue {
    fn default() -> Self {
        Self {
            emotion: AvatarEmotion::Neutral,
            expression_id: None,
            motion_id: None,
            gaze_uv: None,
            mora_envelope: Vec::new(),
            pcm_envelope: None,
            interrupted: false,
        }
    }
}

/// Output produced by [`TtsProvider::synthesize`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TtsSynthesisResult {
    /// Provider name (`"noop"`, `"voicevox"`, `"openai_speech"`, `"espeak_ng"`, `"irodori"`).
    pub provider: String,
    /// MIME type (e.g. `"audio/wav"`, `"audio/mpeg"`).
    pub mime_type: String,
    /// Encoded audio payload (empty for `NoopTtsProvider`).
    pub audio_bytes: Vec<u8>,
    /// Extracted companion lip-sync & emotion cue.
    pub cue: AvatarCue,
    /// Total synthesized audio duration in milliseconds (`0` when empty or non-WAV).
    #[serde(default)]
    pub duration_ms: u32,
    /// Per-frame amplitude and 3-band spectral envelope timeline.
    #[serde(default)]
    pub envelope_timeline: Vec<AudioEnvelopeFrame>,
}

/// Extracted RIFF/WAVE lip-sync timeline containing frame envelopes and estimated visemes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WavLipSyncTimeline {
    /// Sample rate in Hz parsed from the `fmt ` chunk.
    pub sample_rate_hz: u32,
    /// Channel count parsed from the `fmt ` chunk.
    pub channels: u16,
    /// Total audio duration in milliseconds.
    pub duration_ms: u32,
    /// Overall energy/spectral envelope across the whole WAV stream.
    pub overall_envelope: AudioEnvelopeFrame,
    /// Per-frame (`frame_ms.max(10)`) amplitude and band energy frames.
    pub frames: Vec<AudioEnvelopeFrame>,
    /// Coalesced vowel-like viseme timeline (`"a"`, `"i"`, `"u"`, `"e"`, `"o"`, `"sil"`).
    pub visemes: Vec<MoraViseme>,
}

/// Boxed future alias returned by [`TtsProvider::synthesize`].
pub type TtsFuture<'a> =
    Pin<Box<dyn Future<Output = Result<TtsSynthesisResult, TtsError>> + Send + 'a>>;

/// Asynchronous pluggable speech synthesis interface.
pub trait TtsProvider: Send + Sync {
    /// Static identifier of this provider.
    fn provider_name(&self) -> &'static str;

    /// Synthesizes `text` and returns audio bytes paired with an [`AvatarCue`].
    fn synthesize<'a>(&'a self, text: &'a str, emotion: AvatarEmotion) -> TtsFuture<'a>;
}

/// Extracts [`MoraViseme`] timeline items from a VOICEVOX `/audio_query` JSON payload.
///
/// Time complexity: $O(M)$ where $M$ is the total number of moras across all accent phrases.
/// Space complexity: $O(M)$ for the returned viseme sequence.
pub fn extract_voicevox_mora_envelope(
    audio_query: &serde_json::Value,
) -> Result<Vec<MoraViseme>, TtsError> {
    let phrases = audio_query
        .get("accent_phrases")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| TtsError::InvalidAudioQuery("missing accent_phrases array".to_string()))?;

    let speed_scale = audio_query
        .get("speedScale")
        .and_then(serde_json::Value::as_f64)
        .filter(|s| s.is_finite() && *s > 0.0)
        .unwrap_or(1.0);
    let pre_ms = secs_to_ms(
        audio_query
            .get("prePhonemeLength")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0)
            / speed_scale,
    );

    let mut visemes = Vec::new();
    let mut cursor_ms = pre_ms;

    for phrase in phrases {
        if let Some(moras) = phrase.get("moras").and_then(serde_json::Value::as_array) {
            for mora in moras {
                let vowel = mora
                    .get("vowel")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("sil")
                    .to_string();
                let cons_s = mora
                    .get("consonant_length")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(0.0);
                let vowel_s = mora
                    .get("vowel_length")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(0.0);
                let dur_ms = secs_to_ms((cons_s + vowel_s) / speed_scale);
                if dur_ms > 0 {
                    visemes.push(MoraViseme {
                        vowel,
                        start_ms: cursor_ms,
                        duration_ms: dur_ms,
                    });
                    cursor_ms = cursor_ms.saturating_add(dur_ms);
                }
            }
        }
        if let Some(pause) = phrase.get("pause_mora").filter(|v| !v.is_null()) {
            let pause_s = pause
                .get("vowel_length")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.0);
            let dur_ms = secs_to_ms(pause_s / speed_scale);
            if dur_ms > 0 {
                visemes.push(MoraViseme {
                    vowel: "sil".to_string(),
                    start_ms: cursor_ms,
                    duration_ms: dur_ms,
                });
                cursor_ms = cursor_ms.saturating_add(dur_ms);
            }
        }
    }

    Ok(visemes)
}

/// Computes normalized RMS and 3-band temporal energy proxies over signed 16-bit
/// little-endian mono PCM frames (`24 kHz` live downlink or `16 kHz` uplink).
///
/// Time complexity: $O(N)$ where $N = \text{pcm\_le\_i16.len()} / 2$ samples.
/// Space complexity: $O(1)$ accumulators.
pub fn compute_pcm_envelope(pcm_le_i16: &[u8]) -> Result<AudioEnvelopeFrame, TtsError> {
    if !pcm_le_i16.len().is_multiple_of(2) {
        return Err(TtsError::InvalidPcmLength(pcm_le_i16.len()));
    }
    if pcm_le_i16.is_empty() {
        return Ok(AudioEnvelopeFrame {
            rms: 0.0,
            low_band: 0.0,
            mid_band: 0.0,
            high_band: 0.0,
        });
    }

    let sample_count = (pcm_le_i16.len() / 2) as f64;
    let mut sum_sq = 0.0_f64;
    let mut low_lp = 0.0_f64;
    let mut low_sq = 0.0_f64;
    let mut high_sq = 0.0_f64;
    let mut prev = 0.0_f64;

    let (chunks, _) = pcm_le_i16.as_chunks::<2>();
    for &[lo, hi] in chunks {
        let sample = i16::from_le_bytes([lo, hi]) as f64 / 32_768.0;
        sum_sq += sample * sample;

        // First-order IIR split for cheap real-time lip-sync envelope bands.
        low_lp = 0.85 * low_lp + 0.15 * sample;
        let diff = sample - prev;
        let mid_hp = sample - low_lp;
        low_sq += low_lp * low_lp;
        high_sq += diff * diff * 0.25 + mid_hp * mid_hp * 0.25;
        prev = sample;
    }

    let rms = (sum_sq / sample_count).sqrt().clamp(0.0, 1.0) as f32;
    let low_band = (low_sq / sample_count).sqrt().clamp(0.0, 1.0) as f32;
    let high_band = (high_sq / sample_count).sqrt().clamp(0.0, 1.0) as f32;
    let mid_band = (rms - 0.5 * low_band + 0.25 * high_band).clamp(0.0, 1.0);

    Ok(AudioEnvelopeFrame {
        rms,
        low_band,
        mid_band,
        high_band,
    })
}

/// Parses a RIFF/WAVE payload (`fmt ` + `data` subchunks), computes overall and
/// per-frame (`frame_ms.max(10)`) [`AudioEnvelopeFrame`] spectral envelopes, and
/// estimates coalesced vowel-like [`MoraViseme`] segments (`"a"`, `"i"`, `"u"`,
/// `"e"`, `"o"`, `"sil"`).
///
/// Supports canonical PCM (`format == 1`: 8/16/24/32-bit integer) and IEEE float
/// (`format == 3`: 32-bit float), plus `WAVE_FORMAT_EXTENSIBLE` (`0xFFFE`) wrapping
/// PCM or float, downmixing multi-channel streams to mono.
///
/// Time complexity: $O(S)$ over $S$ decoded audio samples.
/// Space complexity: $O(F)$ for $F$ envelope frames and coalesced viseme segments.
pub fn parse_wav_lipsync_timeline(
    wav_bytes: &[u8],
    frame_ms: u32,
) -> Result<WavLipSyncTimeline, TtsError> {
    let parsed = parse_wav_chunks(wav_bytes)?;
    let bytes_per_sample = usize::from(parsed.bits_per_sample / 8);
    let frame_stride = bytes_per_sample
        .checked_mul(usize::from(parsed.channels))
        .filter(|&n| n > 0)
        .ok_or_else(|| TtsError::InvalidWav("zero WAV frame stride".to_string()))?;

    let total_samples = parsed.data.len() / frame_stride;
    if total_samples == 0 {
        return Ok(WavLipSyncTimeline {
            sample_rate_hz: parsed.sample_rate_hz,
            channels: parsed.channels,
            duration_ms: 0,
            overall_envelope: zero_envelope(),
            frames: Vec::new(),
            visemes: Vec::new(),
        });
    }

    let sr = parsed.sample_rate_hz as f64;
    let duration_ms = ((total_samples as u64)
        .saturating_mul(1000)
        .saturating_add(u64::from(parsed.sample_rate_hz) / 2)
        / u64::from(parsed.sample_rate_hz)) as u32;

    let effective_frame_ms = frame_ms.max(10);
    let samples_per_frame =
        ((u64::from(parsed.sample_rate_hz) * u64::from(effective_frame_ms)) / 1000).max(1) as usize;

    let mut bank = BandFilterBank::new(sr);
    let mut overall_acc = BandAccumulator::default();
    let mut frame_acc = BandAccumulator::default();
    let mut frame_start_sample = 0_usize;

    let frame_capacity = total_samples.div_ceil(samples_per_frame);
    let mut frames = Vec::with_capacity(frame_capacity);
    let mut visemes: Vec<MoraViseme> = Vec::with_capacity(frame_capacity);

    for (sample_idx, chunk) in parsed.data.chunks_exact(frame_stride).enumerate() {
        let mono = decode_mono_sample(
            chunk,
            parsed.audio_format,
            parsed.bits_per_sample,
            parsed.channels,
            bytes_per_sample,
        );
        let (low, mid, high) = bank.process(mono);
        overall_acc.push(mono, low, mid, high);
        frame_acc.push(mono, low, mid, high);

        let is_frame_end = (sample_idx + 1 - frame_start_sample) >= samples_per_frame
            || sample_idx + 1 == total_samples;
        if is_frame_end {
            let env = frame_acc.finish();
            let start_ms =
                ((frame_start_sample as u64 * 1000) / u64::from(parsed.sample_rate_hz)) as u32;
            let end_ms = (((sample_idx + 1) as u64 * 1000 + u64::from(parsed.sample_rate_hz) / 2)
                / u64::from(parsed.sample_rate_hz)) as u32;
            let dur_ms = end_ms.saturating_sub(start_ms).max(1);
            let vowel = estimate_vowel_from_envelope(env);
            push_coalesced_viseme(
                &mut visemes,
                vowel,
                start_ms,
                start_ms.saturating_add(dur_ms),
            );
            frames.push(env);
            frame_acc = BandAccumulator::default();
            frame_start_sample = sample_idx + 1;
        }
    }

    Ok(WavLipSyncTimeline {
        sample_rate_hz: parsed.sample_rate_hz,
        channels: parsed.channels,
        duration_ms,
        overall_envelope: overall_acc.finish(),
        frames,
        visemes,
    })
}

/// Synthesizes a deterministic `24 kHz` mono 16-bit PCM RIFF/WAVE test tone
/// (or silence when `freq_hz <= 0.0`).
#[must_use]
pub fn synthesize_test_tone_wav(duration_ms: u32, freq_hz: f32) -> Vec<u8> {
    const SAMPLE_RATE: u32 = 24_000;
    let total_samples = ((u64::from(SAMPLE_RATE) * u64::from(duration_ms)) / 1000) as usize;
    let fade_samples = (SAMPLE_RATE as usize / 200).min(total_samples / 2).max(1);

    let mut pcm = Vec::with_capacity(total_samples * 2);
    for idx in 0..total_samples {
        let env = if !freq_hz.is_finite() || freq_hz <= 0.0 {
            0.0
        } else if idx < fade_samples {
            idx as f64 / fade_samples as f64
        } else if idx + fade_samples > total_samples {
            (total_samples - idx) as f64 / fade_samples as f64
        } else {
            1.0
        };
        let phase = 2.0 * PI * f64::from(freq_hz.max(0.0)) * (idx as f64 / f64::from(SAMPLE_RATE));
        let sample = (phase.sin() * 0.45 * env * 32_767.0).round() as i16;
        pcm.extend_from_slice(&sample.to_le_bytes());
    }

    encode_pcm16_mono_wav(SAMPLE_RATE, &pcm)
}

fn encode_pcm16_mono_wav(sample_rate_hz: u32, pcm_le_i16: &[u8]) -> Vec<u8> {
    let data_len = pcm_le_i16.len() as u32;
    let riff_len = 36_u32.saturating_add(data_len);
    let byte_rate = sample_rate_hz.saturating_mul(2);

    let mut out = Vec::with_capacity(44 + pcm_le_i16.len());
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&riff_len.to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16_u32.to_le_bytes());
    out.extend_from_slice(&1_u16.to_le_bytes());
    out.extend_from_slice(&1_u16.to_le_bytes());
    out.extend_from_slice(&sample_rate_hz.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&2_u16.to_le_bytes());
    out.extend_from_slice(&16_u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    out.extend_from_slice(pcm_le_i16);
    out
}

struct ParsedWavChunks<'a> {
    audio_format: u16,
    channels: u16,
    sample_rate_hz: u32,
    bits_per_sample: u16,
    data: &'a [u8],
}

fn parse_wav_chunks(wav_bytes: &[u8]) -> Result<ParsedWavChunks<'_>, TtsError> {
    if wav_bytes.len() < 12 || &wav_bytes[0..4] != b"RIFF" || &wav_bytes[8..12] != b"WAVE" {
        return Err(TtsError::InvalidWav(
            "expected RIFF/WAVE header".to_string(),
        ));
    }

    let mut fmt: Option<(u16, u16, u32, u16)> = None;
    let mut data: Option<&[u8]> = None;
    let mut offset = 12_usize;

    while offset + 8 <= wav_bytes.len() {
        let chunk_id = &wav_bytes[offset..offset + 4];
        let chunk_size = u32::from_le_bytes([
            wav_bytes[offset + 4],
            wav_bytes[offset + 5],
            wav_bytes[offset + 6],
            wav_bytes[offset + 7],
        ]) as usize;
        let payload_start = offset + 8;

        if chunk_id == b"data" {
            // Streaming stdout writers (e.g. `espeak-ng --stdout`) emit sentinel sizes
            // such as `0x7FFF_F000` or `0xFFFF_FFFF` when length is unknown upfront.
            let is_streaming_sentinel = chunk_size >= 0x7FFF_0000;
            let payload_end = match payload_start.checked_add(chunk_size) {
                Some(end) if end <= wav_bytes.len() => end,
                _ if is_streaming_sentinel => wav_bytes.len(),
                _ => {
                    return Err(TtsError::InvalidWav(format!(
                        "truncated WAV data chunk (declared {chunk_size} bytes, {} available)",
                        wav_bytes.len().saturating_sub(payload_start)
                    )));
                }
            };
            data = Some(&wav_bytes[payload_start..payload_end]);
            if fmt.is_some() {
                break;
            }
            offset = payload_end.saturating_add(chunk_size & 1);
            continue;
        }

        let payload_end = payload_start
            .checked_add(chunk_size)
            .filter(|&end| end <= wav_bytes.len())
            .ok_or_else(|| {
                TtsError::InvalidWav(format!(
                    "truncated WAV subchunk `{}`",
                    String::from_utf8_lossy(chunk_id)
                ))
            })?;

        if chunk_id == b"fmt " {
            let chunk = &wav_bytes[payload_start..payload_end];
            if chunk.len() < 16 {
                return Err(TtsError::InvalidWav(
                    "fmt chunk shorter than 16 bytes".to_string(),
                ));
            }
            let raw_format = u16::from_le_bytes([chunk[0], chunk[1]]);
            let channels = u16::from_le_bytes([chunk[2], chunk[3]]);
            let sample_rate_hz = u32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
            let bits_per_sample = u16::from_le_bytes([chunk[14], chunk[15]]);
            let audio_format = if raw_format == 0xFFFE && chunk.len() >= 40 {
                u16::from_le_bytes([chunk[24], chunk[25]])
            } else {
                raw_format
            };
            if channels == 0 || sample_rate_hz == 0 {
                return Err(TtsError::UnsupportedWavFormat {
                    format: audio_format,
                    bits_per_sample,
                    channels,
                });
            }
            let supported = matches!(
                (audio_format, bits_per_sample),
                (1, 8 | 16 | 24 | 32) | (3, 32)
            );
            if !supported {
                return Err(TtsError::UnsupportedWavFormat {
                    format: audio_format,
                    bits_per_sample,
                    channels,
                });
            }
            fmt = Some((audio_format, channels, sample_rate_hz, bits_per_sample));
            if data.is_some() {
                break;
            }
        }

        offset = payload_end.saturating_add(chunk_size & 1);
    }

    let (audio_format, channels, sample_rate_hz, bits_per_sample) =
        fmt.ok_or_else(|| TtsError::InvalidWav("missing fmt chunk".to_string()))?;
    let data = data.ok_or_else(|| TtsError::InvalidWav("missing data chunk".to_string()))?;

    Ok(ParsedWavChunks {
        audio_format,
        channels,
        sample_rate_hz,
        bits_per_sample,
        data,
    })
}

fn normalize_riff_wav_sizes(wav_bytes: &mut [u8]) {
    if wav_bytes.len() < 44 || &wav_bytes[0..4] != b"RIFF" || &wav_bytes[8..12] != b"WAVE" {
        return;
    }
    let riff_decl = u32::from_le_bytes([wav_bytes[4], wav_bytes[5], wav_bytes[6], wav_bytes[7]]);
    if riff_decl >= 0x7FFF_0000 {
        let actual = (wav_bytes.len().saturating_sub(8)) as u32;
        wav_bytes[4..8].copy_from_slice(&actual.to_le_bytes());
    }
    let mut offset = 12_usize;
    while offset + 8 <= wav_bytes.len() {
        let is_data = &wav_bytes[offset..offset + 4] == b"data";
        let size = u32::from_le_bytes([
            wav_bytes[offset + 4],
            wav_bytes[offset + 5],
            wav_bytes[offset + 6],
            wav_bytes[offset + 7],
        ]);
        let payload_start = offset + 8;
        if is_data {
            if size >= 0x7FFF_0000 {
                let actual = wav_bytes.len().saturating_sub(payload_start) as u32;
                wav_bytes[offset + 4..offset + 8].copy_from_slice(&actual.to_le_bytes());
            }
            break;
        }
        let next = payload_start
            .saturating_add(size as usize)
            .saturating_add((size as usize) & 1);
        if next <= offset || next > wav_bytes.len() {
            break;
        }
        offset = next;
    }
}

fn decode_mono_sample(
    frame_bytes: &[u8],
    audio_format: u16,
    bits_per_sample: u16,
    channels: u16,
    bytes_per_sample: usize,
) -> f64 {
    let mut sum = 0.0_f64;
    for ch in 0..usize::from(channels) {
        let start = ch * bytes_per_sample;
        let sample_bytes = &frame_bytes[start..start + bytes_per_sample];
        let val = match (audio_format, bits_per_sample) {
            (1, 8) => (f64::from(sample_bytes[0]) - 128.0) / 128.0,
            (1, 16) => f64::from(i16::from_le_bytes([sample_bytes[0], sample_bytes[1]])) / 32_768.0,
            (1, 24) => {
                f64::from(i32::from_le_bytes([
                    0,
                    sample_bytes[0],
                    sample_bytes[1],
                    sample_bytes[2],
                ])) / 2_147_483_648.0
            }
            (1, 32) => {
                f64::from(i32::from_le_bytes([
                    sample_bytes[0],
                    sample_bytes[1],
                    sample_bytes[2],
                    sample_bytes[3],
                ])) / 2_147_483_648.0
            }
            (3, 32) => {
                let f = f32::from_le_bytes([
                    sample_bytes[0],
                    sample_bytes[1],
                    sample_bytes[2],
                    sample_bytes[3],
                ]);
                if f.is_finite() {
                    f64::from(f).clamp(-1.0, 1.0)
                } else {
                    0.0
                }
            }
            _ => 0.0,
        };
        sum += val;
    }
    (sum / f64::from(channels)).clamp(-1.0, 1.0)
}

struct BandFilterBank {
    alpha_low: f64,
    alpha_mid_lp: f64,
    decay_low_hp: f64,
    decay_high_hp: f64,
    low_lp1: f64,
    low_lp2: f64,
    prev_in_mid: f64,
    mid_hp1: f64,
    mid_lp1: f64,
    mid_lp2: f64,
    prev_in_high: f64,
    high_hp1: f64,
    prev_hp1: f64,
    high_hp2: f64,
}

impl BandFilterBank {
    fn new(sample_rate_hz: f64) -> Self {
        let low_cut = 500.0_f64.min(0.40 * sample_rate_hz);
        let mid_cut = 2200.0_f64.min(0.45 * sample_rate_hz).max(low_cut + 50.0);
        Self {
            alpha_low: 1.0 - (-2.0 * PI * low_cut / sample_rate_hz).exp(),
            alpha_mid_lp: 1.0 - (-2.0 * PI * mid_cut / sample_rate_hz).exp(),
            decay_low_hp: (-2.0 * PI * low_cut / sample_rate_hz).exp(),
            decay_high_hp: (-2.0 * PI * mid_cut / sample_rate_hz).exp(),
            low_lp1: 0.0,
            low_lp2: 0.0,
            prev_in_mid: 0.0,
            mid_hp1: 0.0,
            mid_lp1: 0.0,
            mid_lp2: 0.0,
            prev_in_high: 0.0,
            high_hp1: 0.0,
            prev_hp1: 0.0,
            high_hp2: 0.0,
        }
    }

    #[inline]
    fn process(&mut self, sample: f64) -> (f64, f64, f64) {
        self.low_lp1 += self.alpha_low * (sample - self.low_lp1);
        self.low_lp2 += self.alpha_low * (self.low_lp1 - self.low_lp2);

        self.mid_hp1 = self.decay_low_hp * (self.mid_hp1 + sample - self.prev_in_mid);
        self.prev_in_mid = sample;
        self.mid_lp1 += self.alpha_mid_lp * (self.mid_hp1 - self.mid_lp1);
        self.mid_lp2 += self.alpha_mid_lp * (self.mid_lp1 - self.mid_lp2);

        self.high_hp1 = self.decay_high_hp * (self.high_hp1 + sample - self.prev_in_high);
        self.prev_in_high = sample;
        self.high_hp2 = self.decay_high_hp * (self.high_hp2 + self.high_hp1 - self.prev_hp1);
        self.prev_hp1 = self.high_hp1;

        (self.low_lp2, self.mid_lp2, self.high_hp2)
    }
}

#[derive(Default)]
struct BandAccumulator {
    count: usize,
    sum_sq: f64,
    low_sq: f64,
    mid_sq: f64,
    high_sq: f64,
}

impl BandAccumulator {
    #[inline]
    fn push(&mut self, sample: f64, low: f64, mid: f64, high: f64) {
        self.count += 1;
        self.sum_sq += sample * sample;
        self.low_sq += low * low;
        self.mid_sq += mid * mid;
        self.high_sq += high * high;
    }

    fn finish(&self) -> AudioEnvelopeFrame {
        if self.count == 0 {
            return zero_envelope();
        }
        let n = self.count as f64;
        AudioEnvelopeFrame {
            rms: (self.sum_sq / n).sqrt().clamp(0.0, 1.0) as f32,
            low_band: (self.low_sq / n).sqrt().clamp(0.0, 1.0) as f32,
            mid_band: (self.mid_sq / n).sqrt().clamp(0.0, 1.0) as f32,
            high_band: (self.high_sq / n).sqrt().clamp(0.0, 1.0) as f32,
        }
    }
}

const fn zero_envelope() -> AudioEnvelopeFrame {
    AudioEnvelopeFrame {
        rms: 0.0,
        low_band: 0.0,
        mid_band: 0.0,
        high_band: 0.0,
    }
}

fn estimate_vowel_from_envelope(env: AudioEnvelopeFrame) -> &'static str {
    if env.rms < 0.015 {
        return "sil";
    }
    let total = env.low_band + env.mid_band + env.high_band + 1e-6;
    let low = env.low_band / total;
    let mid = env.mid_band / total;
    let high = env.high_band / total;

    if high >= 0.40 || (high > low && high >= 0.32) {
        "i"
    } else if low >= 0.48 && high < 0.20 {
        "u"
    } else if mid >= 0.38 && env.rms >= 0.12 && low >= 0.22 {
        "a"
    } else if mid >= 0.34 && high >= 0.24 {
        "e"
    } else if low >= 0.34 && mid >= 0.24 {
        "o"
    } else if mid >= low {
        "a"
    } else {
        "o"
    }
}

fn push_coalesced_viseme(visemes: &mut Vec<MoraViseme>, vowel: &str, start_ms: u32, end_ms: u32) {
    if let Some(last) = visemes.last_mut()
        && last.vowel == vowel
    {
        last.duration_ms = end_ms.saturating_sub(last.start_ms).max(1);
        return;
    }
    visemes.push(MoraViseme {
        vowel: vowel.to_string(),
        start_ms,
        duration_ms: end_ms.saturating_sub(start_ms).max(1),
    });
}

fn enrich_result_from_wav(
    provider: &str,
    mime_type: String,
    audio_bytes: Vec<u8>,
    emotion: AvatarEmotion,
    mut explicit_moras: Vec<MoraViseme>,
    frame_ms: u32,
) -> TtsSynthesisResult {
    let mut duration_ms = 0_u32;
    let mut envelope_timeline = Vec::new();
    let mut pcm_envelope = None;

    if audio_bytes.starts_with(b"RIFF")
        && let Ok(timeline) = parse_wav_lipsync_timeline(&audio_bytes, frame_ms)
    {
        duration_ms = timeline.duration_ms;
        pcm_envelope = Some(timeline.overall_envelope);
        envelope_timeline = timeline.frames;
        if explicit_moras.is_empty() {
            explicit_moras = timeline.visemes;
        }
    }

    TtsSynthesisResult {
        provider: provider.to_string(),
        mime_type,
        audio_bytes,
        cue: AvatarCue {
            emotion,
            mora_envelope: explicit_moras,
            pcm_envelope,
            ..AvatarCue::default()
        },
        duration_ms,
        envelope_timeline,
    }
}

/// Deterministic silent/no-op TTS provider that records utterances in memory.
#[derive(Debug, Default, Clone)]
pub struct NoopTtsProvider {
    recorded: Arc<Mutex<Vec<(String, AvatarEmotion)>>>,
}

impl NoopTtsProvider {
    /// Returns every `(text, emotion)` spoken during this provider's lifetime.
    #[must_use]
    pub fn recorded_utterances(&self) -> Vec<(String, AvatarEmotion)> {
        self.recorded
            .lock()
            .expect("noop tts mutex poisoned")
            .clone()
    }
}

impl TtsProvider for NoopTtsProvider {
    fn provider_name(&self) -> &'static str {
        "noop"
    }

    fn synthesize<'a>(&'a self, text: &'a str, emotion: AvatarEmotion) -> TtsFuture<'a> {
        let owned = text.to_string();
        let log = Arc::clone(&self.recorded);
        Box::pin(async move {
            log.lock()
                .expect("noop tts mutex poisoned")
                .push((owned, emotion));
            Ok(TtsSynthesisResult {
                provider: "noop".to_string(),
                mime_type: "audio/wav".to_string(),
                audio_bytes: Vec::new(),
                cue: AvatarCue {
                    emotion,
                    ..AvatarCue::default()
                },
                duration_ms: 0,
                envelope_timeline: Vec::new(),
            })
        })
    }
}

/// VOICEVOX Engine HTTP client (`/audio_query` + `/synthesis`).
#[derive(Debug, Clone)]
pub struct VoicevoxProvider {
    client: reqwest::Client,
    endpoint: String,
    speaker_id: u32,
}

impl VoicevoxProvider {
    /// Builds a VOICEVOX client targeting `endpoint` (e.g. `http://127.0.0.1:50021`).
    pub fn new(endpoint: impl Into<String>, speaker_id: u32) -> Result<Self, TtsError> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(15))
            .build()?;
        Ok(Self {
            client,
            endpoint: endpoint.into().trim_end_matches('/').to_string(),
            speaker_id,
        })
    }
}

impl TtsProvider for VoicevoxProvider {
    fn provider_name(&self) -> &'static str {
        "voicevox"
    }

    fn synthesize<'a>(&'a self, text: &'a str, emotion: AvatarEmotion) -> TtsFuture<'a> {
        Box::pin(async move {
            let query_resp = self
                .client
                .post(format!("{}/audio_query", self.endpoint))
                .query(&[
                    ("speaker", self.speaker_id.to_string()),
                    ("text", text.to_string()),
                ])
                .send()
                .await?;
            if !query_resp.status().is_success() {
                let status = query_resp.status().as_u16();
                let body = query_resp.text().await.unwrap_or_default();
                return Err(TtsError::RemoteStatus { status, body });
            }
            let audio_query: serde_json::Value = query_resp.json().await?;
            let mora_envelope = extract_voicevox_mora_envelope(&audio_query)?;

            let synth_resp = self
                .client
                .post(format!("{}/synthesis", self.endpoint))
                .query(&[("speaker", self.speaker_id.to_string())])
                .json(&audio_query)
                .send()
                .await?;
            if !synth_resp.status().is_success() {
                let status = synth_resp.status().as_u16();
                let body = synth_resp.text().await.unwrap_or_default();
                return Err(TtsError::RemoteStatus { status, body });
            }
            let audio_bytes = synth_resp.bytes().await?.to_vec();

            Ok(enrich_result_from_wav(
                "voicevox",
                "audio/wav".to_string(),
                audio_bytes,
                emotion,
                mora_envelope,
                20,
            ))
        })
    }
}

/// OpenAI-compatible `/audio/speech` TTS provider.
#[derive(Debug, Clone)]
pub struct OpenAiSpeechProvider {
    client: reqwest::Client,
    base_url: String,
    api_key: String,
    model: String,
    voice: String,
}

impl OpenAiSpeechProvider {
    /// Builds an OpenAI `/audio/speech` client.
    pub fn new(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
        voice: impl Into<String>,
    ) -> Result<Self, TtsError> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(20))
            .build()?;
        Ok(Self {
            client,
            base_url: base_url.into().trim_end_matches('/').to_string(),
            api_key: api_key.into(),
            model: model.into(),
            voice: voice.into(),
        })
    }
}

impl TtsProvider for OpenAiSpeechProvider {
    fn provider_name(&self) -> &'static str {
        "openai_speech"
    }

    fn synthesize<'a>(&'a self, text: &'a str, emotion: AvatarEmotion) -> TtsFuture<'a> {
        Box::pin(async move {
            let mut req = self
                .client
                .post(format!("{}/audio/speech", self.base_url))
                .json(&serde_json::json!({
                    "model": self.model,
                    "voice": self.voice,
                    "input": text,
                    "response_format": "wav",
                }));
            if !self.api_key.is_empty() {
                req = req.bearer_auth(&self.api_key);
            }
            let resp = req.send().await?;
            if !resp.status().is_success() {
                let status = resp.status().as_u16();
                let body = resp.text().await.unwrap_or_default();
                return Err(TtsError::RemoteStatus { status, body });
            }
            let mime_type = resp
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                .map(|s| {
                    s.split(';')
                        .next()
                        .unwrap_or("audio/wav")
                        .trim()
                        .to_string()
                })
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "audio/wav".to_string());
            let audio_bytes = resp.bytes().await?.to_vec();
            Ok(enrich_result_from_wav(
                "openai_speech",
                mime_type,
                audio_bytes,
                emotion,
                Vec::new(),
                20,
            ))
        })
    }
}

/// Local `espeak-ng` CLI speech synthesizer with emotion-mapped prosody and an
/// optional deterministic synthetic WAV fallback for headless CI environments.
#[derive(Debug, Clone)]
pub struct EspeakNgProvider {
    binary_path: String,
    voice: String,
    speed_wpm: u16,
    frame_ms: u32,
    allow_synthetic_fallback: bool,
    force_synthetic: bool,
}

impl Default for EspeakNgProvider {
    fn default() -> Self {
        Self::new("en", 165)
    }
}

impl EspeakNgProvider {
    /// Builds an `espeak-ng` provider targeting `/usr/bin/espeak-ng` (or `espeak-ng` on `PATH`).
    #[must_use]
    pub fn new(voice: impl Into<String>, speed_wpm: u16) -> Self {
        let default_bin = if Path::new("/usr/bin/espeak-ng").exists() {
            "/usr/bin/espeak-ng"
        } else {
            "espeak-ng"
        };
        let voice_str = voice.into();
        Self {
            binary_path: default_bin.to_string(),
            voice: if voice_str.trim().is_empty() {
                "en".to_string()
            } else {
                voice_str
            },
            speed_wpm: speed_wpm.clamp(80, 450),
            frame_ms: 20,
            allow_synthetic_fallback: false,
            force_synthetic: false,
        }
    }

    /// Builds a deterministic in-process synthetic WAV provider (no `espeak-ng` subprocess).
    #[must_use]
    pub fn new_synthetic() -> Self {
        Self {
            force_synthetic: true,
            allow_synthetic_fallback: true,
            ..Self::default()
        }
    }

    /// Overrides the `espeak-ng` executable path.
    #[must_use]
    pub fn with_binary_path(mut self, path: impl Into<String>) -> Self {
        self.binary_path = path.into();
        self
    }

    /// Enables or disables deterministic synthetic WAV fallback when `espeak-ng` is unavailable.
    #[must_use]
    pub fn with_synthetic_fallback(mut self, allow: bool) -> Self {
        self.allow_synthetic_fallback = allow;
        self
    }

    /// Overrides the lip-sync envelope frame interval in milliseconds (minimum 10 ms).
    #[must_use]
    pub fn with_frame_ms(mut self, frame_ms: u32) -> Self {
        self.frame_ms = frame_ms.max(10);
        self
    }

    fn synthesize_sync(
        &self,
        text: &str,
        emotion: AvatarEmotion,
    ) -> Result<TtsSynthesisResult, TtsError> {
        let mut wav_bytes = if text.trim().is_empty() {
            synthesize_test_tone_wav(80, 0.0)
        } else if self.force_synthetic {
            synthesize_deterministic_speech_wav(text, emotion)
        } else {
            let (wpm, pitch, amp) = espeak_prosody_for_emotion(emotion, self.speed_wpm);
            match Command::new(&self.binary_path)
                .arg("--stdout")
                .arg("-v")
                .arg(&self.voice)
                .arg("-s")
                .arg(wpm.to_string())
                .arg("-p")
                .arg(pitch.to_string())
                .arg("-a")
                .arg(amp.to_string())
                .arg("--")
                .arg(text)
                .output()
            {
                Ok(out) if out.status.success() && !out.stdout.is_empty() => {
                    let mut raw = out.stdout;
                    normalize_riff_wav_sizes(&mut raw);
                    raw
                }
                Ok(out) => {
                    if self.allow_synthetic_fallback {
                        synthesize_deterministic_speech_wav(text, emotion)
                    } else {
                        let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
                        return Err(TtsError::ProcessFailed {
                            program: self.binary_path.clone(),
                            detail: if stderr.is_empty() {
                                format!("exit status {}", out.status)
                            } else {
                                stderr
                            },
                        });
                    }
                }
                Err(err) => {
                    if self.allow_synthetic_fallback {
                        synthesize_deterministic_speech_wav(text, emotion)
                    } else {
                        return Err(TtsError::ProcessFailed {
                            program: self.binary_path.clone(),
                            detail: err.to_string(),
                        });
                    }
                }
            }
        };

        normalize_riff_wav_sizes(&mut wav_bytes);
        let timeline = parse_wav_lipsync_timeline(&wav_bytes, self.frame_ms)?;
        Ok(TtsSynthesisResult {
            provider: "espeak_ng".to_string(),
            mime_type: "audio/wav".to_string(),
            audio_bytes: wav_bytes,
            cue: AvatarCue {
                emotion,
                mora_envelope: timeline.visemes,
                pcm_envelope: Some(timeline.overall_envelope),
                ..AvatarCue::default()
            },
            duration_ms: timeline.duration_ms,
            envelope_timeline: timeline.frames,
        })
    }
}

impl TtsProvider for EspeakNgProvider {
    fn provider_name(&self) -> &'static str {
        "espeak_ng"
    }

    fn synthesize<'a>(&'a self, text: &'a str, emotion: AvatarEmotion) -> TtsFuture<'a> {
        let cloned = self.clone();
        let owned = text.to_string();
        Box::pin(async move { cloned.synthesize_sync(&owned, emotion) })
    }
}

fn espeak_prosody_for_emotion(emotion: AvatarEmotion, base_wpm: u16) -> (u16, u8, u8) {
    let clamped = base_wpm.clamp(80, 450);
    match emotion {
        AvatarEmotion::Neutral | AvatarEmotion::Idle => (clamped, 50, 100),
        AvatarEmotion::Thinking => (clamped.saturating_sub(15).max(80), 45, 90),
        AvatarEmotion::Acting => (clamped.saturating_add(10).min(450), 54, 105),
        AvatarEmotion::Speaking => (clamped, 52, 105),
        AvatarEmotion::WaitingHuman => (clamped.saturating_sub(10).max(80), 48, 95),
        AvatarEmotion::Success => (clamped.saturating_add(8).min(450), 62, 110),
        AvatarEmotion::Alert => (clamped.saturating_add(15).min(450), 68, 125),
    }
}

fn synthesize_deterministic_speech_wav(text: &str, emotion: AvatarEmotion) -> Vec<u8> {
    const SAMPLE_RATE: u32 = 24_000;
    let char_count = text.chars().filter(|c| !c.is_whitespace()).count().max(1);
    let syllables = char_count.clamp(2, 12);
    let syllable_samples = (SAMPLE_RATE as usize * 70) / 1000;
    let pause_samples = (SAMPLE_RATE as usize * 18) / 1000;
    let stride = syllable_samples + pause_samples;
    let total_samples = syllables * stride;
    let base_hz = match emotion {
        AvatarEmotion::Alert => 290.0_f64,
        AvatarEmotion::Success => 255.0_f64,
        AvatarEmotion::Thinking => 195.0_f64,
        _ => 220.0_f64,
    };

    let mut pcm = Vec::with_capacity(total_samples * 2);
    for (syl_idx, ch) in text
        .chars()
        .filter(|c| !c.is_whitespace())
        .chain(std::iter::repeat('a'))
        .take(syllables)
        .enumerate()
    {
        let code = ch as u32;
        let formant_hz = match (code + syl_idx as u32) % 4 {
            0 => base_hz,
            1 => 1050.0,
            2 => 2700.0,
            _ => 720.0,
        };
        for s in 0..stride {
            let amp = if s >= syllable_samples {
                0.0
            } else {
                let norm = s as f64 / syllable_samples as f64;
                (norm * PI).sin() * 0.42
            };
            let t = (syl_idx * stride + s) as f64 / f64::from(SAMPLE_RATE);
            let wave = (2.0 * PI * formant_hz * t).sin() + 0.35 * (2.0 * PI * base_hz * t).sin();
            let sample = (wave * amp * 24_000.0).clamp(-32_767.0, 32_767.0).round() as i16;
            pcm.extend_from_slice(&sample.to_le_bytes());
        }
    }
    encode_pcm16_mono_wav(SAMPLE_RATE, &pcm)
}

/// HTTP speech synthesis provider targeting OpenAI-compatible (`/v1/audio/speech`)
/// or local `Irodori-TTS` (`/synthesize`) inference bridges.
#[derive(Debug, Clone)]
pub struct IrodoriHttpProvider {
    client: reqwest::Client,
    endpoint: String,
    api_key: String,
    model: String,
    voice: Option<String>,
    caption: Option<String>,
    prepend_emotion_hint: bool,
    frame_ms: u32,
}

impl IrodoriHttpProvider {
    /// Builds an `Irodori-TTS` HTTP client from a base URL or full endpoint path.
    pub fn new(endpoint: impl Into<String>) -> Result<Self, TtsError> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()?;
        Ok(Self {
            client,
            endpoint: endpoint.into().trim_end_matches('/').to_string(),
            api_key: String::new(),
            model: "irodori-tts-v4.1-small".to_string(),
            voice: None,
            caption: None,
            prepend_emotion_hint: true,
            frame_ms: 20,
        })
    }

    /// Sets an optional Bearer token for gated inference proxies.
    #[must_use]
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = api_key.into();
        self
    }

    /// Sets the model identifier sent to `/v1/audio/speech` bridges.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Sets an optional voice or reference-speaker identifier.
    #[must_use]
    pub fn with_voice(mut self, voice: impl Into<String>) -> Self {
        let v = voice.into();
        self.voice = (!v.trim().is_empty()).then_some(v);
        self
    }

    /// Sets an optional style caption passed alongside the synthesis request.
    #[must_use]
    pub fn with_caption(mut self, caption: impl Into<String>) -> Self {
        let c = caption.into();
        self.caption = (!c.trim().is_empty()).then_some(c);
        self
    }

    /// Controls whether `Irodori-TTS` emoji emotion hints are prepended to text.
    #[must_use]
    pub fn with_emotion_hints(mut self, enabled: bool) -> Self {
        self.prepend_emotion_hint = enabled;
        self
    }

    /// Overrides the lip-sync envelope frame interval in milliseconds (minimum 10 ms).
    #[must_use]
    pub fn with_frame_ms(mut self, frame_ms: u32) -> Self {
        self.frame_ms = frame_ms.max(10);
        self
    }

    fn resolve_target_url(&self) -> (String, bool) {
        if self.endpoint.ends_with("/synthesize") {
            (self.endpoint.clone(), true)
        } else if self.endpoint.ends_with("/audio/speech") {
            (self.endpoint.clone(), false)
        } else if self.endpoint.ends_with("/v1") {
            (format!("{}/audio/speech", self.endpoint), false)
        } else {
            (format!("{}/v1/audio/speech", self.endpoint), false)
        }
    }
}

/// Maps an [`AvatarEmotion`] into an optional `Irodori-TTS` emoji style prefix
/// matching `Irodori-TTS-v4.1-Small/EMOJI_ANNOTATIONS.md`.
#[must_use]
pub fn format_irodori_input(text: &str, emotion: AvatarEmotion, prepend_hint: bool) -> String {
    let trimmed = text.trim();
    if !prepend_hint || trimmed.is_empty() {
        return trimmed.to_string();
    }
    let hint = match emotion {
        AvatarEmotion::Thinking => "🤔",
        AvatarEmotion::Acting => "💪",
        AvatarEmotion::WaitingHuman => "😟",
        AvatarEmotion::Success => "😊",
        AvatarEmotion::Alert => "😰",
        AvatarEmotion::Neutral | AvatarEmotion::Idle | AvatarEmotion::Speaking => "",
    };
    if hint.is_empty() || trimmed.starts_with(hint) {
        trimmed.to_string()
    } else {
        format!("{hint} {trimmed}")
    }
}

impl TtsProvider for IrodoriHttpProvider {
    fn provider_name(&self) -> &'static str {
        "irodori"
    }

    fn synthesize<'a>(&'a self, text: &'a str, emotion: AvatarEmotion) -> TtsFuture<'a> {
        Box::pin(async move {
            let (url, is_native_synthesize) = self.resolve_target_url();
            let input_text = format_irodori_input(text, emotion, self.prepend_emotion_hint);
            let mut body = if is_native_synthesize {
                serde_json::json!({
                    "text": input_text,
                    "response_format": "wav",
                    "emotion": emotion,
                })
            } else {
                serde_json::json!({
                    "model": self.model,
                    "input": input_text,
                    "voice": self.voice.as_deref().unwrap_or("default"),
                    "response_format": "wav",
                })
            };
            if let Some(obj) = body.as_object_mut() {
                if is_native_synthesize && let Some(voice) = &self.voice {
                    obj.insert("voice".to_string(), serde_json::json!(voice));
                }
                if let Some(caption) = &self.caption {
                    obj.insert("caption".to_string(), serde_json::json!(caption));
                }
            }

            let mut req = self.client.post(url).json(&body);
            if !self.api_key.is_empty() {
                req = req.bearer_auth(&self.api_key);
            }
            let resp = req.send().await?;
            if !resp.status().is_success() {
                let status = resp.status().as_u16();
                let err_body = resp.text().await.unwrap_or_default();
                return Err(TtsError::RemoteStatus {
                    status,
                    body: err_body,
                });
            }
            let mime_type = resp
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                .map(|s| {
                    s.split(';')
                        .next()
                        .unwrap_or("audio/wav")
                        .trim()
                        .to_string()
                })
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "audio/wav".to_string());
            let mut audio_bytes = resp.bytes().await?.to_vec();
            normalize_riff_wav_sizes(&mut audio_bytes);

            if audio_bytes.starts_with(b"RIFF") {
                let timeline = parse_wav_lipsync_timeline(&audio_bytes, self.frame_ms)?;
                Ok(TtsSynthesisResult {
                    provider: "irodori".to_string(),
                    mime_type,
                    audio_bytes,
                    cue: AvatarCue {
                        emotion,
                        mora_envelope: timeline.visemes,
                        pcm_envelope: Some(timeline.overall_envelope),
                        ..AvatarCue::default()
                    },
                    duration_ms: timeline.duration_ms,
                    envelope_timeline: timeline.frames,
                })
            } else {
                Ok(TtsSynthesisResult {
                    provider: "irodori".to_string(),
                    mime_type,
                    audio_bytes,
                    cue: AvatarCue {
                        emotion,
                        ..AvatarCue::default()
                    },
                    duration_ms: 0,
                    envelope_timeline: Vec::new(),
                })
            }
        })
    }
}

/// Audio backend selection for [`AudioPlaybackSink`].
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioSinkKind {
    /// Automatically prefer `pw-play -` and fall back to `aplay -q -`.
    #[default]
    Auto,
    /// PipeWire CLI player (`pw-play -`).
    PwPlay,
    /// ALSA CLI player (`aplay -q -`).
    Aplay,
    /// Disabled / silent playback sink (always returns `Ok(false)`).
    Silent,
    /// Custom whitespace-delimited command accepting WAV bytes on stdin.
    Command(String),
}

/// Non-blocking, barge-in-interruptible WAV playback sink backed by an atomic
/// generation counter and tracked subprocess handle.
#[derive(Debug, Clone)]
pub struct AudioPlaybackSink {
    kind: AudioSinkKind,
    active_child: Arc<Mutex<Option<(u64, Child)>>>,
    generation: Arc<AtomicU64>,
}

impl Default for AudioPlaybackSink {
    fn default() -> Self {
        Self::new(AudioSinkKind::Auto)
    }
}

impl AudioPlaybackSink {
    /// Creates an [`AudioPlaybackSink`] configured with `kind`.
    #[must_use]
    pub fn new(kind: AudioSinkKind) -> Self {
        Self {
            kind,
            active_child: Arc::new(Mutex::new(None)),
            generation: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Returns the configured [`AudioSinkKind`].
    #[must_use]
    pub fn kind(&self) -> &AudioSinkKind {
        &self.kind
    }

    /// Returns the monotonic barge-in / playback generation counter.
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::SeqCst)
    }

    /// Returns `true` if a playback subprocess is currently active.
    #[must_use]
    pub fn is_playing(&self) -> bool {
        let mut guard = self
            .active_child
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some((_, child)) = guard.as_mut() else {
            return false;
        };
        match child.try_wait() {
            Ok(None) => true,
            _ => {
                *guard = None;
                false
            }
        }
    }

    /// Immediately aborts any active playback subprocess (for Live barge-in).
    /// Returns `true` if a running child process was killed.
    pub fn stop_playback(&self) -> bool {
        self.generation.fetch_add(1, Ordering::SeqCst);
        self.kill_active_child()
    }

    /// Streams `wav_bytes` into the configured audio player and waits for exit,
    /// aborting immediately and returning `Ok(false)` if [`Self::stop_playback`]
    /// or another playback call supersedes this generation.
    ///
    /// Returns `Ok(false)` cleanly when [`AudioSinkKind::Silent`] is configured,
    /// `wav_bytes` is empty, or no playback binary is found on the host.
    pub fn play_wav_interruptible(&self, wav_bytes: &[u8]) -> Result<bool, TtsError> {
        if wav_bytes.is_empty() || self.kind == AudioSinkKind::Silent {
            return Ok(false);
        }

        let my_gen = self.generation.fetch_add(1, Ordering::SeqCst) + 1;
        self.kill_active_child();

        let Some((program, mut child)) = self.spawn_sink_child()? else {
            return Ok(false);
        };
        let mut stdin = child.stdin.take();

        {
            let mut guard = self
                .active_child
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if self.generation.load(Ordering::SeqCst) != my_gen {
                drop(guard);
                let _ = child.kill();
                let _ = child.wait();
                return Ok(false);
            }
            *guard = Some((my_gen, child));
        }

        if let Some(mut pipe) = stdin.take() {
            for chunk in wav_bytes.chunks(8192) {
                if self.generation.load(Ordering::SeqCst) != my_gen {
                    self.kill_matching_child(my_gen);
                    return Ok(false);
                }
                if let Err(err) = pipe.write_all(chunk) {
                    if self.generation.load(Ordering::SeqCst) != my_gen {
                        self.kill_matching_child(my_gen);
                        return Ok(false);
                    }
                    if err.kind() == io::ErrorKind::BrokenPipe {
                        break;
                    }
                    self.kill_matching_child(my_gen);
                    return Err(TtsError::Io(err));
                }
            }
            let _ = pipe.flush();
        }

        loop {
            if self.generation.load(Ordering::SeqCst) != my_gen {
                self.kill_matching_child(my_gen);
                return Ok(false);
            }

            let polled = {
                let mut guard = self
                    .active_child
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                match guard.as_mut() {
                    Some((child_gen, child)) if *child_gen == my_gen => match child.try_wait() {
                        Ok(Some(status)) => {
                            *guard = None;
                            Some(Ok(status))
                        }
                        Ok(None) => None,
                        Err(err) => {
                            *guard = None;
                            Some(Err(err))
                        }
                    },
                    _ => {
                        return Ok(false);
                    }
                }
            };

            match polled {
                Some(Ok(status)) => {
                    if self.generation.load(Ordering::SeqCst) != my_gen {
                        return Ok(false);
                    }
                    if status.success() {
                        return Ok(true);
                    }
                    return Err(TtsError::ProcessFailed {
                        program,
                        detail: format!("exited with status {status}"),
                    });
                }
                Some(Err(err)) => {
                    if self.generation.load(Ordering::SeqCst) != my_gen {
                        return Ok(false);
                    }
                    return Err(TtsError::Io(err));
                }
                None => thread::sleep(Duration::from_millis(10)),
            }
        }
    }

    fn kill_active_child(&self) -> bool {
        let taken = {
            let mut guard = self
                .active_child
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            guard.take()
        };
        if let Some((_, mut child)) = taken {
            let was_running = matches!(child.try_wait(), Ok(None));
            let _ = child.kill();
            let _ = child.wait();
            was_running
        } else {
            false
        }
    }

    fn kill_matching_child(&self, expected_gen: u64) {
        let taken = {
            let mut guard = self
                .active_child
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            match guard.as_ref() {
                Some((child_gen, _)) if *child_gen == expected_gen => guard.take(),
                _ => None,
            }
        };
        if let Some((_, mut child)) = taken {
            let _ = child.kill();
            let _ = child.wait();
        }
    }

    fn spawn_sink_child(&self) -> Result<Option<(String, Child)>, TtsError> {
        match &self.kind {
            AudioSinkKind::Silent => Ok(None),
            AudioSinkKind::PwPlay => try_spawn_player("pw-play", &["-"]),
            AudioSinkKind::Aplay => try_spawn_player("aplay", &["-q", "-"]),
            AudioSinkKind::Auto => match try_spawn_player("pw-play", &["-"])? {
                Some(pair) => Ok(Some(pair)),
                None => try_spawn_player("aplay", &["-q", "-"]),
            },
            AudioSinkKind::Command(cmd) => {
                let parts: Vec<&str> = cmd.split_whitespace().collect();
                let Some((prog, args)) = parts.split_first() else {
                    return Ok(None);
                };
                try_spawn_player(prog, args)
            }
        }
    }
}

fn try_spawn_player(program: &str, args: &[&str]) -> Result<Option<(String, Child)>, TtsError> {
    match Command::new(program)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
    {
        Ok(child) => Ok(Some((program.to_string(), child))),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(err) => Err(TtsError::Io(err)),
    }
}

/// Structured desktop notification payload emitted by Lua policy hooks or task completions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NotificationSpec {
    /// Notification summary / title.
    pub summary: String,
    /// Optional detailed message body.
    pub body: String,
    /// Urgency level (`"low"`, `"normal"`, `"critical"`).
    pub urgency: String,
}

/// Lightweight D-Bus/`notify-send` notification dispatcher that degrades cleanly
/// when headless or when `enabled == false`.
#[derive(Debug, Clone)]
pub struct DesktopNotifier {
    /// Whether external desktop notification delivery is enabled.
    pub enabled: bool,
    /// Application name shown in notification daemon.
    pub app_name: String,
}

impl Default for DesktopNotifier {
    fn default() -> Self {
        Self {
            enabled: false,
            app_name: "Jiku".to_string(),
        }
    }
}

impl DesktopNotifier {
    /// Dispatches a notification when `enabled` and a D-Bus session exists.
    /// Returns `Ok(true)` if handed to `notify-send`, or `Ok(false)` when skipped cleanly.
    pub fn dispatch(&self, spec: &NotificationSpec) -> io::Result<bool> {
        if !self.enabled || std::env::var_os("DBUS_SESSION_BUS_ADDRESS").is_none() {
            return Ok(false);
        }
        let urgency = match spec.urgency.as_str() {
            "low" | "critical" => spec.urgency.as_str(),
            _ => "normal",
        };
        match Command::new("notify-send")
            .arg("-a")
            .arg(&self.app_name)
            .arg("-u")
            .arg(urgency)
            .arg("--")
            .arg(&spec.summary)
            .arg(&spec.body)
            .output()
        {
            Ok(out) => Ok(out.status.success()),
            Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(false),
            Err(err) => Err(err),
        }
    }
}

fn secs_to_ms(secs: f64) -> u32 {
    if !secs.is_finite() || secs <= 0.0 {
        return 0;
    }
    (secs * 1000.0).round() as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_voicevox_mora_timing_envelope() {
        let query = serde_json::json!({
            "speedScale": 1.0,
            "prePhonemeLength": 0.10,
            "accent_phrases": [
                {
                    "moras": [
                        { "vowel": "o", "consonant_length": 0.0, "vowel_length": 0.12 },
                        { "vowel": "a", "consonant_length": 0.05, "vowel_length": 0.10 }
                    ],
                    "pause_mora": { "vowel": "pau", "vowel_length": 0.08 }
                }
            ]
        });

        let visemes = extract_voicevox_mora_envelope(&query).unwrap();
        assert_eq!(visemes.len(), 3);
        assert_eq!(
            visemes[0],
            MoraViseme {
                vowel: "o".to_string(),
                start_ms: 100,
                duration_ms: 120
            }
        );
        assert_eq!(
            visemes[1],
            MoraViseme {
                vowel: "a".to_string(),
                start_ms: 220,
                duration_ms: 150
            }
        );
        assert_eq!(
            visemes[2],
            MoraViseme {
                vowel: "sil".to_string(),
                start_ms: 370,
                duration_ms: 80
            }
        );
    }

    #[test]
    fn computes_pcm_envelope_for_silence_and_tone() {
        let silence = vec![0_u8; 480];
        let env_sil = compute_pcm_envelope(&silence).unwrap();
        assert_eq!(env_sil.rms, 0.0);

        // Construct a 16-bit LE alternating wave at ~half scale.
        let mut tone = Vec::with_capacity(960);
        for idx in 0..480 {
            let sample: i16 = if idx % 2 == 0 { 16_384 } else { -16_384 };
            tone.extend_from_slice(&sample.to_le_bytes());
        }
        let env_tone = compute_pcm_envelope(&tone).unwrap();
        assert!((env_tone.rms - 0.5).abs() < 0.02);
        assert!(env_tone.high_band > env_tone.low_band);
        assert!(compute_pcm_envelope(&[0_u8; 3]).is_err());
    }

    fn block_on_ready<F: Future>(fut: F) -> F::Output {
        use std::task::{Context, Poll, Waker};

        let mut cx = Context::from_waker(Waker::noop());
        let mut pinned = std::pin::pin!(fut);
        match pinned.as_mut().poll(&mut cx) {
            Poll::Ready(out) => out,
            Poll::Pending => panic!("expected synchronous/ready future in unit test"),
        }
    }

    fn build_custom_wav(
        audio_format: u16,
        channels: u16,
        sample_rate_hz: u32,
        bits_per_sample: u16,
        prefix_junk: bool,
        data: &[u8],
    ) -> Vec<u8> {
        let block_align = channels * (bits_per_sample / 8);
        let byte_rate = sample_rate_hz * u32::from(block_align);
        let mut chunks = Vec::new();
        if prefix_junk {
            chunks.extend_from_slice(b"JUNK");
            chunks.extend_from_slice(&3_u32.to_le_bytes());
            chunks.extend_from_slice(b"abc");
            chunks.push(0); // word alignment pad byte
        }
        chunks.extend_from_slice(b"fmt ");
        chunks.extend_from_slice(&16_u32.to_le_bytes());
        chunks.extend_from_slice(&audio_format.to_le_bytes());
        chunks.extend_from_slice(&channels.to_le_bytes());
        chunks.extend_from_slice(&sample_rate_hz.to_le_bytes());
        chunks.extend_from_slice(&byte_rate.to_le_bytes());
        chunks.extend_from_slice(&block_align.to_le_bytes());
        chunks.extend_from_slice(&bits_per_sample.to_le_bytes());
        chunks.extend_from_slice(b"data");
        chunks.extend_from_slice(&(data.len() as u32).to_le_bytes());
        chunks.extend_from_slice(data);

        let mut out = Vec::with_capacity(12 + chunks.len());
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&((4 + chunks.len()) as u32).to_le_bytes());
        out.extend_from_slice(b"WAVE");
        out.extend_from_slice(&chunks);
        out
    }

    #[test]
    fn parses_wav_lipsync_timeline_across_pcm_formats_and_channels() {
        // 16-bit mono low tone vs high tone spectral separation.
        let low_wav = synthesize_test_tone_wav(200, 220.0);
        let low_tl = parse_wav_lipsync_timeline(&low_wav, 20).unwrap();
        assert_eq!(low_tl.sample_rate_hz, 24_000);
        assert_eq!(low_tl.channels, 1);
        assert_eq!(low_tl.duration_ms, 200);
        assert_eq!(low_tl.frames.len(), 10);
        assert!(low_tl.overall_envelope.rms > 0.2);
        assert!(low_tl.overall_envelope.low_band > low_tl.overall_envelope.high_band);
        assert!(!low_tl.visemes.is_empty());

        let high_wav = synthesize_test_tone_wav(160, 4200.0);
        let high_tl = parse_wav_lipsync_timeline(&high_wav, 20).unwrap();
        assert!(high_tl.overall_envelope.high_band > high_tl.overall_envelope.low_band);
        assert!(high_tl.visemes.iter().any(|v| v.vowel == "i"));

        // 8-bit unsigned stereo PCM with preceding odd-length JUNK subchunk.
        let mut pcm8 = Vec::new();
        for i in 0..480 {
            let s = if i % 4 < 2 { 192_u8 } else { 64_u8 };
            pcm8.push(s);
            pcm8.push(s);
        }
        let wav8 = build_custom_wav(1, 2, 16_000, 8, true, &pcm8);
        let tl8 = parse_wav_lipsync_timeline(&wav8, 5).unwrap(); // frame_ms clamps to >= 10
        assert_eq!(tl8.channels, 2);
        assert_eq!(tl8.duration_ms, 30);
        assert_eq!(tl8.frames.len(), 3);
        assert!(tl8.overall_envelope.rms > 0.3);

        // 24-bit signed mono PCM.
        let mut pcm24 = Vec::new();
        for i in 0..240 {
            let v: i32 = if i % 2 == 0 { 0x40_0000 } else { -0x40_0000 };
            let bytes = v.to_le_bytes();
            pcm24.extend_from_slice(&bytes[0..3]);
        }
        let wav24 = build_custom_wav(1, 1, 24_000, 24, false, &pcm24);
        let tl24 = parse_wav_lipsync_timeline(&wav24, 10).unwrap();
        assert_eq!(tl24.duration_ms, 10);
        assert!(tl24.overall_envelope.rms > 0.3);

        // 32-bit signed mono PCM.
        let mut pcm32 = Vec::new();
        for i in 0..240 {
            let v: i32 = if i % 2 == 0 {
                i32::MAX / 2
            } else {
                -(i32::MAX / 2)
            };
            pcm32.extend_from_slice(&v.to_le_bytes());
        }
        let wav32 = build_custom_wav(1, 1, 24_000, 32, false, &pcm32);
        let tl32 = parse_wav_lipsync_timeline(&wav32, 10).unwrap();
        assert!((tl32.overall_envelope.rms - 0.5).abs() < 0.05);

        // 32-bit IEEE float stereo PCM.
        let mut pcm_f32 = Vec::new();
        for i in 0..480 {
            let f = if i % 8 < 4 { 0.4_f32 } else { -0.4_f32 };
            pcm_f32.extend_from_slice(&f.to_le_bytes());
            pcm_f32.extend_from_slice(&f.to_le_bytes());
        }
        let wav_f32 = build_custom_wav(3, 2, 24_000, 32, false, &pcm_f32);
        let tl_f32 = parse_wav_lipsync_timeline(&wav_f32, 10).unwrap();
        assert_eq!(tl_f32.duration_ms, 20);
        assert!((tl_f32.overall_envelope.rms - 0.4).abs() < 0.05);

        // Streaming sentinel `0x7FFF_F000` (like `espeak-ng --stdout`).
        let mut streaming = low_wav.clone();
        streaming[4..8].copy_from_slice(&0x7FFF_F024_u32.to_le_bytes());
        streaming[40..44].copy_from_slice(&0x7FFF_F000_u32.to_le_bytes());
        let tl_stream = parse_wav_lipsync_timeline(&streaming, 20).unwrap();
        assert_eq!(tl_stream.duration_ms, 200);

        // Malformed / unsupported checks.
        assert!(parse_wav_lipsync_timeline(b"NOT_WAV", 20).is_err());
        let unsupported = build_custom_wav(6, 1, 8000, 8, false, &[0_u8; 8]);
        assert!(matches!(
            parse_wav_lipsync_timeline(&unsupported, 20),
            Err(TtsError::UnsupportedWavFormat { format: 6, .. })
        ));
    }

    #[test]
    fn synthetic_espeak_provider_populates_wav_timeline_and_visemes() {
        let synth = EspeakNgProvider::new_synthetic().with_frame_ms(20);
        let res = block_on_ready(synth.synthesize("Hello Jiku!", AvatarEmotion::Speaking)).unwrap();
        assert_eq!(res.provider, "espeak_ng");
        assert_eq!(res.mime_type, "audio/wav");
        assert!(res.audio_bytes.starts_with(b"RIFF"));
        assert!(res.duration_ms > 150);
        assert!(!res.envelope_timeline.is_empty());
        assert_eq!(res.cue.emotion, AvatarEmotion::Speaking);
        assert!(res.cue.pcm_envelope.is_some());
        assert!(!res.cue.mora_envelope.is_empty());

        let fallback = EspeakNgProvider::new("en", 160)
            .with_binary_path("/nonexistent/espeak-ng-missing")
            .with_synthetic_fallback(true);
        let fb_res =
            block_on_ready(fallback.synthesize("Fallback speech", AvatarEmotion::Success)).unwrap();
        assert!(fb_res.duration_ms > 0);
        assert_eq!(fb_res.cue.emotion, AvatarEmotion::Success);

        let strict_missing = EspeakNgProvider::new("en", 160)
            .with_binary_path("/nonexistent/espeak-ng-missing")
            .with_synthetic_fallback(false);
        assert!(matches!(
            block_on_ready(strict_missing.synthesize("Fail", AvatarEmotion::Alert)),
            Err(TtsError::ProcessFailed { .. })
        ));

        if Path::new("/usr/bin/espeak-ng").exists() {
            let real = EspeakNgProvider::default();
            let real_res =
                block_on_ready(real.synthesize("Jiku ready.", AvatarEmotion::Neutral)).unwrap();
            assert!(real_res.audio_bytes.starts_with(b"RIFF"));
            assert!(real_res.duration_ms > 50);
            assert!(!real_res.envelope_timeline.is_empty());
            assert!(real_res.cue.pcm_envelope.is_some());
        }
    }

    #[test]
    fn irodori_input_formats_emotion_hints_without_duplication() {
        assert_eq!(
            format_irodori_input("Checking window layout", AvatarEmotion::Thinking, true),
            "🤔 Checking window layout"
        );
        assert_eq!(
            format_irodori_input("🤔 Already hinted", AvatarEmotion::Thinking, true),
            "🤔 Already hinted"
        );
        assert_eq!(
            format_irodori_input("Task done", AvatarEmotion::Success, true),
            "😊 Task done"
        );
        assert_eq!(
            format_irodori_input("Protected prompt", AvatarEmotion::Alert, true),
            "😰 Protected prompt"
        );
        assert_eq!(
            format_irodori_input("Neutral text", AvatarEmotion::Neutral, true),
            "Neutral text"
        );
        assert_eq!(
            format_irodori_input("Raw text", AvatarEmotion::Alert, false),
            "Raw text"
        );

        let p1 = IrodoriHttpProvider::new("http://127.0.0.1:8080").unwrap();
        assert_eq!(
            p1.resolve_target_url(),
            ("http://127.0.0.1:8080/v1/audio/speech".to_string(), false)
        );
        let p2 = IrodoriHttpProvider::new("http://127.0.0.1:8080/synthesize").unwrap();
        assert_eq!(
            p2.resolve_target_url(),
            ("http://127.0.0.1:8080/synthesize".to_string(), true)
        );
    }

    #[test]
    fn audio_playback_sink_supports_silent_missing_command_and_barge_in_stop() {
        let wav = synthesize_test_tone_wav(40, 440.0);

        let silent = AudioPlaybackSink::new(AudioSinkKind::Silent);
        assert!(!silent.play_wav_interruptible(&wav).unwrap());
        assert!(!silent.stop_playback());

        let missing = AudioPlaybackSink::new(AudioSinkKind::Command(
            "nonexistent-jiku-player".to_string(),
        ));
        assert!(!missing.play_wav_interruptible(&wav).unwrap());

        let cat_sink = AudioPlaybackSink::new(AudioSinkKind::Command("cat".to_string()));
        assert!(cat_sink.play_wav_interruptible(&wav).unwrap());
        assert!(!cat_sink.is_playing());

        let sleeper = AudioPlaybackSink::new(AudioSinkKind::Command("sleep 10".to_string()));
        let sleeper_bg = sleeper.clone();
        let wav_bg = wav.clone();
        let handle = thread::spawn(move || sleeper_bg.play_wav_interruptible(&wav_bg));

        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while !sleeper.is_playing() && std::time::Instant::now() < deadline {
            thread::sleep(Duration::from_millis(5));
        }
        assert!(sleeper.is_playing(), "expected sleep subprocess to start");

        let start_kill = std::time::Instant::now();
        assert!(sleeper.stop_playback());
        let played = handle.join().unwrap().unwrap();
        assert!(!played, "barge-in should abort playback and return false");
        assert!(!sleeper.is_playing());
        assert!(start_kill.elapsed() < Duration::from_secs(1));
    }

    #[test]
    fn tts_synthesis_result_serde_backwards_compat() {
        let legacy_json = serde_json::json!({
            "provider": "noop",
            "mime_type": "audio/wav",
            "audio_bytes": [],
            "cue": AvatarCue::default()
        });
        let parsed: TtsSynthesisResult = serde_json::from_value(legacy_json).unwrap();
        assert_eq!(parsed.duration_ms, 0);
        assert!(parsed.envelope_timeline.is_empty());
    }
}
