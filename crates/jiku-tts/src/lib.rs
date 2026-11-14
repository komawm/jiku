//! Optional speech synthesis providers, lip-sync envelope extractors, and
//! D-Bus desktop notification helpers for Jiku.
//!
//! Disabled by default unless explicitly enabled in `config.toml` (`[tts] enabled = true`).

use std::{
    future::Future,
    io,
    pin::Pin,
    process::Command,
    str::FromStr,
    sync::{Arc, Mutex},
    time::Duration,
};

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Default local VOICEVOX Engine endpoint.
pub const DEFAULT_VOICEVOX_ENDPOINT: &str = "http://127.0.0.1:50021";

/// Errors returned by [`jiku-tts`](crate) providers and envelope extractors.
#[derive(Debug, Error)]
pub enum TtsError {
    /// Failed parsing VOICEVOX `AudioQuery` phoneme payload.
    #[error("invalid VOICEVOX audio_query payload: {0}")]
    InvalidAudioQuery(String),

    /// Invalid PCM byte buffer length (expected even 16-bit LE samples).
    #[error("invalid 16-bit PCM byte buffer length: {0}")]
    InvalidPcmLength(usize),

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
    /// Provider name (`"noop"`, `"voicevox"`, `"openai_speech"`).
    pub provider: String,
    /// MIME type (e.g. `"audio/wav"`, `"audio/mpeg"`).
    pub mime_type: String,
    /// Encoded audio payload (empty for `NoopTtsProvider`).
    pub audio_bytes: Vec<u8>,
    /// Extracted companion lip-sync & emotion cue.
    pub cue: AvatarCue,
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

            Ok(TtsSynthesisResult {
                provider: "voicevox".to_string(),
                mime_type: "audio/wav".to_string(),
                audio_bytes,
                cue: AvatarCue {
                    emotion,
                    mora_envelope,
                    ..AvatarCue::default()
                },
            })
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
            let audio_bytes = resp.bytes().await?.to_vec();
            Ok(TtsSynthesisResult {
                provider: "openai_speech".to_string(),
                mime_type: "audio/wav".to_string(),
                audio_bytes,
                cue: AvatarCue {
                    emotion,
                    ..AvatarCue::default()
                },
            })
        })
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
}
