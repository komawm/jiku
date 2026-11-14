//! Concrete network transport implementations of the Three-Driver Model Architecture:
//!
//! - [`decisions`]: `POST /v1beta/decisions` (`gemini-decision-1`) + seamless fallback to
//!   [`turn::TurnModelClient`] emulation.
//! - [`turn`]: `OpenCodeGo` (`deepseek-v4.1-flash` with `"thinking": {"type": "disabled"}` and
//!   `x-opencode-session` UUID), OpenAI-compatible, Google Gemini, and Anthropic endpoints.
//! - [`live`]: Stateful full-duplex WebSocket (`BidiGenerateContent`) for `LiveSessionDriver`.

pub mod decisions;
pub mod live;
pub mod turn;

use std::{env, time::Duration};

use thiserror::Error;

/// Inline or Files API multimodal attachment passed to model drivers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DriverMediaInput {
    /// Raw bytes encoded as base64 on the wire.
    Inline {
        /// Standard MIME type (e.g. `"image/png"`, `"image/jpeg"`, `"audio/pcm"`).
        mime_type: String,
        /// Unencoded raw payload bytes.
        bytes: Vec<u8>,
    },
    /// Google Files API URI (`files/...` or `https://...`).
    FilesUri {
        /// Resource URI returned by Gemini Files API.
        uri: String,
    },
}

impl DriverMediaInput {
    /// Returns the contract-level [`crate::DecisionMediaSource`] descriptor for validation.
    #[must_use]
    pub fn to_contract_source(&self) -> crate::DecisionMediaSource {
        match self {
            Self::Inline { mime_type, .. } => crate::DecisionMediaSource::InlineBytes {
                mime_type: mime_type.clone(),
            },
            Self::FilesUri { uri } => crate::DecisionMediaSource::FilesApi {
                file_uri: uri.clone(),
            },
        }
    }
}

/// Errors raised by [`decisions`], [`turn`], and [`live`] network drivers.
#[derive(Debug, Error)]
pub enum DriverError {
    /// Configuration or API key resolution failure.
    #[error("driver configuration error: {0}")]
    Config(String),

    /// Batch failed `DecisionBatchSpec::validate` or `LiveSessionConfig::validate`.
    #[error("contract validation error: {0}")]
    Validation(String),

    /// HTTP request / proxy / TLS error.
    #[error("HTTP transport error: {0}")]
    Http(#[from] reqwest::Error),

    /// Remote API returned HTTP error status.
    #[error("remote API HTTP {status} from {endpoint}: {body}")]
    RemoteStatus {
        /// HTTP status code.
        status: u16,
        /// Endpoint URL (redacted query params if any).
        endpoint: String,
        /// Response error body snippet.
        body: String,
    },

    /// Malformed or unexpected JSON / SSE payload returned by model endpoint.
    #[error("invalid response payload: {0}")]
    InvalidPayload(String),

    /// WebSocket transport failure in `LiveSessionDriver`.
    #[error("WebSocket transport error: {0}")]
    WebSocket(String),
}

/// Constructs a `reqwest::Client` using `rustls-tls` and honoring standard proxy
/// env vars plus `$NETWORK_PROXY` when `HTTPS_PROXY`/`HTTP_PROXY`/`ALL_PROXY` are unset.
pub fn build_http_client(timeout: Duration) -> Result<reqwest::Client, DriverError> {
    let mut builder = reqwest::Client::builder().timeout(timeout);
    if has_no_standard_proxy_env()
        && let Some(proxy_url) = env::var("NETWORK_PROXY")
            .ok()
            .filter(|s| !s.trim().is_empty())
        && let Ok(proxy) = reqwest::Proxy::all(proxy_url.trim())
    {
        builder = builder.proxy(proxy);
    }
    Ok(builder.build()?)
}

fn has_no_standard_proxy_env() -> bool {
    const KEYS: &[&str] = &[
        "HTTPS_PROXY",
        "https_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "ALL_PROXY",
        "all_proxy",
    ];
    !KEYS
        .iter()
        .any(|k| env::var(k).ok().is_some_and(|v| !v.trim().is_empty()))
}
