#![forbid(unsafe_code)]

//! Desktop orchestration daemon (`jikud`), pure-data model driver contracts,
//! Koma IPC integration, Phodopus Lua policy hooks, and JikuBox triage.
//!
//! Pre-`v0.1.0` incubation scope: real-time PipeWire audio streaming, production
//! Landlock sandboxing, and Live2D/VRM avatar rendering remain targeted for `v0.1.0`.
//!
//! The contracts defined here capture the **Three-Driver Model Architecture**:
//! 1. [`ModelDriverKind::Decisions`] — Structured single-pass multi-question
//!    evaluation via `POST /v1beta/decisions` (`gemini-decision-1`).
//! 2. [`ModelDriverKind::TurnSse`] — Autoregressive HTTP/2 SSE turn generation
//!    for multi-step planning, Lua synthesis, and local VLM guards.
//! 3. [`ModelDriverKind::LiveWebSocket`] — Stateful full-duplex WebSocket
//!    streaming (`BidiGenerateContent`) for `jiku live` real-time voice/screen
//!    co-piloting.

pub mod a11y;
pub mod bench_core;
pub mod config;
pub mod daemon;
pub mod drivers;
pub mod grounding;
pub mod jikubox;
pub mod koma_ipc;
pub mod lua_policy;

use std::collections::HashSet;

/// The three distinct model transport drivers orchestrated by `jikud`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelDriverKind {
    /// Structured probability evaluation over up to 1,000 concurrent questions
    /// (`POST /v1beta/decisions` with `gemini-decision-1`).
    Decisions,
    /// Autoregressive turn-based HTTP/2 Server-Sent Events (`streamGenerateContent`,
    /// `/v1/chat/completions`, or `/v1/messages`).
    TurnSse,
    /// Stateful full-duplex WebSocket streaming (`BidiGenerateContent`,
    /// OpenAI Realtime, or local WebRTC/WS pipeline).
    LiveWebSocket,
}

/// Pinned model identifier required by the Decisions API.
pub const GEMINI_DECISION_1_MODEL_ID: &str = "gemini-decision-1";

/// Maximum concurrent questions permitted in a single `POST /v1beta/decisions` call.
pub const MAX_DECISION_QUESTIONS_PER_REQUEST: usize = 1000;

/// Minimum number of choices or rubric levels in a `Choice` or `Score` question.
pub const MIN_CHOICE_OR_SCORE_CARDINALITY: usize = 2;

/// Maximum number of choices or rubric levels in a `Choice` or `Score` question.
pub const MAX_CHOICE_OR_SCORE_CARDINALITY: usize = 255;

/// Mandatory fallback choice token required in every `Choice` question so forced
/// probability normalization (`sum == 1.0`) can express out-of-set states.
pub const MANDATORY_CHOICE_FALLBACK: &str = "NONE_OR_OTHER";

/// Default confidence threshold required before executing a fast-path Koma action.
pub const DEFAULT_FAST_PATH_CONFIDENCE: f32 = 0.85;

/// Workloads routed through `DecisionsDriver` (`gemini-decision-1`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DecisionWorkloadKind {
    /// System 1 GUI grounding, pre-condition checks, and action selection.
    GuiSystem1Action,
    /// Zero-token-generation control-plane intent routing and safety gating.
    ControlPlaneRouter,
    /// Real-time avatar expression, prosody, and interruption classification.
    LiveChatAndAvatarExpression,
    /// `~/JikuBox` dropzone multimodal document/image/audio triage.
    JikuBoxMultimodalTriage,
    /// Second-stage semantic reranking over SQLite FTS5 memory candidates.
    MemoryFts5Rerank,
}

/// Question types supported by `POST /v1beta/decisions`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecisionQuestionKind {
    /// Binary `Yes` / `No` probability evaluation (`noul`).
    Noul {
        /// Plain-text evaluation instruction.
        instructions: String,
    },
    /// Mutually exclusive categorical choice (`choice`, normalized to `1.0`).
    Choice {
        /// Plain-text selection instruction.
        instructions: String,
        /// Candidate choices (`2..=255`), which must include `"NONE_OR_OTHER"`
        /// or `"not_visible"`.
        choices: Vec<String>,
        /// Question ID of a companion [`DecisionQuestionKind::Noul`] scope gate
        /// within the same batch.
        scope_gate_question_id: String,
    },
    /// Ordinal rubric evaluation (`score`, `2..=255` rubric levels).
    Score {
        /// Plain-text scoring instruction.
        instructions: String,
        /// Ordered rubric descriptions (`2..=255`).
        rubric: Vec<String>,
    },
}

/// Multimodal context attachment method for a `DecisionsDriver` batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecisionMediaSource {
    /// Inline base64-encoded payload with an explicit MIME type.
    InlineBytes {
        /// MIME type such as `image/webp`, `image/png`, `image/jpeg`,
        /// `audio/*`, `video/*`, or `application/pdf`.
        mime_type: String,
    },
    /// Pre-uploaded Gemini Files API resource URI.
    FilesApi {
        /// Files API URI (must not be a `gs://` or YouTube URL).
        file_uri: String,
    },
}

/// Pure-data specification for a single `POST /v1beta/decisions` request batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecisionBatchSpec {
    /// Target model identifier (must equal [`GEMINI_DECISION_1_MODEL_ID`]).
    pub model: String,
    /// Multimodal input source.
    pub media_source: DecisionMediaSource,
    /// Ordered list of `(question_id, question_kind)` pairs (`1..=1000`).
    pub questions: Vec<(String, DecisionQuestionKind)>,
}

/// Validation failures for [`DecisionBatchSpec::validate`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecisionValidationError {
    /// Model ID is not pinned to `"gemini-decision-1"`.
    UnpinnedOrInvalidModel {
        /// The rejected model string.
        provided: String,
    },
    /// Batch contains zero questions or more than 1,000 questions.
    InvalidQuestionCount {
        /// Actual number of questions in the batch.
        count: usize,
    },
    /// A question ID is empty or duplicated within the batch.
    InvalidOrDuplicateQuestionId {
        /// Offending question ID.
        question_id: String,
    },
    /// Question `instructions` string is empty or whitespace-only.
    EmptyQuestionInstructions {
        /// Question ID with empty instructions.
        question_id: String,
    },
    /// A `Choice` or `Score` question has fewer than 2 or more than 255 entries,
    /// or contains an empty option string.
    InvalidCardinality {
        /// Question ID with invalid cardinality.
        question_id: String,
        /// Actual cardinality.
        count: usize,
    },
    /// A `Choice` question omits `"NONE_OR_OTHER"` (or `"not_visible"`).
    MissingChoiceFallback {
        /// Question ID missing the fallback choice.
        question_id: String,
    },
    /// A `Choice` question does not reference an existing `Noul` question ID in
    /// the same batch as its `scope_gate_question_id`.
    MissingOrInvalidScopeGate {
        /// Question ID of the `Choice` question.
        question_id: String,
        /// Referenced scope gate question ID.
        scope_gate_question_id: String,
    },
    /// Media source uses an unsupported URI scheme (`gs://`, YouTube) or an
    /// unsupported MIME type.
    UnsupportedMediaSource {
        /// Reason for rejection.
        reason: &'static str,
    },
}

impl DecisionBatchSpec {
    /// Validates all four mandatory engineering guardrails for `gemini-decision-1`:
    ///
    /// 1. `model` must equal literal `"gemini-decision-1"` (rejects `-latest`
    ///    aliases and empty strings).
    /// 2. `questions.len()` must be in `1..=1000` with unique non-empty IDs and
    ///    non-empty plain-text `instructions`.
    /// 3. Every `Choice` question must have `2..=255` choices, include
    ///    `"NONE_OR_OTHER"` (or `"not_visible"`), and reference an existing
    ///    `Noul` question ID in the same batch (`scope_gate_question_id`).
    /// 4. `media_source` must reject `gs://` and YouTube URLs and accept only
    ///    supported inline MIME types or Files API URIs.
    ///
    /// Time complexity: $O(Q \cdot C)$ where $Q$ is the number of questions and
    /// $C$ is the maximum choice/rubric cardinality.
    /// Space complexity: $O(Q)$ auxiliary hash set storage for question IDs.
    pub fn validate(&self) -> Result<(), DecisionValidationError> {
        // Guardrail 1: Pin model strictly to "gemini-decision-1".
        if self.model != GEMINI_DECISION_1_MODEL_ID {
            return Err(DecisionValidationError::UnpinnedOrInvalidModel {
                provided: self.model.clone(),
            });
        }

        // Guardrail 4: Validate media source before iterating questions.
        validate_media_source(&self.media_source)?;

        // Guardrail 2: Question count in 1..=1000, unique non-empty IDs, non-empty instructions.
        let count = self.questions.len();
        if !(1..=MAX_DECISION_QUESTIONS_PER_REQUEST).contains(&count) {
            return Err(DecisionValidationError::InvalidQuestionCount { count });
        }

        let mut all_ids = HashSet::with_capacity(count);
        let mut noul_ids = HashSet::new();

        for (qid, kind) in &self.questions {
            if qid.trim().is_empty() || !all_ids.insert(qid.as_str()) {
                return Err(DecisionValidationError::InvalidOrDuplicateQuestionId {
                    question_id: qid.clone(),
                });
            }
            if let DecisionQuestionKind::Noul { .. } = kind {
                noul_ids.insert(qid.as_str());
            }
        }

        // Guardrail 2 & 3: Validate instructions, cardinality, fallback option, and Noul scope gate.
        for (qid, kind) in &self.questions {
            match kind {
                DecisionQuestionKind::Noul { instructions } => {
                    if instructions.trim().is_empty() {
                        return Err(DecisionValidationError::EmptyQuestionInstructions {
                            question_id: qid.clone(),
                        });
                    }
                }
                DecisionQuestionKind::Choice {
                    instructions,
                    choices,
                    scope_gate_question_id,
                } => {
                    if instructions.trim().is_empty() {
                        return Err(DecisionValidationError::EmptyQuestionInstructions {
                            question_id: qid.clone(),
                        });
                    }
                    let c_len = choices.len();
                    if !(MIN_CHOICE_OR_SCORE_CARDINALITY..=MAX_CHOICE_OR_SCORE_CARDINALITY)
                        .contains(&c_len)
                        || choices.iter().any(|c| c.trim().is_empty())
                    {
                        return Err(DecisionValidationError::InvalidCardinality {
                            question_id: qid.clone(),
                            count: c_len,
                        });
                    }
                    let has_fallback = choices
                        .iter()
                        .any(|c| c == MANDATORY_CHOICE_FALLBACK || c == "not_visible");
                    if !has_fallback {
                        return Err(DecisionValidationError::MissingChoiceFallback {
                            question_id: qid.clone(),
                        });
                    }
                    if !noul_ids.contains(scope_gate_question_id.as_str()) {
                        return Err(DecisionValidationError::MissingOrInvalidScopeGate {
                            question_id: qid.clone(),
                            scope_gate_question_id: scope_gate_question_id.clone(),
                        });
                    }
                }
                DecisionQuestionKind::Score {
                    instructions,
                    rubric,
                } => {
                    if instructions.trim().is_empty() {
                        return Err(DecisionValidationError::EmptyQuestionInstructions {
                            question_id: qid.clone(),
                        });
                    }
                    let r_len = rubric.len();
                    if !(MIN_CHOICE_OR_SCORE_CARDINALITY..=MAX_CHOICE_OR_SCORE_CARDINALITY)
                        .contains(&r_len)
                        || rubric.iter().any(|r| r.trim().is_empty())
                    {
                        return Err(DecisionValidationError::InvalidCardinality {
                            question_id: qid.clone(),
                            count: r_len,
                        });
                    }
                }
            }
        }

        Ok(())
    }
}

fn validate_media_source(source: &DecisionMediaSource) -> Result<(), DecisionValidationError> {
    match source {
        DecisionMediaSource::InlineBytes { mime_type } => {
            let mime = mime_type.trim();
            let valid = matches!(
                mime,
                "image/webp" | "image/png" | "image/jpeg" | "application/pdf"
            ) || mime.starts_with("audio/")
                || mime.starts_with("video/");
            if !valid {
                return Err(DecisionValidationError::UnsupportedMediaSource {
                    reason: "inline bytes must use image/{webp,png,jpeg}, audio/*, video/*, or application/pdf",
                });
            }
            Ok(())
        }
        DecisionMediaSource::FilesApi { file_uri } => {
            let uri = file_uri.trim();
            if uri.is_empty() {
                return Err(DecisionValidationError::UnsupportedMediaSource {
                    reason: "Files API URI must not be empty",
                });
            }
            let lower = uri.to_ascii_lowercase();
            if lower.starts_with("gs://") {
                return Err(DecisionValidationError::UnsupportedMediaSource {
                    reason: "gs:// Cloud Storage URIs are rejected by gemini-decision-1",
                });
            }
            if lower.contains("youtube.com/") || lower.contains("youtu.be/") {
                return Err(DecisionValidationError::UnsupportedMediaSource {
                    reason: "YouTube URLs are rejected by gemini-decision-1",
                });
            }
            if !lower.starts_with("https://") && !lower.starts_with("files/") {
                return Err(DecisionValidationError::UnsupportedMediaSource {
                    reason: "Files API URI must use https:// or files/ resource prefix",
                });
            }
            Ok(())
        }
    }
}

/// Decomposes a multi-criterion rubric into independent binary `Yes/No`
/// ([`DecisionQuestionKind::Noul`]) checklist questions for higher calibration
/// accuracy on `gemini-decision-1`.
#[must_use]
pub fn decompose_rubric_to_noul_checklist(
    prefix: &str,
    criteria: &[&str],
) -> Vec<(String, DecisionQuestionKind)> {
    criteria
        .iter()
        .enumerate()
        .filter(|(_, criterion)| !criterion.trim().is_empty())
        .map(|(idx, criterion)| {
            (
                format!("{prefix}_check_{}", idx + 1),
                DecisionQuestionKind::Noul {
                    instructions: (*criterion).trim().to_string(),
                },
            )
        })
        .collect()
}

/// Control-plane routing outcome derived from a `DecisionsDriver` evaluation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ControlPlaneRoute {
    /// Execute a deterministic fast-path Koma action directly.
    FastPathKoma {
        /// Combined calibrated confidence (`0.0..=1.0`).
        confidence: f32,
    },
    /// Escalate to the autoregressive [`TurnDriverRole::Planner`] (`TurnModelDriver`).
    FallbackTurnPlanner {
        /// Observed confidence that fell below the fast-path threshold.
        confidence: f32,
    },
    /// Halt autonomous execution and ask the human user for verification.
    RequireHumanVerification {
        /// Static explanation for why human confirmation is required.
        reason: &'static str,
    },
}

/// Routes a `DecisionsDriver` result across fast-path Koma execution,
/// turn-planner escalation, or human verification.
#[must_use]
pub fn route_decision(
    in_scope_yes_prob: f32,
    top_choice: &str,
    top_choice_prob: f32,
    safety_blocked: bool,
    threshold: f32,
) -> ControlPlaneRoute {
    if safety_blocked {
        return ControlPlaneRoute::RequireHumanVerification {
            reason: "safety_gate_triggered",
        };
    }
    let combined_confidence = in_scope_yes_prob * top_choice_prob;
    if in_scope_yes_prob < threshold
        || top_choice == MANDATORY_CHOICE_FALLBACK
        || top_choice == "not_visible"
        || combined_confidence < threshold
    {
        ControlPlaneRoute::FallbackTurnPlanner {
            confidence: combined_confidence,
        }
    } else {
        ControlPlaneRoute::FastPathKoma {
            confidence: combined_confidence,
        }
    }
}

/// Roles served by the autoregressive `TurnModelDriver` (`streamGenerateContent`,
/// `/v1/chat/completions`, or `/v1/messages`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TurnDriverRole {
    /// Multi-step task decomposition and failure recovery (`gemini-2.5-pro`, Claude, etc.).
    Planner,
    /// Open-ended natural language generation or staged Lua policy synthesis.
    OpenEndedTextOrLuaSynthesis,
    /// Optional local VLM fast guard (`qwen2.5-vl-3b-instruct`).
    FastGuard,
}

/// Default uplink PCM audio sample rate (`16 kHz`) for `LiveSessionDriver`.
pub const UPLINK_AUDIO_SAMPLE_RATE_HZ: u32 = 16_000;

/// Default native downlink PCM audio sample rate (`24 kHz`) from `LiveSessionDriver`.
pub const DOWNLINK_AUDIO_SAMPLE_RATE_HZ: u32 = 24_000;

/// Minimum supported PCM audio sample rate (`8 kHz`) for `LiveSessionDriver`.
pub const MIN_LIVE_AUDIO_SAMPLE_RATE_HZ: u32 = 8_000;

/// Maximum supported PCM audio sample rate (`48 kHz`) for `LiveSessionDriver`.
pub const MAX_LIVE_AUDIO_SAMPLE_RATE_HZ: u32 = 48_000;

/// Default Koma screen capture frame rate (`2 FPS`) streamed to a Live session.
pub const DEFAULT_SCREEN_FPS: u8 = 2;

/// Maximum configurable Koma screen capture frame rate (`60 FPS`) streamed to a Live session.
pub const MAX_SCREEN_FPS: u8 = 60;

/// Primary response modality for a `LiveSessionDriver` WebSocket session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum LiveResponseModality {
    /// Native audio output stream (`AUDIO`, 24 kHz PCM) with parallel transcription.
    #[default]
    Audio,
    /// Text-only output stream (`TEXT`).
    Text,
}

/// Execution mode for tools declared on a `LiveSessionDriver` session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LiveToolCallBehavior {
    /// Pause model generation until the tool response is returned.
    Blocking,
    /// Continue conversation while Koma / `jikud` executes the tool asynchronously.
    NonBlocking,
}

/// Scheduling policy for delivering asynchronous `NON_BLOCKING` tool results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ToolResponseScheduling {
    /// Immediately interrupt ongoing speech to report a critical failure or safety alert.
    Interrupt,
    /// Wait until the current speech turn finishes before reporting completion.
    WhenIdle,
    /// Absorb the tool result into context silently without speaking.
    Silent,
}

/// Sliding-window context compression parameters for long-running Live sessions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextWindowCompressionSpec {
    /// Token count threshold that triggers sliding-window compression.
    pub trigger_tokens: u32,
    /// Target token count retained after sliding-window compression.
    pub sliding_window_target_tokens: u32,
}

impl Default for ContextWindowCompressionSpec {
    fn default() -> Self {
        Self {
            trigger_tokens: 24_000,
            sliding_window_target_tokens: 12_000,
        }
    }
}

/// Pure-data session configuration for `LiveSessionDriver` (`jiku live`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveSessionConfig {
    /// Live-capable model identifier (default `"gemini-2.5-flash-live"`).
    pub model: String,
    /// Single primary response modality (`Audio` or `Text`; never both).
    pub response_modality: LiveResponseModality,
    /// Request plain-text transcription alongside native audio output.
    pub output_audio_transcription: bool,
    /// Allow the model to remain silent when ambient speech is not addressed to Jiku.
    pub proactive_audio: bool,
    /// Enable affective dialog prosody adaptation on supported Live models.
    pub enable_affective_dialog: bool,
    /// Uplink PCM audio sample rate in Hz (`8_000..=48_000`; default `16_000`).
    pub uplink_sample_rate_hz: u32,
    /// Downlink PCM audio sample rate in Hz (`8_000..=48_000`; default `24_000`).
    pub downlink_sample_rate_hz: u32,
    /// Maximum screen frame rate (`1..=60` FPS; default `2` FPS).
    pub max_screen_fps: u8,
    /// Context window compression thresholds.
    pub compression: ContextWindowCompressionSpec,
    /// Reconnect transparently across ~10-minute WebSocket connection resets.
    pub session_resumption_enabled: bool,
    /// Require every asynchronous Koma action to carry an epoch fence.
    pub require_koma_epoch_fence: bool,
}

impl Default for LiveSessionConfig {
    fn default() -> Self {
        Self {
            model: "gemini-2.5-flash-live".to_string(),
            response_modality: LiveResponseModality::Audio,
            output_audio_transcription: true,
            proactive_audio: false,
            enable_affective_dialog: true,
            uplink_sample_rate_hz: UPLINK_AUDIO_SAMPLE_RATE_HZ,
            downlink_sample_rate_hz: DOWNLINK_AUDIO_SAMPLE_RATE_HZ,
            max_screen_fps: DEFAULT_SCREEN_FPS,
            compression: ContextWindowCompressionSpec::default(),
            session_resumption_enabled: true,
            require_koma_epoch_fence: true,
        }
    }
}

impl LiveSessionConfig {
    /// Validates `LiveSessionDriver` session invariants:
    /// - `model` must not be empty.
    /// - `uplink_sample_rate_hz` and `downlink_sample_rate_hz` must be in `8_000..=48_000` Hz.
    /// - `max_screen_fps` must be in `1..=MAX_SCREEN_FPS` (`1..=60` FPS; default `2` FPS).
    /// - `compression.trigger_tokens` must be positive and strictly greater than
    ///   `compression.sliding_window_target_tokens` (which must also be positive).
    /// - `session_resumption_enabled` and `require_koma_epoch_fence` must be enabled.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.model.trim().is_empty() {
            return Err("LiveSessionConfig model must not be empty");
        }
        if !(MIN_LIVE_AUDIO_SAMPLE_RATE_HZ..=MAX_LIVE_AUDIO_SAMPLE_RATE_HZ)
            .contains(&self.uplink_sample_rate_hz)
        {
            return Err(
                "LiveSessionConfig uplink_sample_rate_hz must be between 8000 and 48000 Hz",
            );
        }
        if !(MIN_LIVE_AUDIO_SAMPLE_RATE_HZ..=MAX_LIVE_AUDIO_SAMPLE_RATE_HZ)
            .contains(&self.downlink_sample_rate_hz)
        {
            return Err(
                "LiveSessionConfig downlink_sample_rate_hz must be between 8000 and 48000 Hz",
            );
        }
        if self.max_screen_fps == 0 || self.max_screen_fps > MAX_SCREEN_FPS {
            return Err("LiveSessionConfig max_screen_fps must be between 1 and 60 FPS");
        }
        if self.compression.sliding_window_target_tokens == 0
            || self.compression.trigger_tokens <= self.compression.sliding_window_target_tokens
        {
            return Err(
                "ContextWindowCompressionSpec requires 0 < sliding_window_target_tokens < trigger_tokens",
            );
        }
        if !self.session_resumption_enabled {
            return Err("LiveSessionConfig requires session_resumption_enabled = true");
        }
        if !self.require_koma_epoch_fence {
            return Err("LiveSessionConfig requires require_koma_epoch_fence = true");
        }
        Ok(())
    }
}

/// Configurable 2D spatial grid specification (`cols x rows`, default `15 x 15`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GridSpec {
    /// Number of horizontal columns (`1..=26`, labelled `A..=Z`; default `15`).
    pub cols: u8,
    /// Number of vertical rows (`1..=255`, labelled `1..=rows`; default `15`).
    pub rows: u8,
}

impl Default for GridSpec {
    fn default() -> Self {
        Self { cols: 15, rows: 15 }
    }
}

impl GridSpec {
    /// Returns the total number of cells (`cols * rows`) in the grid.
    #[must_use]
    pub const fn total_cells(&self) -> usize {
        (self.cols as usize) * (self.rows as usize)
    }

    /// Returns `true` if `cols` fits within ASCII letters `A..=Z` (`1..=26`),
    /// `rows >= 1`, and `total_cells()` satisfies the `/v1beta/decisions`
    /// `Choice` cardinality bounds (`2..=255`).
    #[must_use]
    pub fn is_valid_for_decisions_api(&self) -> bool {
        self.cols > 0
            && self.cols <= 26
            && self.rows > 0
            && (MIN_CHOICE_OR_SCORE_CARDINALITY..=MAX_CHOICE_OR_SCORE_CARDINALITY)
                .contains(&self.total_cells())
    }

    /// Parses a cell label (e.g., `"A1"`, `"L13"`) into 0-based `(col, row)` coordinates
    /// bounded by `self.cols` and `self.rows`.
    #[must_use]
    pub fn parse_cell_label(&self, label: &str) -> Option<(i32, i32)> {
        if self.cols == 0 || self.cols > 26 || self.rows == 0 {
            return None;
        }
        let trimmed = label.trim();
        let mut chars = trimmed.chars();
        let col_char = chars.next()?;
        let max_col_char = (b'A' + self.cols - 1) as char;
        if !('A'..=max_col_char).contains(&col_char) {
            return None;
        }
        let row_str: &str = chars.as_str();
        let row_1_based: i32 = row_str.parse().ok()?;
        if !(1..=i32::from(self.rows)).contains(&row_1_based) {
            return None;
        }
        let col = i32::from(col_char as u8 - b'A');
        let row = row_1_based - 1;
        Some((col, row))
    }

    /// Interpolates a continuous pixel centroid `(x, y)` on `[0.0, width] x [0.0, height]`
    /// from a categorical probability distribution over this `cols x rows` grid by
    /// clustering the mode cell with its 8-connected neighbors whose probabilities are
    /// at least `min_cell_prob`.
    ///
    /// Time complexity: $O(N)$ where $N$ is `probabilities.len()`.
    /// Space complexity: $O(N)$ for parsed valid grid cells.
    #[must_use]
    pub fn interpolate_centroid(
        &self,
        probabilities: &[(String, f32)],
        width: f32,
        height: f32,
        min_cell_prob: f32,
    ) -> Option<Grid225Centroid> {
        if self.cols == 0
            || self.cols > 26
            || self.rows == 0
            || !width.is_finite()
            || !height.is_finite()
            || width <= 0.0
            || height <= 0.0
        {
            return None;
        }

        let parsed: Vec<(&str, i32, i32, f32)> = probabilities
            .iter()
            .filter_map(|(label, prob)| {
                if !prob.is_finite() || *prob <= 0.0 {
                    return None;
                }
                let (col, row) = self.parse_cell_label(label)?;
                Some((label.as_str(), col, row, *prob))
            })
            .collect();

        let &(top_label, tc, tr, top_prob) = parsed.iter().max_by(|a, b| a.3.total_cmp(&b.3))?;

        if top_prob < min_cell_prob {
            return None;
        }

        let cell_w = width / f32::from(self.cols);
        let cell_h = height / f32::from(self.rows);

        let mut mass = 0.0_f32;
        let mut weighted_x = 0.0_f32;
        let mut weighted_y = 0.0_f32;

        for &(_, col, row, prob) in &parsed {
            if (col - tc).abs() <= 1 && (row - tr).abs() <= 1 && prob >= min_cell_prob {
                let cx = (col as f32 + 0.5) * cell_w;
                let cy = (row as f32 + 0.5) * cell_h;
                mass += prob;
                weighted_x += prob * cx;
                weighted_y += prob * cy;
            }
        }

        if mass <= 0.0 {
            return None;
        }

        Some(Grid225Centroid {
            top_cell: top_label.to_string(),
            top_single_prob: top_prob,
            combined_cluster_mass: mass,
            x: weighted_x / mass,
            y: weighted_y / mass,
        })
    }
}

/// Sub-grid probability-weighted centroid interpolated from Koma's spatial grid
/// (default 15x15 `grid_225`, `A1..O15`) categorical distribution.
#[derive(Debug, Clone, PartialEq)]
pub struct Grid225Centroid {
    /// Label of the highest-probability grid cell (e.g., `"L13"`).
    pub top_cell: String,
    /// Probability of `top_cell` alone.
    pub top_single_prob: f32,
    /// Sum of probabilities across `top_cell` and its qualifying 8-connected neighbors.
    pub combined_cluster_mass: f32,
    /// Probability-weighted horizontal pixel coordinate on `[0.0, width]`.
    pub x: f32,
    /// Probability-weighted vertical pixel coordinate on `[0.0, height]`.
    pub y: f32,
}

/// Interpolates a continuous pixel centroid `(x, y)` from a 15x15 `grid_225`
/// (`A1..O15`) categorical probability distribution by clustering the mode cell
/// with its 8-connected neighbors whose probabilities are at least `min_cell_prob`.
///
/// Time complexity: $O(N)$ where $N$ is `probabilities.len()`.
/// Space complexity: $O(N)$ for parsed valid grid cells.
#[must_use]
pub fn interpolate_grid_225_centroid(
    probabilities: &[(String, f32)],
    width: f32,
    height: f32,
    min_cell_prob: f32,
) -> Option<Grid225Centroid> {
    GridSpec::default().interpolate_centroid(probabilities, width, height, min_cell_prob)
}

/// Ordered time-slot `Choice` outcome for Strategy A (`TimeSlotMacro`).
#[derive(Debug, Clone, PartialEq)]
pub struct TimeSlotSelection {
    /// Winning choice token for this time slot (e.g., `"ctrl_c"` or `"stop_done"`).
    pub choice: String,
    /// Calibrated probability (`0.0..=1.0`) of the winning choice.
    pub confidence: f32,
}

/// Strategy A (`TimeSlotMacro`): Compiles an ordered sequence of time-slot
/// `Choice` answers (`slot_1..=slot_K`) into an executable macro sequence,
/// truncating immediately at the first stop sentinel, fallback token, or
/// low-confidence slot.
///
/// Time complexity: $O(K)$ where $K$ is `slots.len()`.
/// Space complexity: $O(K)$ for the compiled macro vector.
#[must_use]
pub fn compile_time_slot_macro(
    slots: &[TimeSlotSelection],
    stop_sentinel: &str,
    min_confidence: f32,
) -> Vec<String> {
    compile_time_slot_macro_with_fallback(
        slots,
        stop_sentinel,
        MANDATORY_CHOICE_FALLBACK,
        min_confidence,
    )
}

fn compile_time_slot_macro_with_fallback(
    slots: &[TimeSlotSelection],
    stop_sentinel: &str,
    fallback_token: &str,
    min_confidence: f32,
) -> Vec<String> {
    let mut compiled = Vec::with_capacity(slots.len());
    for slot in slots {
        let token = slot.choice.trim();
        if token.is_empty()
            || token == stop_sentinel
            || token == fallback_token
            || token == MANDATORY_CHOICE_FALLBACK
            || token == "other"
            || !slot.confidence.is_finite()
            || slot.confidence < min_confidence
        {
            break;
        }
        compiled.push(token.to_string());
    }
    compiled
}

/// Candidate action evaluated with an independent `Noul` activation gate and
/// per-step index probabilities for Strategy B (`BernoulliExpectedStep`).
#[derive(Debug, Clone, PartialEq)]
pub struct BernoulliStepCandidate {
    /// Action or shortcut identifier (e.g., `"ctrl_c"`).
    pub key: String,
    /// Independent `Noul` `Yes` probability that this candidate belongs in the plan.
    pub use_noul_prob: f32,
    /// Pairs of `(step_index, probability)` for ordering active candidates.
    pub step_probabilities: Vec<(u8, f32)>,
}

/// Strategy B (`BernoulliExpectedStep`): Filters candidates whose independent
/// `use_noul_prob >= activation_threshold`, computes each active candidate's
/// conditional expected step index $E[\text{step}] = \frac{\sum k \cdot P(\text{step}_k)}{\sum P(\text{step}_k)}$,
/// and orders them by ascending $E[\text{step}]$.
///
/// Time complexity: $O(N \cdot S + M \log M)$ where $N$ is `candidates.len()`,
/// $S$ is the average `step_probabilities` length, and $M \le N$ is the number
/// of activated candidates.
/// Space complexity: $O(M)$ for the filtered and sorted result vector.
#[must_use]
pub fn compile_bernoulli_step_macro(
    candidates: &[BernoulliStepCandidate],
    activation_threshold: f32,
) -> Vec<(String, f32)> {
    let mut active: Vec<(String, f32)> = candidates
        .iter()
        .filter(|c| {
            !c.key.trim().is_empty()
                && c.use_noul_prob.is_finite()
                && c.use_noul_prob >= activation_threshold
        })
        .filter_map(|c| {
            let mut weighted_sum = 0.0_f32;
            let mut prob_sum = 0.0_f32;
            for &(step, prob) in &c.step_probabilities {
                if prob.is_finite() && prob > 0.0 {
                    weighted_sum += f32::from(step) * prob;
                    prob_sum += prob;
                }
            }
            if prob_sum <= 0.0 {
                None
            } else {
                Some((c.key.clone(), weighted_sum / prob_sum))
            }
        })
        .collect();

    active.sort_by(|a, b| a.1.total_cmp(&b.1));
    active
}

/// Strategy C (`ThresholdProbe`): Filters an order-independent categorical or
/// `Noul` probability distribution down to viable probe candidates (`prob >= min_prob`),
/// excluding fallback sentinels, and orders them by descending probability.
///
/// Time complexity: $O(N \log N)$ where $N$ is `probabilities.len()`.
/// Space complexity: $O(N)$ for the filtered and sorted probe list.
#[must_use]
pub fn compile_threshold_probe(
    probabilities: &[(String, f32)],
    min_prob: f32,
) -> Vec<(String, f32)> {
    compile_threshold_probe_with_fallback(probabilities, MANDATORY_CHOICE_FALLBACK, min_prob)
}

fn compile_threshold_probe_with_fallback(
    probabilities: &[(String, f32)],
    fallback_token: &str,
    min_prob: f32,
) -> Vec<(String, f32)> {
    let mut probes: Vec<(String, f32)> = probabilities
        .iter()
        .filter_map(|(candidate, prob)| {
            let token = candidate.trim();
            if token.is_empty()
                || token == fallback_token
                || token == MANDATORY_CHOICE_FALLBACK
                || token == "other"
                || token == "not_visible"
                || !prob.is_finite()
                || *prob < min_prob
            {
                None
            } else {
                Some((token.to_string(), *prob))
            }
        })
        .collect();

    probes.sort_by(|a, b| b.1.total_cmp(&a.1));
    probes
}

/// Configurable thresholds, grid geometry, and sentinel tokens for `DecisionCompiler`
/// post-processing strategies (spatial centroid interpolation, Strategy A time-slot macros,
/// Strategy B Bernoulli expected-step ordering, and Strategy C threshold probes).
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionCompilerConfig {
    /// Spatial grid geometry (`cols x rows`, default `15 x 15`).
    pub grid: GridSpec,
    /// Confidence threshold required before executing a fast-path Koma action (default `0.85`).
    pub fast_path_confidence: f32,
    /// Minimum cell probability included in 8-connected centroid interpolation (default `0.05`).
    pub min_cell_prob: f32,
    /// Minimum slot confidence for Strategy A (`TimeSlotMacro`, default `0.75`).
    pub time_slot_min_confidence: f32,
    /// Minimum `Noul` activation probability for Strategy B (`BernoulliExpectedStep`, default `0.80`).
    pub bernoulli_activation_threshold: f32,
    /// Minimum candidate probability for Strategy C (`ThresholdProbe`, default `0.15`).
    pub probe_min_prob: f32,
    /// Sentinel choice token that terminates a Strategy A time-slot macro (default `"stop_done"`).
    pub stop_sentinel: String,
    /// Mandatory out-of-set fallback choice token (default `"NONE_OR_OTHER"`).
    pub fallback_token: String,
}

impl Default for DecisionCompilerConfig {
    fn default() -> Self {
        Self {
            grid: GridSpec::default(),
            fast_path_confidence: DEFAULT_FAST_PATH_CONFIDENCE,
            min_cell_prob: 0.05,
            time_slot_min_confidence: 0.75,
            bernoulli_activation_threshold: 0.80,
            probe_min_prob: 0.15,
            stop_sentinel: "stop_done".to_string(),
            fallback_token: MANDATORY_CHOICE_FALLBACK.to_string(),
        }
    }
}

impl DecisionCompilerConfig {
    /// Interpolates a continuous pixel centroid `(x, y)` on `[0.0, width] x [0.0, height]`
    /// using `self.grid` and `self.min_cell_prob`.
    #[must_use]
    pub fn interpolate_grid(
        &self,
        probabilities: &[(String, f32)],
        width: f32,
        height: f32,
    ) -> Option<Grid225Centroid> {
        self.grid
            .interpolate_centroid(probabilities, width, height, self.min_cell_prob)
    }

    /// Compiles an ordered sequence of Strategy A time-slot `Choice` answers using
    /// `self.stop_sentinel`, `self.fallback_token`, and `self.time_slot_min_confidence`.
    #[must_use]
    pub fn compile_time_slot(&self, slots: &[TimeSlotSelection]) -> Vec<String> {
        compile_time_slot_macro_with_fallback(
            slots,
            &self.stop_sentinel,
            &self.fallback_token,
            self.time_slot_min_confidence,
        )
    }

    /// Compiles Strategy B Bernoulli step candidates using `self.bernoulli_activation_threshold`.
    #[must_use]
    pub fn compile_bernoulli_steps(
        &self,
        candidates: &[BernoulliStepCandidate],
    ) -> Vec<(String, f32)> {
        compile_bernoulli_step_macro(candidates, self.bernoulli_activation_threshold)
    }

    /// Compiles Strategy C threshold probes using `self.fallback_token` and `self.probe_min_prob`.
    #[must_use]
    pub fn compile_probe(&self, probabilities: &[(String, f32)]) -> Vec<(String, f32)> {
        compile_threshold_probe_with_fallback(
            probabilities,
            &self.fallback_token,
            self.probe_min_prob,
        )
    }

    /// Routes a `DecisionsDriver` evaluation using `self.fast_path_confidence` and `self.fallback_token`.
    #[must_use]
    pub fn route(
        &self,
        in_scope_yes_prob: f32,
        top_choice: &str,
        top_choice_prob: f32,
        safety_blocked: bool,
    ) -> ControlPlaneRoute {
        if top_choice == self.fallback_token {
            if safety_blocked {
                return ControlPlaneRoute::RequireHumanVerification {
                    reason: "safety_gate_triggered",
                };
            }
            return ControlPlaneRoute::FallbackTurnPlanner {
                confidence: in_scope_yes_prob * top_choice_prob,
            };
        }
        route_decision(
            in_scope_yes_prob,
            top_choice,
            top_choice_prob,
            safety_blocked,
            self.fast_path_confidence,
        )
    }
}

/// Execution mode for structured decision batches in `jikud`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecisionExecutionMode {
    /// Uses the native zero-decode `/v1beta/decisions` endpoint (`model == "gemini-decision-1"`, `0` output tokens).
    NativeDecisionsApi {
        /// Provider key (e.g., `"gemini"`).
        provider: String,
        /// Pinned decision model ID (`"gemini-decision-1"`).
        model: String,
    },
    /// Uses any configured text/multimodal model (e.g., OpenAI GPT, Anthropic Claude,
    /// Gemini Pro/Flash, OpenRouter, or local `vLLM`/`llama.cpp`/`Ollama`) to emulate
    /// the exact [`DecisionBatchSpec`] output contract via prompt-constrained JSON Schema
    /// output when no dedicated decision model is configured (or when the user assigns
    /// a general text model to the decision role).
    EmulatedStructuredTextModel {
        /// Provider key serving the text/multimodal model.
        provider: String,
        /// Target text/multimodal model identifier.
        model: String,
        /// Whether strict JSON Schema output enforcement is enabled.
        enforce_json_schema: bool,
    },
}

impl Default for DecisionExecutionMode {
    fn default() -> Self {
        Self::NativeDecisionsApi {
            provider: "gemini".to_string(),
            model: GEMINI_DECISION_1_MODEL_ID.to_string(),
        }
    }
}

/// Configurable routing and post-processing policy for `DecisionsDriver` (`[roles.decision_router]`).
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionRouterPolicy {
    /// Default or resolved execution mode (`NativeDecisionsApi` vs `EmulatedStructuredTextModel`).
    pub mode: DecisionExecutionMode,
    /// Optional dedicated decision provider key (default `Some("gemini")`).
    pub decision_provider: Option<String>,
    /// Optional dedicated decision model ID (default `Some("gemini-decision-1")`;
    /// setting `None` falls back to `primary_text_model_id`).
    pub decision_model_id: Option<String>,
    /// Primary text/planner provider key used when `decision_model_id` is omitted (default `"gemini"`).
    pub primary_provider: String,
    /// Primary text/planner model ID used when `decision_model_id` is omitted (default `"gemini-2.5-pro"`).
    pub primary_text_model_id: String,
    /// Confidence threshold required before executing a fast-path Koma action (default `0.85`).
    pub fast_path_confidence_threshold: f32,
    /// Maximum concurrent questions per decision batch (default `8`).
    pub max_questions_per_batch: usize,
    /// Configurable compiler parameters (grid dimensions, thresholds, and sentinels).
    pub compiler: DecisionCompilerConfig,
}

impl Default for DecisionRouterPolicy {
    fn default() -> Self {
        Self {
            mode: DecisionExecutionMode::default(),
            decision_provider: Some("gemini".to_string()),
            decision_model_id: Some(GEMINI_DECISION_1_MODEL_ID.to_string()),
            primary_provider: "gemini".to_string(),
            primary_text_model_id: "gemini-2.5-pro".to_string(),
            fast_path_confidence_threshold: DEFAULT_FAST_PATH_CONFIDENCE,
            max_questions_per_batch: 8,
            compiler: DecisionCompilerConfig::default(),
        }
    }
}

impl DecisionRouterPolicy {
    /// Resolves the effective [`DecisionExecutionMode`] from this policy's configured
    /// `decision_model_id` and `primary_text_model_id`.
    pub fn resolve_mode(&self) -> Result<DecisionExecutionMode, &'static str> {
        resolve_decision_execution_mode(
            &self.primary_provider,
            &self.primary_text_model_id,
            self.decision_provider.as_deref(),
            self.decision_model_id.as_deref(),
        )
    }
}

/// Resolves whether a decision batch should execute against the native `/v1beta/decisions`
/// endpoint or fall back to structured-output emulation on the configured/primary text model.
///
/// If `configured_decision_model` is `None` or empty, this function **never errors** and
/// automatically falls back to [`DecisionExecutionMode::EmulatedStructuredTextModel`] using
/// `primary_provider` and `primary_model`.
///
/// Time complexity: $O(1)$ string slice inspection and allocation.
/// Space complexity: $O(P + M)$ for the cloned provider and model strings.
pub fn resolve_decision_execution_mode(
    primary_provider: &str,
    primary_model: &str,
    configured_decision_provider: Option<&str>,
    configured_decision_model: Option<&str>,
) -> Result<DecisionExecutionMode, &'static str> {
    let primary_prov = primary_provider.trim();
    let primary_mod = primary_model.trim();
    if primary_prov.is_empty() || primary_mod.is_empty() {
        return Err("primary_provider and primary_model must not be empty");
    }

    let resolved_provider = configured_decision_provider
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .unwrap_or(primary_prov);

    match configured_decision_model.map(str::trim) {
        Some(m) if !m.is_empty() => {
            if m == "gemini-decision-1-latest" || m == "gemini-decision-latest" {
                return Err("unpinned -latest alias is not supported on /v1beta/decisions");
            }
            if m == GEMINI_DECISION_1_MODEL_ID {
                Ok(DecisionExecutionMode::NativeDecisionsApi {
                    provider: resolved_provider.to_string(),
                    model: GEMINI_DECISION_1_MODEL_ID.to_string(),
                })
            } else {
                Ok(DecisionExecutionMode::EmulatedStructuredTextModel {
                    provider: resolved_provider.to_string(),
                    model: m.to_string(),
                    enforce_json_schema: true,
                })
            }
        }
        _ => Ok(DecisionExecutionMode::EmulatedStructuredTextModel {
            provider: primary_prov.to_string(),
            model: primary_mod.to_string(),
            enforce_json_schema: true,
        }),
    }
}

/// Deterministically generates a structured-output constraint prompt instructing a
/// general text/VLM model (GPT / Claude / local LLM) to output ONLY a valid JSON
/// object `{"answers": { ... }}` matching the exact `noul`, `choice`, and `score`
/// schema for every question in `questions`.
///
/// Time complexity: $O(Q \cdot C)$ where $Q$ is `questions.len()` and $C$ is the
/// average number of choices/rubric items per question.
/// Space complexity: $O(Q \cdot C)$ for the synthesized prompt string.
#[must_use]
pub fn synthesize_emulated_decision_prompt(questions: &[(String, DecisionQuestionKind)]) -> String {
    let mut out = String::from(
        "Output ONLY a valid JSON object matching {\"answers\": { ... }} with no markdown fences or extra prose.\n\
         Schema rules per question type:\n\
         - noul: {\"type\":\"noul\",\"noul\":0.01..0.99,\"confidence\":0.50..0.99}\n\
         - choice: {\"type\":\"choice\",\"choice\":\"<winner>\",\"probabilities\":{...},\"confidence\":0.0..1.0} (probabilities must sum to 1.0)\n\
         - score: {\"type\":\"score\",\"score\":1..N,\"probabilities\":{\"1\":...,\"N\":...},\"confidence\":0.0..1.0} (probabilities must sum to 1.0)\n\
         Questions:\n",
    );

    for (qid, kind) in questions {
        match kind {
            DecisionQuestionKind::Noul { instructions } => {
                out.push_str(&format!(
                    "- id=\"{qid}\" type=\"noul\" instructions=\"{instructions}\"\n"
                ));
            }
            DecisionQuestionKind::Choice {
                instructions,
                choices,
                scope_gate_question_id,
            } => {
                let joined_choices = choices.join(", ");
                out.push_str(&format!(
                    "- id=\"{qid}\" type=\"choice\" scope_gate=\"{scope_gate_question_id}\" choices=[{joined_choices}] instructions=\"{instructions}\"\n"
                ));
            }
            DecisionQuestionKind::Score {
                instructions,
                rubric,
            } => {
                let joined_rubric = rubric.join(" | ");
                out.push_str(&format!(
                    "- id=\"{qid}\" type=\"score\" levels=1..={} rubric=[{joined_rubric}] instructions=\"{instructions}\"\n",
                    rubric.len()
                ));
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_batch() -> DecisionBatchSpec {
        DecisionBatchSpec {
            model: GEMINI_DECISION_1_MODEL_ID.to_string(),
            media_source: DecisionMediaSource::InlineBytes {
                mime_type: "image/webp".to_string(),
            },
            questions: vec![
                (
                    "q_in_scope".to_string(),
                    DecisionQuestionKind::Noul {
                        instructions: "Is the target dialog visible on screen?".to_string(),
                    },
                ),
                (
                    "q_target_cell".to_string(),
                    DecisionQuestionKind::Choice {
                        instructions: "Which grid cell contains the confirm button?".to_string(),
                        choices: vec![
                            "G7".to_string(),
                            "H8".to_string(),
                            MANDATORY_CHOICE_FALLBACK.to_string(),
                        ],
                        scope_gate_question_id: "q_in_scope".to_string(),
                    },
                ),
            ],
        }
    }

    #[test]
    fn valid_decision_batch_passes_all_guardrails() {
        let batch = valid_batch();
        assert_eq!(batch.validate(), Ok(()));
    }

    #[test]
    fn guardrail_1_rejects_unpinned_or_latest_model() {
        let mut batch = valid_batch();
        batch.model = "gemini-decision-latest".to_string();
        assert_eq!(
            batch.validate(),
            Err(DecisionValidationError::UnpinnedOrInvalidModel {
                provided: "gemini-decision-latest".to_string(),
            })
        );
    }

    #[test]
    fn guardrail_2_rejects_empty_duplicate_or_blank_instructions() {
        let mut empty_batch = valid_batch();
        empty_batch.questions.clear();
        assert_eq!(
            empty_batch.validate(),
            Err(DecisionValidationError::InvalidQuestionCount { count: 0 })
        );

        let mut dup_batch = valid_batch();
        dup_batch.questions.push((
            "q_in_scope".to_string(),
            DecisionQuestionKind::Noul {
                instructions: "Duplicate ID".to_string(),
            },
        ));
        assert_eq!(
            dup_batch.validate(),
            Err(DecisionValidationError::InvalidOrDuplicateQuestionId {
                question_id: "q_in_scope".to_string(),
            })
        );

        let mut blank_inst = valid_batch();
        blank_inst.questions[0].1 = DecisionQuestionKind::Noul {
            instructions: "   ".to_string(),
        };
        assert_eq!(
            blank_inst.validate(),
            Err(DecisionValidationError::EmptyQuestionInstructions {
                question_id: "q_in_scope".to_string(),
            })
        );
    }

    #[test]
    fn guardrail_3_enforces_choice_fallback_and_noul_scope_gate() {
        let mut no_fallback = valid_batch();
        no_fallback.questions[1].1 = DecisionQuestionKind::Choice {
            instructions: "Pick a cell".to_string(),
            choices: vec!["G7".to_string(), "H8".to_string()],
            scope_gate_question_id: "q_in_scope".to_string(),
        };
        assert_eq!(
            no_fallback.validate(),
            Err(DecisionValidationError::MissingChoiceFallback {
                question_id: "q_target_cell".to_string(),
            })
        );

        let mut bad_gate = valid_batch();
        bad_gate.questions[1].1 = DecisionQuestionKind::Choice {
            instructions: "Pick a cell".to_string(),
            choices: vec!["G7".to_string(), MANDATORY_CHOICE_FALLBACK.to_string()],
            scope_gate_question_id: "missing_gate".to_string(),
        };
        assert_eq!(
            bad_gate.validate(),
            Err(DecisionValidationError::MissingOrInvalidScopeGate {
                question_id: "q_target_cell".to_string(),
                scope_gate_question_id: "missing_gate".to_string(),
            })
        );
    }

    #[test]
    fn guardrail_4_rejects_gs_and_youtube_uris() {
        let mut gs_batch = valid_batch();
        gs_batch.media_source = DecisionMediaSource::FilesApi {
            file_uri: "gs://bucket/frame.webp".to_string(),
        };
        assert!(matches!(
            gs_batch.validate(),
            Err(DecisionValidationError::UnsupportedMediaSource { .. })
        ));

        let mut yt_batch = valid_batch();
        yt_batch.media_source = DecisionMediaSource::FilesApi {
            file_uri: "https://www.youtube.com/watch?v=dQw4w9WgXcQ".to_string(),
        };
        assert!(matches!(
            yt_batch.validate(),
            Err(DecisionValidationError::UnsupportedMediaSource { .. })
        ));
    }

    #[test]
    fn rubric_decomposition_produces_noul_checklist() {
        let checklist = decompose_rubric_to_noul_checklist(
            "ui_ready",
            &[
                "Window is focused",
                "Dialog has no modal blocker",
                "Confirm button is enabled",
            ],
        );
        assert_eq!(checklist.len(), 3);
        assert_eq!(checklist[0].0, "ui_ready_check_1");
        assert_eq!(
            checklist[2].1,
            DecisionQuestionKind::Noul {
                instructions: "Confirm button is enabled".to_string(),
            }
        );
    }

    #[test]
    fn route_decision_respects_safety_and_confidence_thresholds() {
        assert_eq!(
            route_decision(0.95, "G7", 0.92, true, DEFAULT_FAST_PATH_CONFIDENCE),
            ControlPlaneRoute::RequireHumanVerification {
                reason: "safety_gate_triggered",
            }
        );

        assert!(matches!(
            route_decision(0.95, "G7", 0.92, false, DEFAULT_FAST_PATH_CONFIDENCE),
            ControlPlaneRoute::FastPathKoma { confidence } if confidence >= DEFAULT_FAST_PATH_CONFIDENCE
        ));

        assert!(matches!(
            route_decision(
                0.95,
                MANDATORY_CHOICE_FALLBACK,
                0.92,
                false,
                DEFAULT_FAST_PATH_CONFIDENCE
            ),
            ControlPlaneRoute::FallbackTurnPlanner { .. }
        ));
    }

    #[test]
    fn live_session_config_validates_fps_compression_and_epoch_fences() {
        let valid = LiveSessionConfig::default();
        assert_eq!(valid.model, "gemini-2.5-flash-live");
        assert_eq!(valid.uplink_sample_rate_hz, 16_000);
        assert_eq!(valid.downlink_sample_rate_hz, 24_000);
        assert_eq!(valid.max_screen_fps, 2);
        assert!(valid.enable_affective_dialog);
        assert!(!valid.proactive_audio);
        assert_eq!(valid.compression.trigger_tokens, 24_000);
        assert_eq!(valid.validate(), Ok(()));

        let mut custom_valid = valid.clone();
        custom_valid.uplink_sample_rate_hz = 24_000;
        custom_valid.downlink_sample_rate_hz = 48_000;
        custom_valid.max_screen_fps = 5;
        custom_valid.proactive_audio = true;
        assert_eq!(custom_valid.validate(), Ok(()));

        let mut invalid_zero_fps = valid.clone();
        invalid_zero_fps.max_screen_fps = 0;
        assert!(invalid_zero_fps.validate().is_err());

        let mut invalid_high_fps = valid.clone();
        invalid_high_fps.max_screen_fps = 120;
        assert!(invalid_high_fps.validate().is_err());

        let mut invalid_uplink_hz = valid.clone();
        invalid_uplink_hz.uplink_sample_rate_hz = 0;
        assert!(invalid_uplink_hz.validate().is_err());

        let mut invalid_downlink_hz = valid.clone();
        invalid_downlink_hz.downlink_sample_rate_hz = 96_000;
        assert!(invalid_downlink_hz.validate().is_err());

        let mut missing_fence = valid;
        missing_fence.require_koma_epoch_fence = false;
        assert!(missing_fence.validate().is_err());
    }

    #[test]
    fn custom_grid_spec_and_decision_compiler_config_work_with_defaults_and_overrides() {
        let default_grid = GridSpec::default();
        assert_eq!(default_grid.cols, 15);
        assert_eq!(default_grid.rows, 15);
        assert_eq!(default_grid.total_cells(), 225);
        assert!(default_grid.is_valid_for_decisions_api());

        let grid_10x10 = GridSpec { cols: 10, rows: 10 };
        assert_eq!(grid_10x10.total_cells(), 100);
        assert!(grid_10x10.is_valid_for_decisions_api());
        assert_eq!(grid_10x10.parse_cell_label("J10"), Some((9, 9)));
        assert_eq!(grid_10x10.parse_cell_label("K10"), None);
        assert_eq!(grid_10x10.parse_cell_label("A11"), None);

        let oversized_grid = GridSpec { cols: 20, rows: 20 };
        assert!(!oversized_grid.is_valid_for_decisions_api());

        let mut compiler = DecisionCompilerConfig {
            grid: grid_10x10,
            stop_sentinel: "halt_now".to_string(),
            fallback_token: "CUSTOM_FALLBACK".to_string(),
            time_slot_min_confidence: 0.70,
            ..DecisionCompilerConfig::default()
        };
        assert!((compiler.fast_path_confidence - 0.85).abs() < 1e-6);
        assert!((compiler.min_cell_prob - 0.05).abs() < 1e-6);
        assert!((compiler.bernoulli_activation_threshold - 0.80).abs() < 1e-6);
        assert!((compiler.probe_min_prob - 0.15).abs() < 1e-6);

        let dist_10x10 = vec![
            ("E5".to_string(), 0.60),
            ("F5".to_string(), 0.30),
            ("CUSTOM_FALLBACK".to_string(), 0.10),
        ];
        let centroid = compiler
            .interpolate_grid(&dist_10x10, 1000.0, 1000.0)
            .expect("expected valid 10x10 centroid");
        assert_eq!(centroid.top_cell, "E5");
        assert!((centroid.combined_cluster_mass - 0.90).abs() < 1e-5);
        assert!((centroid.y - 450.0).abs() < 1e-4);
        assert!(centroid.x > 450.0 && centroid.x < 550.0);

        let slots = vec![
            TimeSlotSelection {
                choice: "ctrl_a".to_string(),
                confidence: 0.72,
            },
            TimeSlotSelection {
                choice: "halt_now".to_string(),
                confidence: 0.99,
            },
            TimeSlotSelection {
                choice: "ctrl_v".to_string(),
                confidence: 0.95,
            },
        ];
        assert_eq!(compiler.compile_time_slot(&slots), vec!["ctrl_a"]);

        let probes = compiler.compile_probe(&dist_10x10);
        assert_eq!(
            probes,
            vec![("E5".to_string(), 0.60), ("F5".to_string(), 0.30)]
        );

        assert!(matches!(
            compiler.route(0.95, "CUSTOM_FALLBACK", 0.95, false),
            ControlPlaneRoute::FallbackTurnPlanner { .. }
        ));

        compiler.bernoulli_activation_threshold = 0.90;
        let bernoulli = vec![
            BernoulliStepCandidate {
                key: "step_b".to_string(),
                use_noul_prob: 0.95,
                step_probabilities: vec![(2, 1.0)],
            },
            BernoulliStepCandidate {
                key: "step_a".to_string(),
                use_noul_prob: 0.85,
                step_probabilities: vec![(1, 1.0)],
            },
        ];
        let active = compiler.compile_bernoulli_steps(&bernoulli);
        assert_eq!(active, vec![("step_b".to_string(), 2.0)]);

        let default_policy = DecisionRouterPolicy::default();
        assert_eq!(
            default_policy.decision_model_id.as_deref(),
            Some("gemini-decision-1")
        );
        assert_eq!(default_policy.primary_text_model_id, "gemini-2.5-pro");
        assert_eq!(default_policy.max_questions_per_batch, 8);
        assert_eq!(
            default_policy.resolve_mode(),
            Ok(DecisionExecutionMode::NativeDecisionsApi {
                provider: "gemini".to_string(),
                model: "gemini-decision-1".to_string(),
            })
        );

        let fallback_policy = DecisionRouterPolicy {
            decision_model_id: None,
            primary_provider: "openai".to_string(),
            primary_text_model_id: "gpt-4o".to_string(),
            ..DecisionRouterPolicy::default()
        };
        assert_eq!(
            fallback_policy.resolve_mode(),
            Ok(DecisionExecutionMode::EmulatedStructuredTextModel {
                provider: "openai".to_string(),
                model: "gpt-4o".to_string(),
                enforce_json_schema: true,
            })
        );
    }

    #[test]
    fn decision_compiler_interpolates_grid_225_sub_cell_centroid() {
        let dist = vec![
            ("L13".to_string(), 0.54),
            ("M13".to_string(), 0.41),
            ("A1".to_string(), 0.03),
            (MANDATORY_CHOICE_FALLBACK.to_string(), 0.02),
        ];

        let centroid = interpolate_grid_225_centroid(&dist, 750.0, 750.0, 0.05)
            .expect("expected valid sub-grid centroid");

        assert_eq!(centroid.top_cell, "L13");
        assert!((centroid.top_single_prob - 0.54).abs() < 1e-5);
        assert!((centroid.combined_cluster_mass - 0.95).abs() < 1e-5);
        assert!((centroid.y - 625.0).abs() < 1e-4);
        assert!(centroid.x > 575.0 && centroid.x < 625.0);
    }

    #[test]
    fn decision_compiler_strategy_a_compiles_time_slot_macros_and_truncates() {
        let five_step = vec![
            TimeSlotSelection {
                choice: "ctrl_a".to_string(),
                confidence: 0.96,
            },
            TimeSlotSelection {
                choice: "ctrl_c".to_string(),
                confidence: 0.94,
            },
            TimeSlotSelection {
                choice: "ctrl_t".to_string(),
                confidence: 0.91,
            },
            TimeSlotSelection {
                choice: "ctrl_v".to_string(),
                confidence: 0.93,
            },
            TimeSlotSelection {
                choice: "ctrl_s".to_string(),
                confidence: 0.89,
            },
            TimeSlotSelection {
                choice: "stop_done".to_string(),
                confidence: 0.98,
            },
        ];
        assert_eq!(
            compile_time_slot_macro(&five_step, "stop_done", 0.80),
            vec!["ctrl_a", "ctrl_c", "ctrl_t", "ctrl_v", "ctrl_s"]
        );

        let repeated_keys = vec![
            TimeSlotSelection {
                choice: "ctrl_shift_p".to_string(),
                confidence: 0.95,
            },
            TimeSlotSelection {
                choice: "down".to_string(),
                confidence: 0.92,
            },
            TimeSlotSelection {
                choice: "down".to_string(),
                confidence: 0.90,
            },
            TimeSlotSelection {
                choice: "enter".to_string(),
                confidence: 0.94,
            },
            TimeSlotSelection {
                choice: "stop_done".to_string(),
                confidence: 0.97,
            },
            TimeSlotSelection {
                choice: "enter".to_string(),
                confidence: 0.99,
            },
        ];
        assert_eq!(
            compile_time_slot_macro(&repeated_keys, "stop_done", 0.80),
            vec!["ctrl_shift_p", "down", "down", "enter"]
        );

        let low_confidence = vec![
            TimeSlotSelection {
                choice: "ctrl_a".to_string(),
                confidence: 0.95,
            },
            TimeSlotSelection {
                choice: "ctrl_c".to_string(),
                confidence: 0.62,
            },
            TimeSlotSelection {
                choice: "ctrl_v".to_string(),
                confidence: 0.95,
            },
        ];
        assert_eq!(
            compile_time_slot_macro(&low_confidence, "stop_done", 0.80),
            vec!["ctrl_a"]
        );
    }

    #[test]
    fn decision_compiler_strategy_b_orders_bernoulli_candidates_by_expected_step() {
        let candidates = vec![
            BernoulliStepCandidate {
                key: "ctrl_v".to_string(),
                use_noul_prob: 0.94,
                step_probabilities: vec![(1, 0.01), (2, 0.04), (3, 0.90), (4, 0.05)],
            },
            BernoulliStepCandidate {
                key: "ctrl_a".to_string(),
                use_noul_prob: 0.98,
                step_probabilities: vec![(1, 0.95), (2, 0.05)],
            },
            BernoulliStepCandidate {
                key: "alt_f4".to_string(),
                use_noul_prob: 0.02,
                step_probabilities: vec![(1, 0.90)],
            },
            BernoulliStepCandidate {
                key: "ctrl_c".to_string(),
                use_noul_prob: 0.96,
                step_probabilities: vec![(1, 0.05), (2, 0.90), (3, 0.05)],
            },
            BernoulliStepCandidate {
                key: "ctrl_z".to_string(),
                use_noul_prob: 0.12,
                step_probabilities: vec![(1, 0.80)],
            },
        ];

        let compiled = compile_bernoulli_step_macro(&candidates, 0.80);
        let keys: Vec<&str> = compiled.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, vec!["ctrl_a", "ctrl_c", "ctrl_v"]);
        assert!(compiled[0].1 < compiled[1].1 && compiled[1].1 < compiled[2].1);
    }

    #[test]
    fn decision_compiler_strategy_c_ranks_threshold_probes_descending() {
        let probs = vec![
            ("space".to_string(), 0.09),
            ("escape".to_string(), 0.62),
            ("ctrl_w".to_string(), 0.03),
            ("q".to_string(), 0.24),
            (MANDATORY_CHOICE_FALLBACK.to_string(), 0.85),
        ];

        let probes = compile_threshold_probe(&probs, 0.15);
        assert_eq!(
            probes,
            vec![("escape".to_string(), 0.62), ("q".to_string(), 0.24)]
        );
    }

    #[test]
    fn resolve_decision_mode_selects_native_api_for_gemini_decision_1() {
        let mode = resolve_decision_execution_mode(
            "openai",
            "gpt-4o",
            Some("gemini"),
            Some("gemini-decision-1"),
        )
        .expect("expected native decision mode");
        assert_eq!(
            mode,
            DecisionExecutionMode::NativeDecisionsApi {
                provider: "gemini".to_string(),
                model: GEMINI_DECISION_1_MODEL_ID.to_string(),
            }
        );
        assert!(
            resolve_decision_execution_mode(
                "gemini",
                "gemini-2.5-pro",
                Some("gemini"),
                Some("gemini-decision-1-latest"),
            )
            .is_err()
        );
    }

    #[test]
    fn resolve_decision_mode_falls_back_to_primary_text_model_when_unconfigured() {
        let mode_none = resolve_decision_execution_mode("openai", "gpt-4o", None, None)
            .expect("fallback on None must never error");
        assert_eq!(
            mode_none,
            DecisionExecutionMode::EmulatedStructuredTextModel {
                provider: "openai".to_string(),
                model: "gpt-4o".to_string(),
                enforce_json_schema: true,
            }
        );

        let mode_empty =
            resolve_decision_execution_mode("anthropic", "claude-3-7-sonnet", Some(""), Some("  "))
                .expect("fallback on empty string must never error");
        assert_eq!(
            mode_empty,
            DecisionExecutionMode::EmulatedStructuredTextModel {
                provider: "anthropic".to_string(),
                model: "claude-3-7-sonnet".to_string(),
                enforce_json_schema: true,
            }
        );
    }

    #[test]
    fn resolve_decision_mode_uses_emulated_structured_model_for_custom_text_or_local_vlm() {
        let mode = resolve_decision_execution_mode(
            "openai",
            "gpt-4o",
            Some("local_vlm"),
            Some("qwen2.5-vl-7b"),
        )
        .expect("custom local VLM decision role must succeed");
        assert_eq!(
            mode,
            DecisionExecutionMode::EmulatedStructuredTextModel {
                provider: "local_vlm".to_string(),
                model: "qwen2.5-vl-7b".to_string(),
                enforce_json_schema: true,
            }
        );
    }

    #[test]
    fn synthesize_emulated_decision_prompt_includes_all_questions_and_schema_constraints() {
        let questions = vec![
            (
                "q_in_scope".to_string(),
                DecisionQuestionKind::Noul {
                    instructions: "Is the terminal visible?".to_string(),
                },
            ),
            (
                "q_action".to_string(),
                DecisionQuestionKind::Choice {
                    instructions: "Select the best shortcut".to_string(),
                    choices: vec![
                        "ctrl_c".to_string(),
                        "ctrl_v".to_string(),
                        MANDATORY_CHOICE_FALLBACK.to_string(),
                    ],
                    scope_gate_question_id: "q_in_scope".to_string(),
                },
            ),
            (
                "q_quality".to_string(),
                DecisionQuestionKind::Score {
                    instructions: "Rate layout clarity".to_string(),
                    rubric: vec!["Poor".to_string(), "Good".to_string()],
                },
            ),
        ];

        let prompt = synthesize_emulated_decision_prompt(&questions);
        assert!(prompt.contains("{\"answers\": { ... }}"));
        assert!(prompt.contains("q_in_scope"));
        assert!(prompt.contains("type=\"noul\""));
        assert!(prompt.contains("q_action"));
        assert!(prompt.contains("type=\"choice\""));
        assert!(prompt.contains("ctrl_c, ctrl_v, NONE_OR_OTHER"));
        assert!(prompt.contains("q_quality"));
        assert!(prompt.contains("type=\"score\""));
        assert!(prompt.contains("Poor | Good"));
    }
}
