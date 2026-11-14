//! Pure-data model driver contracts and validation invariants for the Jiku
//! desktop orchestration daemon (`jikud`).
//!
//! This crate remains a library-only incubation scaffold prior to `v0.1.0`:
//! network I/O, runtime daemon processes, policy execution, and Koma IPC
//! integration are not implemented yet.
//!
//! The contracts defined here capture the **Three-Driver Model Architecture**:
//! 1. [`ModelDriverKind::Decisions`] — Structured single-pass multi-question
//!    evaluation via `POST /v1beta/decisions` (`gemini-decision-1`).
//! 2. [`ModelDriverKind::TurnSse`] — Autoregressive HTTP/2 SSE turn generation
//!    for multi-step planning, Lua synthesis, and local VLM guards.
//! 3. [`ModelDriverKind::LiveWebSocket`] — Stateful full-duplex WebSocket
//!    streaming (`BidiGenerateContent`) for `jiku live` real-time voice/screen
//!    co-piloting.

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

/// Required uplink PCM audio sample rate (`16 kHz`) for `LiveSessionDriver`.
pub const UPLINK_AUDIO_SAMPLE_RATE_HZ: u32 = 16_000;

/// Native downlink PCM audio sample rate (`24 kHz`) from `LiveSessionDriver`.
pub const DOWNLINK_AUDIO_SAMPLE_RATE_HZ: u32 = 24_000;

/// Maximum Koma screen capture frame rate (`2 FPS`) streamed to a Live session.
pub const MAX_SCREEN_FPS: u8 = 2;

/// Primary response modality for a `LiveSessionDriver` WebSocket session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LiveResponseModality {
    /// Native audio output stream (`AUDIO`, 24 kHz PCM) with parallel transcription.
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

/// Pure-data session configuration for `LiveSessionDriver` (`jiku live`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveSessionConfig {
    /// Live-capable model identifier (e.g., `"gemini-2.5-flash-live"`).
    pub model: String,
    /// Single primary response modality (`Audio` or `Text`; never both).
    pub response_modality: LiveResponseModality,
    /// Request plain-text transcription alongside native audio output.
    pub output_audio_transcription: bool,
    /// Allow the model to remain silent when ambient speech is not addressed to Jiku.
    pub proactive_audio: bool,
    /// Maximum screen frame rate (`1..=2` FPS).
    pub max_screen_fps: u8,
    /// Context window compression thresholds.
    pub compression: ContextWindowCompressionSpec,
    /// Reconnect transparently across ~10-minute WebSocket connection resets.
    pub session_resumption_enabled: bool,
    /// Require every asynchronous Koma action to carry an epoch fence.
    pub require_koma_epoch_fence: bool,
}

impl LiveSessionConfig {
    /// Validates `LiveSessionDriver` session invariants:
    /// - `model` must not be empty.
    /// - `max_screen_fps` must be in `1..=MAX_SCREEN_FPS` (`2` FPS).
    /// - `compression.trigger_tokens` must be positive and strictly greater than
    ///   `compression.sliding_window_target_tokens` (which must also be positive).
    /// - `session_resumption_enabled` and `require_koma_epoch_fence` must be enabled.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.model.trim().is_empty() {
            return Err("LiveSessionConfig model must not be empty");
        }
        if self.max_screen_fps == 0 || self.max_screen_fps > MAX_SCREEN_FPS {
            return Err("LiveSessionConfig max_screen_fps must be between 1 and 2 FPS");
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

/// Sub-grid probability-weighted centroid interpolated from Koma's 15x15
/// `grid_225` (`A1..O15`) categorical distribution.
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

fn parse_grid_225_label(label: &str) -> Option<(i32, i32)> {
    let trimmed = label.trim();
    let mut chars = trimmed.chars();
    let col_char = chars.next()?;
    if !('A'..='O').contains(&col_char) {
        return None;
    }
    let row_str: &str = chars.as_str();
    let row_1_based: i32 = row_str.parse().ok()?;
    if !(1..=15).contains(&row_1_based) {
        return None;
    }
    let col = i32::from(col_char as u8 - b'A');
    let row = row_1_based - 1;
    Some((col, row))
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
    if !width.is_finite() || !height.is_finite() || width <= 0.0 || height <= 0.0 {
        return None;
    }

    let parsed: Vec<(&str, i32, i32, f32)> = probabilities
        .iter()
        .filter_map(|(label, prob)| {
            if !prob.is_finite() || *prob <= 0.0 {
                return None;
            }
            let (col, row) = parse_grid_225_label(label)?;
            Some((label.as_str(), col, row, *prob))
        })
        .collect();

    let &(top_label, tc, tr, top_prob) = parsed.iter().max_by(|a, b| a.3.total_cmp(&b.3))?;

    if top_prob < min_cell_prob {
        return None;
    }

    let cell_w = width / 15.0;
    let cell_h = height / 15.0;

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
    let mut compiled = Vec::with_capacity(slots.len());
    for slot in slots {
        let token = slot.choice.trim();
        if token.is_empty()
            || token == stop_sentinel
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
    let mut probes: Vec<(String, f32)> = probabilities
        .iter()
        .filter_map(|(candidate, prob)| {
            let token = candidate.trim();
            if token.is_empty()
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
        let valid = LiveSessionConfig {
            model: "gemini-2.5-flash-live".to_string(),
            response_modality: LiveResponseModality::Audio,
            output_audio_transcription: true,
            proactive_audio: true,
            max_screen_fps: MAX_SCREEN_FPS,
            compression: ContextWindowCompressionSpec {
                trigger_tokens: 10_000,
                sliding_window_target_tokens: 2_000,
            },
            session_resumption_enabled: true,
            require_koma_epoch_fence: true,
        };
        assert_eq!(valid.validate(), Ok(()));

        let mut invalid_fps = valid.clone();
        invalid_fps.max_screen_fps = 5;
        assert!(invalid_fps.validate().is_err());

        let mut missing_fence = valid;
        missing_fence.require_koma_epoch_fence = false;
        assert!(missing_fence.validate().is_err());
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
}
