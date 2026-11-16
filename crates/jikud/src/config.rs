//! Runtime TOML configuration schema, validation, and environment/secret resolution.

use std::{
    collections::BTreeMap,
    env, fs,
    path::{Path, PathBuf},
};

use jiku_memory::{DEFAULT_APP_QUIRK_MAX_CHARS, DEFAULT_L0_FRAME_RING_SIZE, expand_tilde_path};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub use crate::context_budget::ContextConfig;
use crate::{
    ContextWindowCompressionSpec, DEFAULT_FAST_PATH_CONFIDENCE, DEFAULT_SCREEN_FPS,
    DOWNLINK_AUDIO_SAMPLE_RATE_HZ, DecisionCompilerConfig, DecisionRouterPolicy,
    GEMINI_DECISION_1_MODEL_ID, GridSpec, LiveResponseModality, LiveSessionConfig,
    MANDATORY_CHOICE_FALLBACK, UPLINK_AUDIO_SAMPLE_RATE_HZ, resolve_decision_execution_mode,
};

/// Errors produced when loading or validating [`JikuConfig`].
#[derive(Debug, Error)]
pub enum ConfigError {
    /// Failed reading a configuration file from disk.
    #[error("failed reading config file {path}: {source}")]
    Read {
        /// Config path attempted.
        path: PathBuf,
        /// Underlying I/O error.
        source: std::io::Error,
    },

    /// Failed parsing TOML syntax or schema.
    #[error("failed parsing config TOML: {0}")]
    Parse(#[from] toml::de::Error),

    /// Semantic validation failed.
    #[error("invalid Jiku configuration: {0}")]
    Invalid(String),
}

/// Complete runtime configuration for `jikud` and `jiku`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct JikuConfig {
    /// Daemon locale, socket filename, dropzone path, and audit log switch.
    pub daemon: DaemonConfig,
    /// Koma IPC defaults and perceptual-hash thresholds.
    pub koma: KomaTargetConfig,
    /// Configured model providers keyed by logical ID (`"gemini"`, `"opencode_go"`, ...).
    pub providers: BTreeMap<String, ProviderSpec>,
    /// Role bindings (`planner`, `actor`, `decision_router`, `live_companion`, `fast_guard`).
    pub roles: BTreeMap<String, RoleSpec>,
    /// Unified decision router policy.
    pub decision_router: DecisionRouterToml,
    /// Full-duplex live session parameters.
    pub live_session: LiveSessionToml,
    /// `~/JikuBox/` batch triage configuration.
    pub jikubox: JikuBoxTable,
    /// Three-tier SQLite WAL + FTS5 memory settings.
    pub memory: MemoryToml,
    /// Desktop notification settings.
    pub notify: NotifyToml,
    /// Optional TTS synthesizer settings (`enabled = false` by default).
    pub tts: TtsToml,
    /// Companion EventBus cue emission flags.
    pub avatar: AvatarToml,
    /// Prompt token budgeting and progressive compaction settings (`[context]`).
    #[serde(default)]
    pub context: ContextConfig,
}

impl Default for JikuConfig {
    fn default() -> Self {
        let mut providers = BTreeMap::new();
        providers.insert(
            "gemini".to_string(),
            ProviderSpec {
                kind: ProviderKind::GoogleGemini,
                base_url: "https://generativelanguage.googleapis.com/v1beta".to_string(),
                api_key_env: "GEMINI_API_KEY".to_string(),
                thinking: None,
            },
        );
        providers.insert(
            "opencode_go".to_string(),
            ProviderSpec {
                kind: ProviderKind::OpenCodeGo,
                base_url: "https://opencode.ai/inference/go/openai/v1".to_string(),
                api_key_env: "OPENCODE_GO_API_KEY".to_string(),
                thinking: Some(false),
            },
        );

        let mut roles = BTreeMap::new();
        roles.insert(
            "planner".to_string(),
            RoleSpec {
                provider: "opencode_go".to_string(),
                model: "deepseek-v4.1-flash".to_string(),
                driver: "turn_sse".to_string(),
                thinking: Some(false),
                temperature: Some(0.2),
                ..RoleSpec::default()
            },
        );
        roles.insert(
            "actor".to_string(),
            RoleSpec {
                provider: "gemini".to_string(),
                model: GEMINI_DECISION_1_MODEL_ID.to_string(),
                driver: "decisions".to_string(),
                temperature: Some(0.0),
                coordinate_mode: Some("grid_225_or_normalized_1000".to_string()),
                ..RoleSpec::default()
            },
        );
        roles.insert(
            "decision_router".to_string(),
            RoleSpec {
                provider: "gemini".to_string(),
                model: GEMINI_DECISION_1_MODEL_ID.to_string(),
                driver: "decisions".to_string(),
                optional: true,
                fallback_to_primary_text_model: Some(true),
                fast_path_confidence: Some(DEFAULT_FAST_PATH_CONFIDENCE),
                max_questions_per_batch: Some(8),
                require_none_or_other: Some(true),
                require_scope_gate: Some(true),
                ..RoleSpec::default()
            },
        );

        Self {
            daemon: DaemonConfig::default(),
            koma: KomaTargetConfig::default(),
            providers,
            roles,
            decision_router: DecisionRouterToml::default(),
            live_session: LiveSessionToml::default(),
            jikubox: JikuBoxTable::default(),
            memory: MemoryToml::default(),
            notify: NotifyToml::default(),
            tts: TtsToml::default(),
            avatar: AvatarToml::default(),
            context: ContextConfig::default(),
        }
    }
}

impl JikuConfig {
    /// Parses a TOML configuration document from `path`.
    pub fn load_from_path(path: &Path) -> Result<Self, ConfigError> {
        let resolved = expand_tilde_path(path);
        let raw = fs::read_to_string(&resolved).map_err(|source| ConfigError::Read {
            path: resolved,
            source,
        })?;
        let cfg: Self = toml::from_str(&raw)?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Resolves configuration from explicit `--config`, `$JIKU_CONFIG`,
    /// `$XDG_CONFIG_HOME/jiku/config.toml` / `~/.config/jiku/config.toml`,
    /// or returns validated built-in defaults if none of those files exist.
    pub fn load_resolved(
        explicit_path: Option<&Path>,
    ) -> Result<(Self, Option<PathBuf>), ConfigError> {
        if let Some(p) = explicit_path {
            let cfg = Self::load_from_path(p)?;
            return Ok((cfg, Some(expand_tilde_path(p))));
        }

        if let Some(env_path) = env::var_os("JIKU_CONFIG").filter(|v| !v.is_empty()) {
            let p = expand_tilde_path(Path::new(&env_path));
            let cfg = Self::load_from_path(&p)?;
            return Ok((cfg, Some(p)));
        }

        let xdg_candidate = env::var_os("XDG_CONFIG_HOME")
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
            .or_else(|| {
                env::var_os("HOME")
                    .filter(|v| !v.is_empty())
                    .map(|h| PathBuf::from(h).join(".config"))
            })
            .map(|dir| dir.join("jiku").join("config.toml"));

        if let Some(p) = xdg_candidate
            && p.is_file()
        {
            let cfg = Self::load_from_path(&p)?;
            return Ok((cfg, Some(p)));
        }

        let def = Self::default();
        def.validate()?;
        Ok((def, None))
    }

    /// Validates all provider references, compiler bounds, grid constraints, and
    /// `LiveSessionConfig` invariants.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.daemon.socket_name.trim().is_empty() {
            return Err(ConfigError::Invalid(
                "daemon.socket_name must not be empty".to_string(),
            ));
        }

        for (name, spec) in &self.providers {
            if spec.base_url.trim().is_empty() {
                return Err(ConfigError::Invalid(format!(
                    "providers.{name}.base_url must not be empty"
                )));
            }
        }

        for (role_name, role) in &self.roles {
            if !self.providers.contains_key(&role.provider) {
                return Err(ConfigError::Invalid(format!(
                    "roles.{role_name}.provider references unknown provider `{}`",
                    role.provider
                )));
            }
            if role.model.trim().is_empty() {
                return Err(ConfigError::Invalid(format!(
                    "roles.{role_name}.model must not be empty"
                )));
            }
        }

        let policy = self.to_decision_router_policy();
        policy
            .resolve_mode()
            .map_err(|msg| ConfigError::Invalid(msg.to_string()))?;
        if !policy.compiler.grid.is_valid_for_decisions_api() {
            return Err(ConfigError::Invalid(format!(
                "decision_router.compiler.grid {}x{} must yield 2..=255 total cells and cols <= 26",
                policy.compiler.grid.cols, policy.compiler.grid.rows
            )));
        }
        if self.decision_router.max_questions_per_batch == 0
            || self.decision_router.max_questions_per_batch > 1000
        {
            return Err(ConfigError::Invalid(
                "decision_router.max_questions_per_batch must be in 1..=1000".to_string(),
            ));
        }

        self.to_live_session_config()
            .validate()
            .map_err(|msg| ConfigError::Invalid(msg.to_string()))?;

        if self.memory.l0_frame_ring_size == 0 || self.memory.app_quirk_max_chars == 0 {
            return Err(ConfigError::Invalid(
                "memory.l0_frame_ring_size and memory.app_quirk_max_chars must be > 0".to_string(),
            ));
        }

        Ok(())
    }

    /// Converts TOML policy tables into a runtime [`DecisionRouterPolicy`].
    #[must_use]
    pub fn to_decision_router_policy(&self) -> DecisionRouterPolicy {
        let mut policy = DecisionRouterPolicy {
            mode: Default::default(),
            decision_provider: self.decision_router.decision_provider.clone(),
            decision_model_id: self.decision_router.decision_model_id.clone(),
            primary_provider: self.decision_router.primary_provider.clone(),
            primary_text_model_id: self.decision_router.primary_text_model_id.clone(),
            fast_path_confidence_threshold: self.decision_router.fast_path_confidence_threshold,
            max_questions_per_batch: self.decision_router.max_questions_per_batch,
            compiler: DecisionCompilerConfig {
                grid: GridSpec {
                    cols: self.decision_router.compiler.grid.cols,
                    rows: self.decision_router.compiler.grid.rows,
                },
                fast_path_confidence: self.decision_router.compiler.fast_path_confidence,
                min_cell_prob: self.decision_router.compiler.min_cell_prob,
                time_slot_min_confidence: self.decision_router.compiler.time_slot_min_confidence,
                bernoulli_activation_threshold: self
                    .decision_router
                    .compiler
                    .bernoulli_activation_threshold,
                probe_min_prob: self.decision_router.compiler.probe_min_prob,
                stop_sentinel: self.decision_router.compiler.stop_sentinel.clone(),
                fallback_token: self.decision_router.compiler.fallback_token.clone(),
            },
        };
        if let Ok(resolved) = resolve_decision_execution_mode(
            &policy.primary_provider,
            &policy.primary_text_model_id,
            policy.decision_provider.as_deref(),
            policy.decision_model_id.as_deref(),
        ) {
            policy.mode = resolved;
        }
        policy
    }

    /// Converts TOML `[live_session]` into a validated [`LiveSessionConfig`].
    #[must_use]
    pub fn to_live_session_config(&self) -> LiveSessionConfig {
        let modality = match self
            .live_session
            .response_modality
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "text" => LiveResponseModality::Text,
            _ => LiveResponseModality::Audio,
        };
        LiveSessionConfig {
            model: self.live_session.model.clone(),
            response_modality: modality,
            output_audio_transcription: self.live_session.output_audio_transcription,
            proactive_audio: self.live_session.proactive_audio,
            enable_affective_dialog: self.live_session.enable_affective_dialog,
            uplink_sample_rate_hz: self.live_session.uplink_sample_rate_hz,
            downlink_sample_rate_hz: self.live_session.downlink_sample_rate_hz,
            max_screen_fps: self.live_session.max_screen_fps,
            compression: ContextWindowCompressionSpec {
                trigger_tokens: self.live_session.context_compression_trigger_tokens,
                sliding_window_target_tokens: self.live_session.sliding_window_target_tokens,
            },
            session_resumption_enabled: self.live_session.session_resumption_enabled,
            require_koma_epoch_fence: self.live_session.require_koma_epoch_fence,
        }
    }

    /// Resolves whether extended thinking mode is enabled for a role/provider pair.
    ///
    /// Precedence: explicit `role.thinking` -> `provider.thinking` -> default
    /// (`false` for `opencode_go`, `true` otherwise).
    #[must_use]
    pub fn resolve_thinking_enabled(
        &self,
        role: Option<&RoleSpec>,
        provider: &ProviderSpec,
    ) -> bool {
        if let Some(flag) = role.and_then(|r| r.thinking) {
            return flag;
        }
        if let Some(flag) = provider.thinking {
            return flag;
        }
        !matches!(provider.kind, ProviderKind::OpenCodeGo)
    }

    /// Resolves the planner role (`[roles.planner]`) or synthesizes it from
    /// `[decision_router]`.
    #[must_use]
    pub fn planner_binding(&self) -> (String, String, Option<&RoleSpec>) {
        if let Some(role) = self.roles.get("planner") {
            return (role.provider.clone(), role.model.clone(), Some(role));
        }
        (
            self.decision_router.primary_provider.clone(),
            self.decision_router.primary_text_model_id.clone(),
            None,
        )
    }

    /// Looks up a configured provider API key from the process environment.
    /// Returns `Ok(None)` when `api_key_env` is empty (local vLLM / Ollama),
    /// and `Err` when the referenced env var is missing or empty.
    pub fn read_provider_api_key(&self, provider_id: &str) -> Result<Option<String>, ConfigError> {
        let spec = self
            .providers
            .get(provider_id)
            .ok_or_else(|| ConfigError::Invalid(format!("unknown provider `{provider_id}`")))?;
        let env_name = spec.api_key_env.trim();
        if env_name.is_empty() {
            return Ok(None);
        }
        match env::var(env_name) {
            Ok(val) if !val.trim().is_empty() => Ok(Some(val.trim().to_string())),
            _ => Err(ConfigError::Invalid(format!(
                "environment variable `{env_name}` is unset or empty for provider `{provider_id}`"
            ))),
        }
    }

    /// Resolves `$XDG_RUNTIME_DIR/<daemon.socket_name>` or a safe `/tmp/jiku-<uid>.sock` fallback.
    #[must_use]
    pub fn daemon_socket_path(&self) -> PathBuf {
        let sock = self.daemon.socket_name.trim();
        if let Some(dir) = env::var_os("XDG_RUNTIME_DIR").filter(|d| !d.is_empty()) {
            PathBuf::from(dir).join(sock)
        } else {
            let uid = rustix::process::getuid().as_raw();
            PathBuf::from(format!("/tmp/jiku-{uid}")).join(sock)
        }
    }

    /// Resolves the SQLite database file path with `~/` expansion.
    #[must_use]
    pub fn memory_db_path(&self) -> PathBuf {
        expand_tilde_path(Path::new(&self.memory.db_path))
    }

    /// Resolves the audit log path alongside the SQLite state database.
    #[must_use]
    pub fn audit_log_path(&self) -> PathBuf {
        let db = self.memory_db_path();
        db.parent()
            .map(|p| p.join("audit.jsonl"))
            .unwrap_or_else(|| PathBuf::from("/tmp/jiku-audit.jsonl"))
    }
}

/// Supported LLM / decision API provider wire families.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    /// Google AI Studio / Gemini REST API (`x-goog-api-key`, `/v1beta/decisions`, `:generateContent`).
    GoogleGemini,
    /// OpenCode Go OpenAI-compatible gateway (`x-opencode-session` UUID + optional `"thinking": {"type": "disabled"}`).
    #[serde(rename = "opencode_go")]
    OpenCodeGo,
    /// Standard OpenAI-compatible `/chat/completions` endpoint (OpenAI, OpenRouter, vLLM, llama.cpp, DeepSeek).
    #[serde(rename = "openai_compatible", alias = "open_ai_compatible")]
    OpenAiCompatible,
    /// Anthropic Messages API.
    Anthropic,
}

/// Declarative upstream provider configuration (`[providers.<id>]`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderSpec {
    /// Wire protocol family.
    pub kind: ProviderKind,
    /// Base URL excluding trailing slash.
    pub base_url: String,
    /// Name of the environment variable holding the bearer/API key (empty for unauthenticated local models).
    #[serde(default)]
    pub api_key_env: String,
    /// Optional explicit toggle for extended thinking/reasoning mode.
    #[serde(default)]
    pub thinking: Option<bool>,
}

/// Role routing entry (`[roles.<role>]`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct RoleSpec {
    /// Key under `[providers]`.
    pub provider: String,
    /// Upstream model identifier.
    pub model: String,
    /// Expected driver (`"decisions"`, `"turn_sse"`, `"live_websocket"`).
    pub driver: String,
    /// Override sampling temperature.
    pub temperature: Option<f32>,
    /// Override thinking/reasoning mode (`false` sends `"thinking": {"type": "disabled"}`).
    pub thinking: Option<bool>,
    /// Whether absence degrades cleanly instead of failing.
    pub optional: bool,
    /// Whether `decision_router` falls back to `planner`.
    pub fallback_to_primary_text_model: Option<bool>,
    /// Fast-path confidence override.
    pub fast_path_confidence: Option<f32>,
    /// Maximum questions per batch override.
    pub max_questions_per_batch: Option<usize>,
    /// Require `NONE_OR_OTHER` fallback.
    pub require_none_or_other: Option<bool>,
    /// Require companion `Noul` scope gate.
    pub require_scope_gate: Option<bool>,
    /// Preferred visual grounding format (`grid_225_or_normalized_1000`).
    pub coordinate_mode: Option<String>,
    /// Live session primary response modality (`"AUDIO"` | `"TEXT"`).
    pub response_modality: Option<String>,
    /// Live output transcription toggle.
    pub output_audio_transcription: Option<bool>,
    /// Live proactive audio toggle.
    pub proactive_audio: Option<bool>,
    /// Live uplink PCM rate in Hz.
    pub uplink_pcm_hz: Option<u32>,
    /// Live downlink PCM rate in Hz.
    pub downlink_pcm_hz: Option<u32>,
    /// Live max screen capture FPS.
    pub max_screen_fps: Option<u8>,
    /// Live context compression trigger tokens.
    pub trigger_tokens: Option<u32>,
    /// Live context compression target tokens.
    pub sliding_window_target_tokens: Option<u32>,
    /// Live session resumption toggle.
    pub session_resumption: Option<bool>,
    /// Non-blocking tool execution toggle.
    pub non_blocking_tools: Option<bool>,
    /// Tool scheduling mode (`INTERRUPT`, `WHEN_IDLE`, `SILENT`).
    pub default_tool_scheduling: Option<String>,
    /// Optional model context window ceiling in tokens (overrides `[context].max_context_tokens`).
    #[serde(default)]
    pub max_context_tokens: Option<u32>,
}

/// `[daemon]` settings.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct DaemonConfig {
    /// UI/notification locale (default `"zh-CN"`).
    pub locale: String,
    /// Socket filename under `$XDG_RUNTIME_DIR` (default `"jiku.sock"`).
    pub socket_name: String,
    /// Default triage dropzone path (default `"~/JikuBox"`).
    pub dropbox_dir: String,
    /// Emit structured JSON lines to `audit.jsonl` (default `true`).
    pub audit_log: bool,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            locale: "zh-CN".to_string(),
            socket_name: "jiku.sock".to_string(),
            dropbox_dir: "~/JikuBox".to_string(),
            audit_log: true,
        }
    }
}

/// `[koma]` compositor IPC defaults.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct KomaTargetConfig {
    /// Default Koma workspace ID (default `"ws_01"`).
    pub default_workspace: String,
    /// Default Koma frame capture overlay (`"grid_225"`).
    pub default_overlay: String,
    /// Maximum Hamming bit delta treated as unchanged screen state (default `4`).
    pub perceptual_hash_threshold: u32,
    /// Automatic Bezier hover-glide duration in milliseconds before off-target clicks or strokes
    /// (`0` disables automatic pre-step hover glides; default `140`).
    pub hover_glide_ms: u64,
    /// Sandbox application profiles allowlist.
    pub allowed_app_profiles: Vec<String>,
}

impl Default for KomaTargetConfig {
    fn default() -> Self {
        Self {
            default_workspace: "ws_01".to_string(),
            default_overlay: "grid_225".to_string(),
            perceptual_hash_threshold: 4,
            hover_glide_ms: 140,
            allowed_app_profiles: vec![
                "chromium-research".to_string(),
                "libreoffice-headless".to_string(),
                "gimp-batch".to_string(),
            ],
        }
    }
}

/// `[decision_router]` settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DecisionRouterToml {
    /// Dedicated decision provider key.
    pub decision_provider: Option<String>,
    /// Pinned decision model ID.
    pub decision_model_id: Option<String>,
    /// Primary fallback provider key.
    pub primary_provider: String,
    /// Primary fallback text/VLM model ID.
    pub primary_text_model_id: String,
    /// Fast-path gate confidence.
    pub fast_path_confidence_threshold: f32,
    /// Batch size limit.
    pub max_questions_per_batch: usize,
    /// Compiler hyperparameters.
    pub compiler: DecisionCompilerToml,
}

impl Default for DecisionRouterToml {
    fn default() -> Self {
        Self {
            decision_provider: Some("gemini".to_string()),
            decision_model_id: Some(GEMINI_DECISION_1_MODEL_ID.to_string()),
            primary_provider: "opencode_go".to_string(),
            primary_text_model_id: "deepseek-v4.1-flash".to_string(),
            fast_path_confidence_threshold: DEFAULT_FAST_PATH_CONFIDENCE,
            max_questions_per_batch: 8,
            compiler: DecisionCompilerToml::default(),
        }
    }
}

/// `[decision_router.compiler]` settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DecisionCompilerToml {
    /// Fast-path confidence gate.
    pub fast_path_confidence: f32,
    /// Minimum cell probability included in centroid interpolation.
    pub min_cell_prob: f32,
    /// Strategy A minimum confidence.
    pub time_slot_min_confidence: f32,
    /// Strategy B `Noul` threshold.
    pub bernoulli_activation_threshold: f32,
    /// Strategy C minimum candidate probability.
    pub probe_min_prob: f32,
    /// Strategy A termination sentinel.
    pub stop_sentinel: String,
    /// Mandatory fallback token.
    pub fallback_token: String,
    /// Spatial grid dimensions.
    pub grid: GridToml,
}

impl Default for DecisionCompilerToml {
    fn default() -> Self {
        Self {
            fast_path_confidence: DEFAULT_FAST_PATH_CONFIDENCE,
            min_cell_prob: 0.05,
            time_slot_min_confidence: 0.75,
            bernoulli_activation_threshold: 0.80,
            probe_min_prob: 0.15,
            stop_sentinel: "stop_done".to_string(),
            fallback_token: MANDATORY_CHOICE_FALLBACK.to_string(),
            grid: GridToml::default(),
        }
    }
}

/// `[decision_router.compiler.grid]` dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct GridToml {
    /// Horizontal column count (`1..=26`).
    pub cols: u8,
    /// Vertical row count (`1..=255`).
    pub rows: u8,
}

impl Default for GridToml {
    fn default() -> Self {
        Self { cols: 15, rows: 15 }
    }
}

/// `[live_session]` settings.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct LiveSessionToml {
    /// Live model ID.
    pub model: String,
    /// `"Audio"` or `"Text"`.
    pub response_modality: String,
    /// Audio transcription toggle.
    pub output_audio_transcription: bool,
    /// Proactive audio toggle.
    pub proactive_audio: bool,
    /// Affective dialog toggle.
    pub enable_affective_dialog: bool,
    /// Uplink sample rate in Hz.
    pub uplink_sample_rate_hz: u32,
    /// Downlink sample rate in Hz.
    pub downlink_sample_rate_hz: u32,
    /// Screen frame cap in FPS.
    pub max_screen_fps: u8,
    /// Compression trigger threshold in tokens.
    pub context_compression_trigger_tokens: u32,
    /// Sliding window target tokens.
    pub sliding_window_target_tokens: u32,
    /// Enable transparent session resumption.
    pub session_resumption_enabled: bool,
    /// Require epoch fences on tool calls.
    pub require_koma_epoch_fence: bool,
}

impl Default for LiveSessionToml {
    fn default() -> Self {
        Self {
            model: "gemini-2.5-flash-live".to_string(),
            response_modality: "Audio".to_string(),
            output_audio_transcription: true,
            proactive_audio: false,
            enable_affective_dialog: true,
            uplink_sample_rate_hz: UPLINK_AUDIO_SAMPLE_RATE_HZ,
            downlink_sample_rate_hz: DOWNLINK_AUDIO_SAMPLE_RATE_HZ,
            max_screen_fps: DEFAULT_SCREEN_FPS,
            context_compression_trigger_tokens: 24_000,
            sliding_window_target_tokens: 12_000,
            session_resumption_enabled: true,
            require_koma_epoch_fence: true,
        }
    }
}

/// `[jikubox]` configuration wrapper.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct JikuBoxTable {
    /// `[jikubox.triage]` settings.
    pub triage: JikuBoxTriageToml,
}

/// `[jikubox.triage]` settings.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct JikuBoxTriageToml {
    /// Preferred driver (`"decisions"`).
    pub driver: String,
    /// Pinned model (`"gemini-decision-1"`).
    pub model: String,
    /// Max questions per artifact batch (`<= 1000`).
    pub max_questions_per_batch: usize,
    /// Prefer decomposed binary `noul` checklists over coarse `score`.
    pub prefer_noul_checklists: bool,
}

impl Default for JikuBoxTriageToml {
    fn default() -> Self {
        Self {
            driver: "decisions".to_string(),
            model: GEMINI_DECISION_1_MODEL_ID.to_string(),
            max_questions_per_batch: 1000,
            prefer_noul_checklists: true,
        }
    }
}

/// `[memory]` configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct MemoryToml {
    /// Storage backend (`"sqlite_fts5"`).
    pub backend: String,
    /// SQLite database path.
    pub db_path: String,
    /// L0 perceptual-hash frame ring size.
    pub l0_frame_ring_size: usize,
    /// Max length of L1 per-app quirk notes.
    pub app_quirk_max_chars: usize,
    /// Whether to stage synthesized candidate skills.
    pub stage_new_skills: bool,
}

impl Default for MemoryToml {
    fn default() -> Self {
        Self {
            backend: "sqlite_fts5".to_string(),
            db_path: "~/.local/state/jiku/state.db".to_string(),
            l0_frame_ring_size: DEFAULT_L0_FRAME_RING_SIZE,
            app_quirk_max_chars: DEFAULT_APP_QUIRK_MAX_CHARS,
            stage_new_skills: true,
        }
    }
}

/// `[notify]` configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct NotifyToml {
    /// Enable desktop D-Bus/`notify-send` alerts.
    pub desktop_dbus: bool,
    /// Application display name.
    pub app_name: String,
    /// Events triggering desktop notifications.
    pub notify_on: Vec<String>,
}

impl Default for NotifyToml {
    fn default() -> Self {
        Self {
            desktop_dbus: true,
            app_name: "Jiku".to_string(),
            notify_on: vec![
                "task_completed".to_string(),
                "task_failed".to_string(),
                "verification_needed".to_string(),
                "skill_staged".to_string(),
            ],
        }
    }
}

/// `[tts]` optional speech synthesis configuration (`enabled = false` by default).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct TtsToml {
    /// Master enable switch (false by default).
    pub enabled: bool,
    /// Provider kind (`"voicevox"`, `"openai_speech"`, `"espeak_ng"`, `"irodori"`, `"noop"`).
    pub provider: String,
    /// Local or remote endpoint URL.
    pub endpoint: String,
    /// VOICEVOX speaker index.
    pub speaker_id: u32,
    /// Events triggering speech synthesis when enabled.
    pub speak_on: Vec<String>,
    /// Preferred audio playback sink (`""` / `"auto"`, `"pw-play"`, `"aplay"`, `"silent"`, or command).
    #[serde(default)]
    pub audio_sink: String,
    /// Provider-specific voice identifier (e.g. `espeak-ng` voice `"en"`, `"ja"`, or Irodori voice).
    #[serde(default)]
    pub voice: String,
}

/// Alias preserving external references to `TtsConfig`.
pub type TtsConfig = TtsToml;

impl Default for TtsToml {
    fn default() -> Self {
        Self {
            enabled: false,
            provider: "voicevox".to_string(),
            endpoint: "http://127.0.0.1:50021".to_string(),
            speaker_id: 3,
            speak_on: vec![
                "task_completed".to_string(),
                "verification_needed".to_string(),
            ],
            audio_sink: String::new(),
            voice: String::new(),
        }
    }
}

/// `[avatar]` companion EventBus cue configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct AvatarToml {
    /// Emit `AvatarCue` JSON packets on daemon status/subscribers.
    pub emit_cues: bool,
    /// Track active virtual pointer coordinates in gaze `(u, v)`.
    pub idle_gaze_follow_vseat: bool,
}

impl Default for AvatarToml {
    fn default() -> Self {
        Self {
            emit_cues: true,
            idle_gaze_follow_vseat: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_repository_example_config() {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let example_path = manifest_dir.join("../../examples/config.toml");
        let cfg =
            JikuConfig::load_from_path(&example_path).expect("example config.toml must validate");

        let planner = cfg.roles.get("planner").expect("planner role");
        assert_eq!(planner.provider, "opencode_go");
        let planner_provider = cfg
            .providers
            .get(&planner.provider)
            .expect("planner provider");
        assert_eq!(planner_provider.kind, ProviderKind::OpenCodeGo);
        assert_eq!(planner_provider.api_key_env, "JIKU_PLANNER_API_KEY");
        assert!(!cfg.resolve_thinking_enabled(Some(planner), planner_provider));

        let decision = cfg
            .roles
            .get("decision_router")
            .expect("decision_router role");
        let decision_provider = cfg
            .providers
            .get(&decision.provider)
            .expect("decision provider");
        assert_eq!(decision_provider.kind, ProviderKind::GoogleGemini);
        assert_eq!(decision_provider.api_key_env, "JIKU_DECISION_API_KEY");
        assert!(!cfg.tts.enabled);
    }

    #[test]
    fn default_config_uses_deepseek_v41_flash_with_thinking_disabled() {
        let cfg = JikuConfig::default();
        cfg.validate().unwrap();
        let (prov_id, model_id, role) = cfg.planner_binding();
        assert_eq!(prov_id, "opencode_go");
        assert_eq!(model_id, "deepseek-v4.1-flash");
        let prov = cfg.providers.get(&prov_id).unwrap();
        assert!(!cfg.resolve_thinking_enabled(role, prov));
    }
}
