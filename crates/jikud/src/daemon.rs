//! Core `jikud` desktop orchestration engine (`JikuEngine`) and `$XDG_RUNTIME_DIR/jiku.sock`
//! asynchronous Unix domain socket RPC server.

use std::{
    fs::{self, OpenOptions},
    io::Write as _,
    os::unix::fs::PermissionsExt as _,
    path::{Path, PathBuf},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context as _, anyhow, bail};
use jiku_memory::{
    ContextProvider as _, EpisodeRecord, L0FrameEntry, L0FrameRing, NewEpisode, SqliteMemoryStore,
    StagedSkillRecord, expand_tilde_path,
};
use jiku_tts::{
    AvatarCue, AvatarEmotion, DesktopNotifier, NoopTtsProvider, OpenAiSpeechProvider, TtsProvider,
    VoicevoxProvider,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::{
    io::{AsyncReadExt as _, AsyncWriteExt as _},
    net::{UnixListener, UnixStream},
    signal::unix::{SignalKind, signal},
};

use crate::{
    ControlPlaneRoute, DecisionBatchSpec, DecisionMediaSource, DecisionQuestionKind,
    GEMINI_DECISION_1_MODEL_ID, GridSpec, MANDATORY_CHOICE_FALLBACK,
    config::JikuConfig,
    drivers::{
        DriverMediaInput,
        decisions::{DecisionAnswer, DecisionBatchResponse, DecisionExecutor},
        turn::{
            TurnMessage, TurnModelClient, TurnRequest, TurnResponse, strip_markdown_json_fences,
        },
    },
    jikubox::{JikuBoxReport, classify_dropbox_dir},
    koma_ipc::{
        ERR_CODE_HUMAN_PREEMPTED, ERR_CODE_PROTECTED_SURFACE, ERR_CODE_STALE_EPOCH,
        KomaCapturedFrame, KomaIpcClient, KomaIpcError, KomaWorkspaceSummary,
    },
    lua_policy::JikuLuaPolicy,
};

const MAX_DAEMON_FRAME_BYTES: usize = 16 * 1024 * 1024;
const MAX_TASK_ACTIONS: usize = 48;

/// Serializable decision question DTO accepted over `jiku.sock` and CLI/MCP surfaces.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RpcDecisionQuestion {
    /// Binary calibrated probability question (`Yes`/`No`).
    Noul {
        /// Unique question identifier within the batch.
        id: String,
        /// Natural-language yes/no criteria.
        instructions: String,
    },
    /// Mutually exclusive categorical choice question.
    Choice {
        /// Unique question identifier within the batch.
        id: String,
        /// Natural-language selection instruction.
        instructions: String,
        /// Candidate labels (automatically padded with `NONE_OR_OTHER` if missing).
        choices: Vec<String>,
        /// Optional linked companion `Noul` scope-gate question ID.
        #[serde(default)]
        scope_gate_question_id: Option<String>,
    },
}

/// Wire request envelope accepted by the `jikud` Unix domain socket server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonRpcRequest {
    /// Monotonic caller request identifier.
    pub id: u64,
    /// Method name (`status`, `decide`, `plan`, `run_task`, `box.classify`,
    /// `memory.search`, `skills.list`, `skills.diff`, `skills.stage`, `skills.approve`, `shutdown`).
    pub method: String,
    /// Method parameters object.
    #[serde(default)]
    pub params: Value,
}

/// Structured error envelope returned by the `jikud` daemon.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonRpcError {
    /// Machine-readable error code (`400`, `404`, `500`).
    pub code: u16,
    /// Diagnostic explanation (never contains raw credentials).
    pub message: String,
}

/// Wire response envelope returned by the `jikud` Unix domain socket server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonRpcResponse {
    /// Matching request identifier.
    pub id: u64,
    /// Result JSON payload on success.
    #[serde(default)]
    pub result: Option<Value>,
    /// Error object on failure.
    #[serde(default)]
    pub error: Option<DaemonRpcError>,
}

/// Complete health and wiring summary returned by `jiku status` / `jikud` status RPC.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonStatusReport {
    /// Package version.
    pub version: String,
    /// Resolved `config.toml` path, if loaded from disk (`None` = built-in defaults).
    pub config_path: Option<String>,
    /// Resolved `init.lua` path, if loaded from disk (`None` = empty sandbox).
    pub policy_path: Option<String>,
    /// Configured daemon socket path.
    pub daemon_socket_path: String,
    /// Configured planner provider key.
    pub planner_provider: String,
    /// Configured planner model ID.
    pub planner_model: String,
    /// Resolved thinking mode flag for planner calls (`false` on OpenCode Go by default).
    pub planner_thinking_enabled: bool,
    /// Resolved decision router execution mode summary.
    pub decision_mode: String,
    /// Decision model ID if configured.
    pub decision_model_id: Option<String>,
    /// Registered Phodopus Lua `(prefetch_hooks, event_hooks)` counts.
    pub lua_hook_counts: (usize, usize),
    /// Resolved SQLite WAL + FTS5 path (or `":memory:"`).
    pub memory_db_path: String,
    /// Recent episodes persisted in L2 memory.
    pub episode_count: usize,
    /// Total staged/approved/rejected skill records.
    pub skill_count: usize,
    /// L0 perceptual-hash frame ring occupancy.
    pub l0_ring_len: usize,
    /// Whether TTS synthesis is enabled.
    pub tts_enabled: bool,
    /// Active TTS provider identifier (`"noop"`, `"voicevox"`, `"openai_speech"`).
    pub tts_provider: String,
    /// Last emitted companion EventBus cue, if any.
    pub last_avatar_cue: Option<AvatarCue>,
    /// Whether `koma-agent.sock` was reachable and authenticated.
    pub koma_connected: bool,
    /// Connected Koma runtime instance directory, if reachable.
    pub koma_instance_dir: Option<String>,
    /// Active Koma workspace inventory snapshots when reachable.
    pub koma_workspaces: Vec<KomaWorkspaceSummary>,
    /// Diagnostic note if Koma was unreachable.
    pub koma_error: Option<String>,
}

/// Concrete Koma step action executed during [`JikuEngine::run_task`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum PlannedKomaStep {
    /// Click centered inside a 15x15 `grid_225` cell.
    ClickCell {
        /// Cell label (`"A1"`..=`"O15"`).
        cell: String,
        /// Mouse button (`"left"`, `"right"`, `"middle"`).
        #[serde(default = "default_mouse_button")]
        button: String,
        /// Click multiplicity (`1` = single, `2` = double).
        #[serde(default = "default_clicks")]
        clicks: u8,
    },
    /// Click explicit logical workspace coordinates `(x, y)`.
    ClickXy {
        /// Horizontal pixel coordinate.
        x: f64,
        /// Vertical pixel coordinate.
        y: f64,
        /// Mouse button.
        #[serde(default = "default_mouse_button")]
        button: String,
        /// Click multiplicity.
        #[serde(default = "default_clicks")]
        clicks: u8,
    },
    /// Drag the pointer linearly or along a kinematic profile between two local `(x, y)` coordinates.
    DragXy {
        /// Stroke starting horizontal coordinate.
        from_x: f64,
        /// Stroke starting vertical coordinate.
        from_y: f64,
        /// Stroke ending horizontal coordinate.
        to_x: f64,
        /// Stroke ending vertical coordinate.
        to_y: f64,
        /// Mouse button (`"left"`, `"right"`, `"middle"`).
        #[serde(default = "default_mouse_button")]
        button: String,
        /// Trajectory profile (`"spring"`, `"minimum_jerk"`, `"bezier"`, `"instant"`).
        #[serde(default = "default_drag_profile")]
        profile: String,
        /// Requested trajectory duration in milliseconds.
        #[serde(default = "default_drag_duration_ms")]
        duration_ms: u64,
    },
    /// Drag the pointer between two 15x15 `grid_225` cells.
    DragCell {
        /// Starting cell label (`"A1"`..=`"O15"`).
        from_cell: String,
        /// Ending cell label (`"A1"`..=`"O15"`).
        to_cell: String,
        /// Mouse button (`"left"`, `"right"`, `"middle"`).
        #[serde(default = "default_mouse_button")]
        button: String,
        /// Trajectory profile (`"spring"`, `"minimum_jerk"`, `"bezier"`, `"instant"`).
        #[serde(default = "default_drag_profile")]
        profile: String,
        /// Requested trajectory duration in milliseconds.
        #[serde(default = "default_drag_duration_ms")]
        duration_ms: u64,
    },
    /// Inject UTF-8 text into the focused Wayland client via `vseat.inject_text`.
    InjectText {
        /// Text string to commit via `zwp_text_input_v3` / clipboard fallback.
        text: String,
    },
    /// Tap an xkbcommon key chord via `vseat.key` (e.g. `"Return"`, `"Ctrl+l"`).
    Key {
        /// Chord descriptor understood by Koma.
        chord: String,
    },
    /// Wait `ms` milliseconds for animations or client repaints to settle.
    WaitMs {
        /// Delay duration in milliseconds (clamped at execution time to 5000 ms).
        ms: u64,
    },
}

fn default_mouse_button() -> String {
    "left".to_string()
}

const fn default_clicks() -> u8 {
    1
}

fn default_drag_profile() -> String {
    "spring".to_string()
}

const fn default_drag_duration_ms() -> u64 {
    120
}

fn normalize_kinematic_profile(raw: &str) -> &'static str {
    match raw.trim().to_ascii_lowercase().as_str() {
        "instant" => "instant",
        "minimum_jerk" | "minimum-jerk" | "min_jerk" => "minimum_jerk",
        "bezier" => "bezier",
        _ => "spring",
    }
}

/// Execution outcome of a single planned step in [`RunTaskReport`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutedStepReport {
    /// Planned step executed (or previewed in `--dry-run`).
    pub step: PlannedKomaStep,
    /// Execution status (`"ok"`, `"dry_run"`, `"halted_stale_epoch"`, etc.).
    pub status: String,
    /// Workspace epoch observed after this step, if Koma was connected.
    pub epoch_after: Option<u64>,
}

/// Complete report returned by `jiku run` / `run_task`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunTaskReport {
    /// Target workspace identifier.
    pub workspace: String,
    /// User goal prompt.
    pub goal: String,
    /// Route selected by `DecisionCompilerConfig::route`.
    pub route: String,
    /// Whether `--dry-run` skipped Koma `vseat.*` mutations.
    pub dry_run: bool,
    /// Whether Koma IPC capture/action was connected.
    pub koma_connected: bool,
    /// Workspace epoch observed before execution, if any.
    pub initial_epoch: Option<u64>,
    /// Final workspace epoch after execution, if any.
    pub final_epoch: Option<u64>,
    /// Frame dimensions `(width, height)` when Koma frame capture ran.
    pub captured_dimensions: Option<(u32, u32)>,
    /// Perceptual 64-bit `dHash` hex string of captured workspace frame, if any.
    pub frame_dhash_hex: Option<String>,
    /// Whether Koma's Ring 0 protected-surface blackout obscured sensitive areas.
    pub has_protected_blackout: bool,
    /// Decision gate outputs (`execution_mode`, calibrated answers, token usage).
    pub decision: Option<DecisionBatchResponse>,
    /// Autoregressive planner summary when `TurnModelClient` ran.
    pub planner_summary: String,
    /// Executed (or dry-run) step log.
    pub executed_steps: Vec<ExecutedStepReport>,
    /// Prefetch notes injected by L1/L2 memory + Phodopus `init.lua`.
    pub injected_notes: Vec<String>,
    /// Persisted L2 SQLite episode row ID.
    pub episode_id: i64,
    /// Wall-clock duration in milliseconds.
    pub duration_ms: u64,
    /// Optional non-fatal stop/verification reason.
    pub halt_reason: Option<String>,
    /// Candidate skill ID staged during this episode, if any.
    pub staged_skill_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct PlannerPlanPayload {
    #[serde(default)]
    summary: String,
    #[serde(default)]
    steps: Vec<PlannedKomaStep>,
    #[serde(default)]
    candidate_skill: Option<CandidateSkillPayload>,
}

#[derive(Debug, Deserialize)]
struct CandidateSkillPayload {
    id: String,
    title: String,
    body: String,
}

/// Stateful desktop orchestration engine unifying configuration, Koma IPC, Three-Driver
/// inference (`gemini-decision-1` + OpenCode Go `deepseek-v4.1-flash`), Phodopus Lua
/// hooks, SQLite WAL+FTS5 memory, notifications, and optional TTS cues.
pub struct JikuEngine {
    config: JikuConfig,
    config_path: Option<PathBuf>,
    policy_path: Option<PathBuf>,
    lua_policy: JikuLuaPolicy,
    memory: SqliteMemoryStore,
    l0_ring: L0FrameRing,
    decision_executor: DecisionExecutor,
    turn_client: TurnModelClient,
    notifier: DesktopNotifier,
    tts: Box<dyn TtsProvider>,
    last_avatar_cue: Option<AvatarCue>,
}

impl JikuEngine {
    /// Initializes [`JikuEngine`] from resolved `--config` / `$JIKU_CONFIG` / XDG config
    /// and `--policy` / `$JIKU_POLICY` / XDG `init.lua`.
    pub fn from_resolved(
        config_path: Option<&Path>,
        policy_path: Option<&Path>,
        in_memory_db: bool,
    ) -> anyhow::Result<Self> {
        let (config, resolved_cfg) =
            JikuConfig::load_resolved(config_path).context("failed loading Jiku config")?;
        let (lua_policy, resolved_pol) =
            JikuLuaPolicy::load_resolved(policy_path).context("failed loading init.lua policy")?;

        let memory = if in_memory_db {
            SqliteMemoryStore::open_in_memory(config.memory.app_quirk_max_chars)?
        } else {
            SqliteMemoryStore::open(&config.memory_db_path(), config.memory.app_quirk_max_chars)?
        };

        let l0_ring = L0FrameRing::new(config.memory.l0_frame_ring_size);
        let decision_executor = DecisionExecutor::new(Duration::from_secs(45))?;
        let turn_client = TurnModelClient::new(Duration::from_secs(90))?;
        let notifier = DesktopNotifier {
            enabled: config.notify.desktop_dbus,
            app_name: config.notify.app_name.clone(),
        };
        let tts = build_tts_provider(&config);

        Ok(Self {
            config,
            config_path: resolved_cfg,
            policy_path: resolved_pol,
            lua_policy,
            memory,
            l0_ring,
            decision_executor,
            turn_client,
            notifier,
            tts,
            last_avatar_cue: None,
        })
    }

    /// Returns a read-only reference to the validated configuration.
    #[must_use]
    pub const fn config(&self) -> &JikuConfig {
        &self.config
    }

    /// Collects runtime health, model routing, memory metrics, and live Koma connectivity.
    pub fn status(&self) -> anyhow::Result<DaemonStatusReport> {
        let (planner_provider, planner_model, planner_role) = self.config.planner_binding();
        let planner_thinking_enabled = self
            .config
            .providers
            .get(&planner_provider)
            .is_some_and(|p| self.config.resolve_thinking_enabled(planner_role, p));
        let policy = self.config.to_decision_router_policy();
        let decision_mode = match policy.mode {
            crate::DecisionExecutionMode::NativeDecisionsApi { .. } => "native_decisions_api",
            crate::DecisionExecutionMode::EmulatedStructuredTextModel { .. } => {
                "emulated_structured_text_model"
            }
        }
        .to_string();

        let episode_count = self.memory.recent_episodes(500)?.len();
        let skill_count = self.memory.list_skills()?.len();
        let db_display = self
            .memory
            .db_path()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| ":memory:".to_string());

        let (koma_connected, koma_instance_dir, koma_workspaces, koma_error) =
            match KomaIpcClient::connect_discovered(&self.config.koma.default_workspace, true) {
                Ok(mut ipc) => {
                    let dir = ipc.instance_dir().display().to_string();
                    match ipc.inventory() {
                        Ok(ws) => (true, Some(dir), ws, None),
                        Err(e) => (true, Some(dir), Vec::new(), Some(e.to_string())),
                    }
                }
                Err(e) => (false, None, Vec::new(), Some(e.to_string())),
            };

        Ok(DaemonStatusReport {
            version: env!("CARGO_PKG_VERSION").to_string(),
            config_path: self.config_path.as_ref().map(|p| p.display().to_string()),
            policy_path: self.policy_path.as_ref().map(|p| p.display().to_string()),
            daemon_socket_path: self.config.daemon_socket_path().display().to_string(),
            planner_provider,
            planner_model,
            planner_thinking_enabled,
            decision_mode,
            decision_model_id: self.config.decision_router.decision_model_id.clone(),
            lua_hook_counts: self.lua_policy.hook_counts(),
            memory_db_path: db_display,
            episode_count,
            skill_count,
            l0_ring_len: self.l0_ring.entries().len(),
            tts_enabled: self.config.tts.enabled,
            tts_provider: self.tts.provider_name().to_string(),
            last_avatar_cue: self.last_avatar_cue.clone(),
            koma_connected,
            koma_instance_dir,
            koma_workspaces,
            koma_error,
        })
    }

    /// Evaluates a calibrated batch of `Noul` / `Choice` questions with optional image input.
    pub async fn decide(
        &self,
        image_path: Option<&Path>,
        prompt: &str,
        raw_questions: &[RpcDecisionQuestion],
        force_emulate: bool,
    ) -> anyhow::Result<DecisionBatchResponse> {
        let (media_source, media_input) = load_optional_media(image_path)?;
        let questions = build_decision_questions(raw_questions, prompt)?;
        let spec = DecisionBatchSpec {
            model: self
                .config
                .decision_router
                .decision_model_id
                .clone()
                .unwrap_or_else(|| GEMINI_DECISION_1_MODEL_ID.to_string()),
            media_source,
            questions,
        };
        let policy = self.config.to_decision_router_policy();
        let fallback_to_primary = self
            .config
            .roles
            .get("decision_router")
            .and_then(|r| r.fallback_to_primary_text_model)
            .unwrap_or(true);

        let resp = self
            .decision_executor
            .evaluate(
                &self.config,
                &policy,
                &spec,
                prompt,
                media_input.as_ref(),
                force_emulate,
                fallback_to_primary,
            )
            .await?;
        Ok(resp)
    }

    /// Invokes the configured `TurnModelDriver` (`deepseek-v4.1-flash` with thinking disabled
    /// by default on OpenCode Go) with optional multimodal image attachment.
    pub async fn plan(
        &self,
        image_path: Option<&Path>,
        system: Option<&str>,
        prompt: &str,
        json_object: bool,
        stream: bool,
    ) -> anyhow::Result<TurnResponse> {
        if prompt.trim().is_empty() {
            bail!("plan prompt must not be empty");
        }
        let (_, media_input) = load_optional_media(image_path)?;
        let (provider_id, model, role) = self.config.planner_binding();
        let provider = self
            .config
            .providers
            .get(&provider_id)
            .with_context(|| format!("missing planner provider `{provider_id}`"))?;
        let thinking_enabled = self.config.resolve_thinking_enabled(role, provider);
        let temperature = role.and_then(|r| r.temperature).or(Some(0.2));

        let mut media = Vec::new();
        if let Some(m) = media_input {
            media.push(m);
        }

        let system_prompt = system
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or("You are Jiku, a concise and accurate desktop orchestration assistant.")
            .to_string();

        let resp = self
            .turn_client
            .generate(
                &self.config,
                &TurnRequest {
                    provider_id,
                    model,
                    system_prompt,
                    messages: vec![TurnMessage {
                        role: "user".to_string(),
                        text: prompt.to_string(),
                        media,
                    }],
                    temperature,
                    max_tokens: Some(1536),
                    json_object,
                    stream,
                    thinking_enabled,
                },
            )
            .await?;
        Ok(resp)
    }

    /// Executes an end-to-end Observe -> Memory/Lua Prefetch -> Decision Gate -> Planner ->
    /// `Epoch`-fenced Koma action -> L2 Persist episode.
    pub async fn run_task(
        &mut self,
        workspace_override: Option<&str>,
        goal: &str,
        dry_run: bool,
        force_emulate: bool,
    ) -> anyhow::Result<RunTaskReport> {
        if goal.trim().is_empty() {
            bail!("run goal must not be empty");
        }

        let started = Instant::now();
        let requested_ws = workspace_override
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or(&self.config.koma.default_workspace)
            .to_string();

        let mut ipc_opt: Option<KomaIpcClient> = None;
        let mut target_ws = requested_ws.clone();
        let mut active_app_id = String::new();
        let mut active_title = String::new();
        let mut active_win_id = String::new();
        let mut captured: Option<KomaCapturedFrame> = None;
        let mut inventory: Vec<KomaWorkspaceSummary> = Vec::new();

        match KomaIpcClient::connect_discovered(&requested_ws, true) {
            Ok(mut ipc) => {
                inventory = ipc.inventory()?;
                if let Some(ws) = inventory
                    .iter()
                    .find(|w| w.ws_id == requested_ws)
                    .or_else(|| inventory.first())
                {
                    target_ws = ws.ws_id.clone();
                    if let Some(win) = &ws.active_window {
                        active_app_id = win.app_id.clone();
                        active_title = win.title.clone();
                        active_win_id = win.win_id.clone();
                    }
                } else {
                    bail!("Koma reported zero active workspaces");
                }

                let frame =
                    ipc.capture_frame(&target_ws, &self.config.koma.default_overlay, 1280)?;
                self.l0_ring.push(L0FrameEntry {
                    epoch: frame.epoch,
                    workspace: target_ws.clone(),
                    width: frame.width,
                    height: frame.height,
                    dhash: frame.dhash64,
                    timestamp_ms: now_unix_ms(),
                });
                captured = Some(frame);
                ipc_opt = Some(ipc);
            }
            Err(err) => {
                if !dry_run {
                    return Err(anyhow!(
                        "cannot run live task without active Koma compositor IPC (pass --dry-run to preview without Koma): {err}"
                    ));
                }
            }
        }

        let mut injected_notes = Vec::new();
        if !active_app_id.is_empty()
            && let Some(quirk) = self.memory.get_app_quirk(&active_app_id)?
        {
            injected_notes.push(format!("[L1 quirk:{active_app_id}] {}", quirk.notes));
        }
        let prior_episodes = self.memory.search_episodes(goal, 3)?;
        for ep in &prior_episodes {
            injected_notes.push(format!(
                "[L2 prior #{} ({})] {}",
                ep.id, ep.route, ep.summary
            ));
        }
        let lua_notes = self
            .lua_policy
            .run_prefetch(&active_app_id, &target_ws, goal)?;
        for note in lua_notes {
            injected_notes.push(format!("[lua prefetch] {note}"));
        }

        let initial_epoch = captured.as_ref().map(|f| f.epoch);
        let has_protected_blackout = captured.as_ref().is_some_and(|f| f.has_protected_blackout);
        let captured_dimensions = captured.as_ref().map(|f| (f.width, f.height));
        let frame_dhash_hex = captured.as_ref().map(|f| format!("{:016x}", f.dhash64));

        let context_summary = build_runtime_context_summary(
            &target_ws,
            goal,
            &inventory,
            &active_app_id,
            &active_title,
            &active_win_id,
            initial_epoch,
            has_protected_blackout,
            &injected_notes,
        );

        let gate_questions = vec![
            (
                "q_in_scope".to_string(),
                DecisionQuestionKind::Noul {
                    instructions: "Is the requested desktop or terminal task in scope and clearly actionable from the current workspace context or screenshot?".to_string(),
                },
            ),
            (
                "q_safety_risk".to_string(),
                DecisionQuestionKind::Noul {
                    instructions: "Only answer Yes if the goal deletes system/user files recursively, exfiltrates passwords/API keys, executes financial transfers, or asks to bypass a lock/security prompt. Benign terminal commands, test output, drawing/painting strokes, or UI clicks MUST be answered No.".to_string(),
                },
            ),
            (
                "q_action_kind".to_string(),
                DecisionQuestionKind::Choice {
                    instructions: "Select the best control strategy for executing this goal.".to_string(),
                    choices: vec![
                        "type_or_key".to_string(),
                        "click_target".to_string(),
                        "drag_or_draw".to_string(),
                        "multi_step_plan".to_string(),
                        MANDATORY_CHOICE_FALLBACK.to_string(),
                    ],
                    scope_gate_question_id: "q_in_scope".to_string(),
                },
            ),
        ];

        let (media_source, media_input) = if let Some(frame) = &captured {
            (
                DecisionMediaSource::InlineBytes {
                    mime_type: "image/png".to_string(),
                },
                Some(DriverMediaInput::Inline {
                    mime_type: "image/png".to_string(),
                    bytes: frame.png_bytes.clone(),
                }),
            )
        } else {
            (
                DecisionMediaSource::InlineBytes {
                    mime_type: "image/png".to_string(),
                },
                None,
            )
        };

        let decision_spec = DecisionBatchSpec {
            model: self
                .config
                .decision_router
                .decision_model_id
                .clone()
                .unwrap_or_else(|| GEMINI_DECISION_1_MODEL_ID.to_string()),
            media_source,
            questions: gate_questions,
        };
        let router_policy = self.config.to_decision_router_policy();
        let fallback_to_primary = self
            .config
            .roles
            .get("decision_router")
            .and_then(|r| r.fallback_to_primary_text_model)
            .unwrap_or(true);

        let decision_resp = self
            .decision_executor
            .evaluate(
                &self.config,
                &router_policy,
                &decision_spec,
                &context_summary,
                media_input.as_ref(),
                force_emulate,
                fallback_to_primary,
            )
            .await?;

        let in_scope_prob = decision_resp
            .answers
            .get("q_in_scope")
            .and_then(DecisionAnswer::noul_prob)
            .unwrap_or(0.8);
        let safety_prob = decision_resp
            .answers
            .get("q_safety_risk")
            .and_then(DecisionAnswer::noul_prob)
            .unwrap_or(0.0);
        let (top_choice, top_choice_prob) = decision_resp
            .answers
            .get("q_action_kind")
            .and_then(DecisionAnswer::top_choice)
            .map(|(c, p)| (c.to_string(), p))
            .unwrap_or_else(|| (MANDATORY_CHOICE_FALLBACK.to_string(), 0.5));

        let safety_blocked = safety_prob >= 0.65 || (has_protected_blackout && safety_prob >= 0.45);
        let route = router_policy.compiler.route(
            in_scope_prob,
            &top_choice,
            top_choice_prob,
            safety_blocked,
        );

        let route_label = match route {
            ControlPlaneRoute::FastPathKoma { .. } => "fast_path_koma",
            ControlPlaneRoute::FallbackTurnPlanner { .. } => "turn_planner",
            ControlPlaneRoute::RequireHumanVerification { .. } => "human_verification",
        };

        if let ControlPlaneRoute::RequireHumanVerification { reason } = route {
            let summary = format!("Halted for human verification ({reason})");
            let ep_id = self.memory.record_episode(&NewEpisode {
                workspace: target_ws.clone(),
                goal: goal.to_string(),
                route: route_label.to_string(),
                summary: summary.clone(),
                actions_json: "[]".to_string(),
                confidence: safety_prob,
                duration_ms: started.elapsed().as_millis() as u64,
            })?;
            self.dispatch_lifecycle_event(
                "security.verification_needed",
                json!({
                    "workspace": target_ws,
                    "goal": goal,
                    "reason": reason,
                    "safety_prob": safety_prob,
                    "has_protected_blackout": has_protected_blackout,
                }),
                AvatarEmotion::WaitingHuman,
                Some("Jiku verification required"),
            )
            .await?;

            return Ok(RunTaskReport {
                workspace: target_ws,
                goal: goal.to_string(),
                route: route_label.to_string(),
                dry_run,
                koma_connected: ipc_opt.is_some(),
                initial_epoch,
                final_epoch: initial_epoch,
                captured_dimensions,
                frame_dhash_hex,
                has_protected_blackout,
                decision: Some(decision_resp),
                planner_summary: summary,
                executed_steps: Vec::new(),
                injected_notes,
                episode_id: ep_id,
                duration_ms: started.elapsed().as_millis() as u64,
                halt_reason: Some(reason.to_string()),
                staged_skill_id: None,
            });
        }

        let planner_system = build_planner_system_prompt();
        let planner_user = format!(
            "{context_summary}\n\nDecision gate output:\n- route_label: {route_label}\n- q_in_scope: {in_scope_prob:.3}\n- q_safety_risk: {safety_prob:.3}\n- q_action_kind: {top_choice} ({top_choice_prob:.3})\n\nProduce the strict JSON plan now."
        );
        let (provider_id, model, role) = self.config.planner_binding();
        let provider = self
            .config
            .providers
            .get(&provider_id)
            .with_context(|| format!("missing planner provider `{provider_id}`"))?;
        let thinking_enabled = self.config.resolve_thinking_enabled(role, provider);

        let mut media_vec = Vec::new();
        if let Some(m) = media_input {
            media_vec.push(m);
        }

        let planner_resp = self
            .turn_client
            .generate(
                &self.config,
                &TurnRequest {
                    provider_id,
                    model,
                    system_prompt: planner_system,
                    messages: vec![TurnMessage {
                        role: "user".to_string(),
                        text: planner_user,
                        media: media_vec,
                    }],
                    temperature: Some(0.1),
                    max_tokens: Some(4096),
                    json_object: true,
                    stream: false,
                    thinking_enabled,
                },
            )
            .await?;

        let plan_json = strip_markdown_json_fences(&planner_resp.text);
        let parsed_plan: PlannerPlanPayload =
            serde_json::from_str(plan_json).with_context(|| {
                format!(
                    "planner did not return valid JSON action plan: {}",
                    planner_resp.text
                )
            })?;

        let capped_steps: Vec<PlannedKomaStep> = parsed_plan
            .steps
            .into_iter()
            .take(MAX_TASK_ACTIONS)
            .collect();
        let mut executed_steps = Vec::with_capacity(capped_steps.len());
        let mut current_epoch = initial_epoch.unwrap_or(1);
        let mut halt_reason: Option<String> = None;

        if dry_run {
            for step in capped_steps {
                executed_steps.push(ExecutedStepReport {
                    step,
                    status: "dry_run".to_string(),
                    epoch_after: initial_epoch,
                });
            }
        } else if let Some(ipc) = ipc_opt.as_mut() {
            for step in capped_steps {
                let outcome = execute_step_with_epoch_retry(
                    ipc,
                    &target_ws,
                    &mut current_epoch,
                    &step,
                    &self.config.koma.default_overlay,
                );
                match outcome {
                    Ok(settle_delay) => {
                        if !settle_delay.is_zero() {
                            tokio::time::sleep(settle_delay).await;
                        }
                        executed_steps.push(ExecutedStepReport {
                            step,
                            status: "ok".to_string(),
                            epoch_after: Some(current_epoch),
                        });
                    }
                    Err(StepHalt::HumanPreempted) => {
                        halt_reason = Some("human_preempted".to_string());
                        executed_steps.push(ExecutedStepReport {
                            step,
                            status: "halted_human_preempted".to_string(),
                            epoch_after: Some(current_epoch),
                        });
                        break;
                    }
                    Err(StepHalt::ProtectedSurface) => {
                        halt_reason = Some("protected_surface".to_string());
                        executed_steps.push(ExecutedStepReport {
                            step,
                            status: "halted_protected_surface".to_string(),
                            epoch_after: Some(current_epoch),
                        });
                        break;
                    }
                    Err(StepHalt::Fatal(e)) => return Err(e),
                }
            }
        }

        let mut staged_skill_id = None;
        if self.config.memory.stage_new_skills
            && let Some(cand) = parsed_plan.candidate_skill
            && !cand.id.trim().is_empty()
            && !cand.body.trim().is_empty()
            && let Ok(staged) = self
                .memory
                .stage_skill(&cand.id, &cand.title, &cand.body, "")
        {
            staged_skill_id = Some(staged.skill_id);
        }

        let duration_ms = started.elapsed().as_millis() as u64;
        let confidence = (in_scope_prob * top_choice_prob).clamp(0.0, 1.0);
        let summary = if parsed_plan.summary.trim().is_empty() {
            format!("Executed {} step(s) for goal", executed_steps.len())
        } else {
            parsed_plan.summary
        };
        let actions_json = serde_json::to_string(&executed_steps)?;

        let episode_id = self.memory.record_episode(&NewEpisode {
            workspace: target_ws.clone(),
            goal: goal.to_string(),
            route: route_label.to_string(),
            summary: summary.clone(),
            actions_json,
            confidence,
            duration_ms,
        })?;

        let emotion = if halt_reason.is_some() {
            AvatarEmotion::Alert
        } else {
            AvatarEmotion::Success
        };
        self.dispatch_lifecycle_event(
            "episode.completed",
            Self::episode_completed_payload(
                episode_id,
                &target_ws,
                goal,
                route_label,
                &summary,
                dry_run,
                halt_reason.as_deref(),
                executed_steps.len(),
                duration_ms,
            ),
            emotion,
            Some(&summary),
        )
        .await?;

        Ok(RunTaskReport {
            workspace: target_ws,
            goal: goal.to_string(),
            route: route_label.to_string(),
            dry_run,
            koma_connected: ipc_opt.is_some(),
            initial_epoch,
            final_epoch: ipc_opt.as_ref().map(|_| current_epoch),
            captured_dimensions,
            frame_dhash_hex,
            has_protected_blackout,
            decision: Some(decision_resp),
            planner_summary: summary,
            executed_steps,
            injected_notes,
            episode_id,
            duration_ms,
            halt_reason,
            staged_skill_id,
        })
    }

    /// Scans `dir` (defaulting to `config.daemon.dropbox_dir`) with `classify_dropbox_dir`.
    pub async fn box_classify(
        &self,
        dir: Option<&Path>,
        force_emulate: bool,
    ) -> anyhow::Result<JikuBoxReport> {
        let target = dir
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from(&self.config.daemon.dropbox_dir));
        let report = classify_dropbox_dir(
            &self.config,
            &self.decision_executor,
            &target,
            force_emulate,
        )
        .await?;
        Ok(report)
    }

    /// Searches persisted L2 episodes using SQLite FTS5 BM25 ranking.
    pub fn memory_search(&self, query: &str, limit: usize) -> anyhow::Result<Vec<EpisodeRecord>> {
        Ok(self.memory.search_episodes(query, limit.clamp(1, 100))?)
    }

    /// Lists staged, approved, and rejected skills.
    pub fn skills_list(&self) -> anyhow::Result<Vec<StagedSkillRecord>> {
        Ok(self.memory.list_skills()?)
    }

    /// Produces a unified diff string for a staged skill.
    pub fn skills_diff(&self, skill_id: &str) -> anyhow::Result<String> {
        Ok(self.memory.diff_skill(skill_id)?)
    }

    /// Stages a new sanitized skill candidate awaiting human approval.
    pub fn skills_stage(
        &self,
        skill_id: &str,
        title: &str,
        body: &str,
    ) -> anyhow::Result<StagedSkillRecord> {
        Ok(self.memory.stage_skill(skill_id, title, body, "")?)
    }

    /// Approves and promotes a staged skill candidate, optionally writing it into `promote_dir`.
    pub fn skills_approve(
        &self,
        skill_id: &str,
        promote_dir: Option<&Path>,
    ) -> anyhow::Result<(StagedSkillRecord, Option<PathBuf>)> {
        Ok(self.memory.approve_skill(skill_id, promote_dir)?)
    }

    async fn dispatch_lifecycle_event(
        &mut self,
        event_name: &str,
        payload: Value,
        default_emotion: AvatarEmotion,
        tts_fallback_text: Option<&str>,
    ) -> anyhow::Result<()> {
        let effects = self.lua_policy.emit_event(event_name, &payload)?;
        for note in &effects.notifications {
            let _ = self.notifier.dispatch(note);
        }

        let effective_emotion = effects
            .avatar_emotions
            .last()
            .copied()
            .unwrap_or(default_emotion);
        self.last_avatar_cue = Some(AvatarCue {
            emotion: effective_emotion,
            ..AvatarCue::default()
        });

        let mut utterances = effects.tts_utterances;
        if utterances.is_empty()
            && self.config.tts.enabled
            && let Some(text) = tts_fallback_text
        {
            utterances.push(text.to_string());
        }
        for line in utterances {
            if let Ok(synth) = self.tts.synthesize(&line, effective_emotion).await {
                self.last_avatar_cue = Some(synth.cue);
            }
        }

        if self.config.daemon.audit_log {
            let _ = append_audit_line(
                &self.config.audit_log_path(),
                &json!({
                    "ts_ms": now_unix_ms(),
                    "event": event_name,
                    "payload": payload,
                }),
            );
        }
        Ok(())
    }

    /// Builds the `episode.completed` Lua policy event payload.
    #[allow(clippy::too_many_arguments)]
    fn episode_completed_payload(
        episode_id: i64,
        workspace: &str,
        goal: &str,
        route: &str,
        summary: &str,
        dry_run: bool,
        halt_reason: Option<&str>,
        step_count: usize,
        duration_ms: u64,
    ) -> Value {
        json!({
            "episode_id": episode_id,
            "workspace": workspace,
            "goal": goal,
            "route": route,
            "summary": summary,
            "dry_run": dry_run,
            "halt_reason": halt_reason,
            "step_count": step_count,
            "duration_ms": duration_ms,
        })
    }

    async fn handle_rpc(&mut self, req: &DaemonRpcRequest) -> (DaemonRpcResponse, bool) {
        let mut should_shutdown = false;
        let outcome: anyhow::Result<Value> = match req.method.as_str() {
            "status" => self.status().and_then(|s| Ok(serde_json::to_value(s)?)),
            "decide" => {
                let image = req
                    .params
                    .get("image_path")
                    .and_then(Value::as_str)
                    .map(PathBuf::from);
                let prompt = req
                    .params
                    .get("prompt")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let force_emulate = req
                    .params
                    .get("force_emulate")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                let questions: Vec<RpcDecisionQuestion> = match req.params.get("questions").cloned()
                {
                    Some(raw) => match serde_json::from_value(raw) {
                        Ok(q) => q,
                        Err(e) => {
                            return (
                                DaemonRpcResponse {
                                    id: req.id,
                                    result: None,
                                    error: Some(DaemonRpcError {
                                        code: 400,
                                        message: format!("invalid `questions` array: {e}"),
                                    }),
                                },
                                false,
                            );
                        }
                    },
                    None => Vec::new(),
                };
                self.decide(image.as_deref(), prompt, &questions, force_emulate)
                    .await
                    .and_then(|r| Ok(serde_json::to_value(r)?))
            }
            "plan" => {
                let image = req
                    .params
                    .get("image_path")
                    .and_then(Value::as_str)
                    .map(PathBuf::from);
                let system = req.params.get("system").and_then(Value::as_str);
                let prompt = req
                    .params
                    .get("prompt")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let json_object = req
                    .params
                    .get("json_object")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                let stream = req
                    .params
                    .get("stream")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                self.plan(image.as_deref(), system, prompt, json_object, stream)
                    .await
                    .and_then(|r| Ok(serde_json::to_value(r)?))
            }
            "run_task" => {
                let ws = req.params.get("workspace").and_then(Value::as_str);
                let goal = req.params.get("goal").and_then(Value::as_str).unwrap_or("");
                let dry_run = req
                    .params
                    .get("dry_run")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                let force_emulate = req
                    .params
                    .get("force_emulate")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                self.run_task(ws, goal, dry_run, force_emulate)
                    .await
                    .and_then(|r| Ok(serde_json::to_value(r)?))
            }
            "box.classify" => {
                let dir = req
                    .params
                    .get("directory")
                    .and_then(Value::as_str)
                    .map(PathBuf::from);
                let force_emulate = req
                    .params
                    .get("force_emulate")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                self.box_classify(dir.as_deref(), force_emulate)
                    .await
                    .and_then(|r| Ok(serde_json::to_value(r)?))
            }
            "memory.search" => {
                let query = req
                    .params
                    .get("query")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let limit = req
                    .params
                    .get("limit")
                    .and_then(Value::as_u64)
                    .unwrap_or(10) as usize;
                self.memory_search(query, limit)
                    .and_then(|r| Ok(serde_json::to_value(r)?))
            }
            "skills.list" => self
                .skills_list()
                .and_then(|r| Ok(serde_json::to_value(r)?)),
            "skills.diff" => {
                let skill_id = req
                    .params
                    .get("skill_id")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                self.skills_diff(skill_id)
                    .map(|diff| json!({ "skill_id": skill_id, "diff": diff }))
            }
            "skills.stage" => {
                let skill_id = req
                    .params
                    .get("skill_id")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let title = req
                    .params
                    .get("title")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let body = req.params.get("body").and_then(Value::as_str).unwrap_or("");
                self.skills_stage(skill_id, title, body)
                    .and_then(|r| Ok(serde_json::to_value(r)?))
            }
            "skills.approve" => {
                let skill_id = req
                    .params
                    .get("skill_id")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let promote_dir = req
                    .params
                    .get("promote_dir")
                    .and_then(Value::as_str)
                    .map(PathBuf::from);
                self.skills_approve(skill_id, promote_dir.as_deref()).map(
                    |(record, promoted_path)| {
                        json!({
                            "record": record,
                            "promoted_path": promoted_path.map(|p| p.display().to_string()),
                        })
                    },
                )
            }
            "shutdown" => {
                should_shutdown = true;
                Ok(json!({ "shutting_down": true }))
            }
            other => Err(anyhow!("unknown daemon RPC method `{other}`")),
        };

        let resp = match outcome {
            Ok(result) => DaemonRpcResponse {
                id: req.id,
                result: Some(result),
                error: None,
            },
            Err(err) => DaemonRpcResponse {
                id: req.id,
                result: None,
                error: Some(DaemonRpcError {
                    code: 500,
                    message: format!("{err:#}"),
                }),
            },
        };
        (resp, should_shutdown)
    }
}

enum StepHalt {
    HumanPreempted,
    ProtectedSurface,
    Fatal(anyhow::Error),
}

fn execute_step_with_epoch_retry(
    ipc: &mut KomaIpcClient,
    ws_id: &str,
    current_epoch: &mut u64,
    step: &PlannedKomaStep,
    default_overlay: &str,
) -> Result<Duration, StepHalt> {
    if let PlannedKomaStep::WaitMs { ms } = step {
        return Ok(Duration::from_millis((*ms).min(5_000)));
    }

    for attempt in 0..=1 {
        let (res, settle_delay) = match step {
            PlannedKomaStep::ClickCell {
                cell,
                button,
                clicks,
            } => {
                let normalized = cell.trim().to_ascii_uppercase();
                if GridSpec::default().parse_cell_label(&normalized).is_none() {
                    return Err(StepHalt::Fatal(anyhow!(
                        "planner produced invalid grid_225 cell label `{cell}`"
                    )));
                }
                (
                    ipc.click_grid(ws_id, *current_epoch, &normalized, button, (*clicks).max(1)),
                    Duration::from_millis(35),
                )
            }
            PlannedKomaStep::ClickXy {
                x,
                y,
                button,
                clicks,
            } => (
                ipc.click_xy(ws_id, *current_epoch, *x, *y, button, (*clicks).max(1)),
                Duration::from_millis(35),
            ),
            PlannedKomaStep::DragXy {
                from_x,
                from_y,
                to_x,
                to_y,
                button,
                profile,
                duration_ms,
            } => {
                let norm_profile = normalize_kinematic_profile(profile);
                let dur = (*duration_ms).min(5_000);
                (
                    ipc.drag_xy(
                        ws_id,
                        *current_epoch,
                        *from_x,
                        *from_y,
                        *to_x,
                        *to_y,
                        button,
                        norm_profile,
                        dur,
                    ),
                    drag_settle_delay(norm_profile, dur),
                )
            }
            PlannedKomaStep::DragCell {
                from_cell,
                to_cell,
                button,
                profile,
                duration_ms,
            } => {
                let norm_from = from_cell.trim().to_ascii_uppercase();
                if GridSpec::default().parse_cell_label(&norm_from).is_none() {
                    return Err(StepHalt::Fatal(anyhow!(
                        "planner produced invalid grid_225 cell label `{from_cell}`"
                    )));
                }
                let norm_to = to_cell.trim().to_ascii_uppercase();
                if GridSpec::default().parse_cell_label(&norm_to).is_none() {
                    return Err(StepHalt::Fatal(anyhow!(
                        "planner produced invalid grid_225 cell label `{to_cell}`"
                    )));
                }
                let norm_profile = normalize_kinematic_profile(profile);
                let dur = (*duration_ms).min(5_000);
                (
                    ipc.drag_grid(
                        ws_id,
                        *current_epoch,
                        &norm_from,
                        &norm_to,
                        button,
                        norm_profile,
                        dur,
                    ),
                    drag_settle_delay(norm_profile, dur),
                )
            }
            PlannedKomaStep::InjectText { text } => (
                ipc.inject_text(ws_id, *current_epoch, text),
                Duration::from_millis(20),
            ),
            PlannedKomaStep::Key { chord } => (
                ipc.tap_key(ws_id, *current_epoch, chord),
                Duration::from_millis(20),
            ),
            PlannedKomaStep::WaitMs { .. } => unreachable!("handled prior to RPC retry loop"),
        };

        match res {
            Ok(new_epoch) => {
                *current_epoch = new_epoch;
                return Ok(settle_delay);
            }
            Err(KomaIpcError::Rpc(err)) if err.code == ERR_CODE_HUMAN_PREEMPTED => {
                return Err(StepHalt::HumanPreempted);
            }
            Err(KomaIpcError::Rpc(err)) if err.code == ERR_CODE_PROTECTED_SURFACE => {
                return Err(StepHalt::ProtectedSurface);
            }
            Err(KomaIpcError::Rpc(err)) if err.code == ERR_CODE_STALE_EPOCH && attempt == 0 => {
                let refreshed = ipc
                    .capture_frame(ws_id, default_overlay, 1280)
                    .map_err(|e| StepHalt::Fatal(e.into()))?;
                *current_epoch = err.current_epoch.unwrap_or(refreshed.epoch);
            }
            Err(err) => return Err(StepHalt::Fatal(err.into())),
        }
    }
    Err(StepHalt::Fatal(anyhow!(
        "exhausted stale-epoch retry budget on workspace `{ws_id}`"
    )))
}

struct SocketUnlinkGuard {
    path: PathBuf,
}

impl Drop for SocketUnlinkGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

/// Runs the persistent `jikud` Unix socket daemon until `SIGINT`, `SIGTERM`, or `shutdown` RPC.
pub async fn run_daemon_server(
    socket_override: Option<&Path>,
    config_path: Option<&Path>,
    policy_path: Option<&Path>,
    in_memory_db: bool,
) -> anyhow::Result<()> {
    let mut engine = JikuEngine::from_resolved(config_path, policy_path, in_memory_db)?;
    let socket_path = socket_override
        .map(expand_tilde_path)
        .unwrap_or_else(|| engine.config().daemon_socket_path());

    run_daemon_loop_on_path(&mut engine, &socket_path).await
}

async fn run_daemon_loop_on_path(
    engine: &mut JikuEngine,
    socket_path: &Path,
) -> anyhow::Result<()> {
    if let Some(parent) = socket_path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }

    if socket_path.exists() {
        if UnixStream::connect(socket_path).await.is_ok() {
            bail!(
                "another jikud instance is already listening on {}",
                socket_path.display()
            );
        }
        let _ = fs::remove_file(socket_path);
    }

    let listener = UnixListener::bind(socket_path)
        .with_context(|| format!("failed binding jikud socket {}", socket_path.display()))?;
    fs::set_permissions(socket_path, fs::Permissions::from_mode(0o600))?;
    let _guard = SocketUnlinkGuard {
        path: socket_path.to_path_buf(),
    };

    tracing::info!(socket = %socket_path.display(), "jikud daemon listening");

    let expected_uid = rustix::process::getuid().as_raw();
    let mut sigterm = signal(SignalKind::terminate())?;

    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                tracing::info!("received SIGINT; shutting down jikud");
                break;
            }
            _ = sigterm.recv() => {
                tracing::info!("received SIGTERM; shutting down jikud");
                break;
            }
            accepted = listener.accept() => {
                let (mut stream, _) = match accepted {
                    Ok(pair) => pair,
                    Err(e) => {
                        tracing::warn!("accept error on jikud socket: {e}");
                        continue;
                    }
                };
                match stream.peer_cred() {
                    Ok(cred) if cred.uid() == expected_uid => {}
                    Ok(cred) => {
                        tracing::warn!(peer_uid = cred.uid(), expected_uid, "rejected unauthorized UID on jiku.sock");
                        continue;
                    }
                    Err(e) => {
                        tracing::warn!("failed inspecting SO_PEERCRED: {e}");
                        continue;
                    }
                }

                match read_daemon_frame::<DaemonRpcRequest>(&mut stream).await {
                    Ok(req) => {
                        let (resp, shutdown) = engine.handle_rpc(&req).await;
                        if let Err(e) = write_daemon_frame(&mut stream, &resp).await {
                            tracing::warn!("failed writing daemon response: {e}");
                        }
                        if shutdown {
                            tracing::info!("received shutdown RPC; exiting cleanly");
                            break;
                        }
                    }
                    Err(e) => {
                        let err_resp = DaemonRpcResponse {
                            id: 0,
                            result: None,
                            error: Some(DaemonRpcError {
                                code: 400,
                                message: format!("malformed request frame: {e}"),
                            }),
                        };
                        let _ = write_daemon_frame(&mut stream, &err_resp).await;
                    }
                }
            }
        }
    }

    Ok(())
}

/// Sends a single length-prefixed JSON RPC frame to a listening `jikud` socket.
pub async fn send_daemon_rpc(
    socket_path: &Path,
    req: &DaemonRpcRequest,
) -> anyhow::Result<DaemonRpcResponse> {
    let mut stream = UnixStream::connect(socket_path).await.with_context(|| {
        format!(
            "failed connecting to jikud socket {}",
            socket_path.display()
        )
    })?;
    write_daemon_frame(&mut stream, req).await?;
    let resp: DaemonRpcResponse = read_daemon_frame(&mut stream).await?;
    Ok(resp)
}

async fn write_daemon_frame<T: Serialize>(
    stream: &mut UnixStream,
    value: &T,
) -> anyhow::Result<()> {
    let payload = serde_json::to_vec(value)?;
    if payload.is_empty() || payload.len() > MAX_DAEMON_FRAME_BYTES {
        bail!("daemon RPC frame size out of bounds");
    }
    let len_le = (payload.len() as u32).to_le_bytes();
    stream.write_all(&len_le).await?;
    stream.write_all(&payload).await?;
    stream.flush().await?;
    Ok(())
}

async fn read_daemon_frame<T: for<'de> Deserialize<'de>>(
    stream: &mut UnixStream,
) -> anyhow::Result<T> {
    let mut len_buf = [0_u8; 4];
    stream.read_exact(&mut len_buf).await?;
    let len = u32::from_le_bytes(len_buf) as usize;
    if len == 0 || len > MAX_DAEMON_FRAME_BYTES {
        bail!("incoming daemon RPC frame length {len} exceeds bounds");
    }
    let mut buf = vec![0_u8; len];
    stream.read_exact(&mut buf).await?;
    Ok(serde_json::from_slice(&buf)?)
}

fn build_tts_provider(config: &JikuConfig) -> Box<dyn TtsProvider> {
    if !config.tts.enabled {
        return Box::new(NoopTtsProvider::default());
    }
    match config.tts.provider.trim().to_ascii_lowercase().as_str() {
        "voicevox" => VoicevoxProvider::new(&config.tts.endpoint, config.tts.speaker_id)
            .map(|p| Box::new(p) as Box<dyn TtsProvider>)
            .unwrap_or_else(|_| Box::new(NoopTtsProvider::default())),
        "openai_speech" => {
            let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
            OpenAiSpeechProvider::new(&config.tts.endpoint, api_key, "gpt-4o-mini-tts", "alloy")
                .map(|p| Box::new(p) as Box<dyn TtsProvider>)
                .unwrap_or_else(|_| Box::new(NoopTtsProvider::default()))
        }
        _ => Box::new(NoopTtsProvider::default()),
    }
}

fn build_decision_questions(
    raw: &[RpcDecisionQuestion],
    default_prompt: &str,
) -> anyhow::Result<Vec<(String, DecisionQuestionKind)>> {
    if raw.is_empty() {
        if default_prompt.trim().is_empty() {
            bail!("decide requires either a prompt or at least one --question / --choice");
        }
        return Ok(vec![(
            "q_yes_no".to_string(),
            DecisionQuestionKind::Noul {
                instructions: default_prompt.trim().to_string(),
            },
        )]);
    }

    let first_noul_id = raw.iter().find_map(|q| match q {
        RpcDecisionQuestion::Noul { id, .. } => Some(id.clone()),
        RpcDecisionQuestion::Choice { .. } => None,
    });
    let has_choice_needing_gate = raw.iter().any(|q| match q {
        RpcDecisionQuestion::Choice {
            scope_gate_question_id,
            ..
        } => scope_gate_question_id
            .as_deref()
            .is_none_or(|s| s.trim().is_empty()),
        RpcDecisionQuestion::Noul { .. } => false,
    });

    let mut out = Vec::with_capacity(raw.len() + 1);
    let fallback_gate_id = if let Some(id) = first_noul_id {
        Some(id)
    } else if has_choice_needing_gate {
        let gate_id = "q_scope_gate".to_string();
        out.push((
            gate_id.clone(),
            DecisionQuestionKind::Noul {
                instructions: if default_prompt.trim().is_empty() {
                    "Is the target context visible and applicable to the choices?".to_string()
                } else {
                    format!("Is this context applicable: {}", default_prompt.trim())
                },
            },
        ));
        Some(gate_id)
    } else {
        None
    };

    for item in raw {
        match item {
            RpcDecisionQuestion::Noul { id, instructions } => {
                out.push((
                    id.clone(),
                    DecisionQuestionKind::Noul {
                        instructions: instructions.clone(),
                    },
                ));
            }
            RpcDecisionQuestion::Choice {
                id,
                instructions,
                choices,
                scope_gate_question_id,
            } => {
                let mut padded = choices.clone();
                if !padded.iter().any(|c| c == MANDATORY_CHOICE_FALLBACK) {
                    padded.push(MANDATORY_CHOICE_FALLBACK.to_string());
                }
                let gate_id = scope_gate_question_id
                    .clone()
                    .filter(|s| !s.trim().is_empty())
                    .or_else(|| fallback_gate_id.clone())
                    .unwrap_or_default();
                out.push((
                    id.clone(),
                    DecisionQuestionKind::Choice {
                        instructions: instructions.clone(),
                        choices: padded,
                        scope_gate_question_id: gate_id,
                    },
                ));
            }
        }
    }

    Ok(out)
}

fn load_optional_media(
    image_path: Option<&Path>,
) -> anyhow::Result<(DecisionMediaSource, Option<DriverMediaInput>)> {
    let Some(path) = image_path else {
        return Ok((
            DecisionMediaSource::InlineBytes {
                mime_type: "image/png".to_string(),
            },
            None,
        ));
    };
    let resolved = expand_tilde_path(path);
    let bytes = fs::read(&resolved)
        .with_context(|| format!("failed reading image attachment {}", resolved.display()))?;
    if bytes.is_empty() {
        bail!("image attachment is empty: {}", resolved.display());
    }
    let mime = detect_image_mime(&resolved, &bytes).to_string();
    Ok((
        DecisionMediaSource::InlineBytes {
            mime_type: mime.clone(),
        },
        Some(DriverMediaInput::Inline {
            mime_type: mime,
            bytes,
        }),
    ))
}

fn detect_image_mime(path: &Path, bytes: &[u8]) -> &'static str {
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        return "image/png";
    }
    if bytes.starts_with(&[0xff, 0xd8, 0xff]) {
        return "image/jpeg";
    }
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase())
        .as_deref()
    {
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("webp") => "image/webp",
        Some("gif") => "image/gif",
        _ => "image/png",
    }
}

fn drag_settle_delay(profile: &str, duration_ms: u64) -> Duration {
    match profile {
        "instant" => Duration::from_millis(25),
        "spring" => Duration::from_millis(duration_ms.max(260) + 60),
        _ => Duration::from_millis(duration_ms + 50),
    }
}

#[allow(clippy::too_many_arguments)]
fn build_runtime_context_summary(
    ws_id: &str,
    goal: &str,
    inventory: &[KomaWorkspaceSummary],
    active_app_id: &str,
    active_title: &str,
    active_win_id: &str,
    epoch: Option<u64>,
    has_protected_blackout: bool,
    injected_notes: &[String],
) -> String {
    let mut lines = vec![
        format!("Goal: {goal}"),
        format!("Workspace: {ws_id}"),
        format!(
            "Epoch: {}",
            epoch.map_or_else(|| "none".to_string(), |e| e.to_string())
        ),
        format!(
            "Active window: win_id={active_win_id}, app_id={active_app_id}, title={active_title}"
        ),
        format!("Protected blackout present: {has_protected_blackout}"),
    ];
    if let Some(ws) = inventory.iter().find(|w| w.ws_id == ws_id) {
        lines.push(format!(
            "Viewport: {}x{}, layout={}",
            ws.viewport_width, ws.viewport_height, ws.layout_mode
        ));
        if let Some(win) = &ws.active_window
            && let (Some(x), Some(y), Some(w), Some(h)) =
                (win.rect_x, win.rect_y, win.rect_w, win.rect_h)
        {
            lines.push(format!("Active window geometry: {w}x{h} at ({x}, {y})"));
        }
    }
    if !injected_notes.is_empty() {
        lines.push("Prefetch notes:".to_string());
        for note in injected_notes {
            lines.push(format!("- {note}"));
        }
    }
    lines.join("\n")
}

fn build_planner_system_prompt() -> String {
    concat!(
        "You are Jiku's desktop action planner operating inside the Koma Wayland compositor.\n",
        "Respond with STRICT JSON ONLY (no markdown fences) matching:\n",
        "{\n",
        "  \"summary\": \"one concise sentence describing the plan\",\n",
        "  \"steps\": [\n",
        "    {\"action\": \"click_cell\", \"cell\": \"H8\", \"button\": \"left\", \"clicks\": 1},\n",
        "    {\"action\": \"click_xy\", \"x\": 640, \"y\": 400, \"button\": \"left\", \"clicks\": 1},\n",
        "    {\"action\": \"drag_xy\", \"from_x\": 180, \"from_y\": 260, \"to_x\": 520, \"to_y\": 260, \"button\": \"left\", \"profile\": \"spring\", \"duration_ms\": 140},\n",
        "    {\"action\": \"drag_cell\", \"from_cell\": \"C4\", \"to_cell\": \"M4\", \"button\": \"left\", \"profile\": \"bezier\", \"duration_ms\": 120},\n",
        "    {\"action\": \"inject_text\", \"text\": \"command without trailing newline\"},\n",
        "    {\"action\": \"key\", \"chord\": \"Return\"},\n",
        "    {\"action\": \"wait_ms\", \"ms\": 150}\n",
        "  ],\n",
        "  \"candidate_skill\": null\n",
        "}\n",
        "Rules:\n",
        "- Only use action values: \"click_cell\", \"click_xy\", \"drag_xy\", \"drag_cell\", \"inject_text\", \"key\", \"wait_ms\".\n",
        "- Grid overlay cells use 15x15 labels A1..O15 (columns A..O left-to-right, rows 1..15 top-to-bottom).\n",
        "- When `Active window geometry: WxH at (X, Y)` is present, `click_xy` and `drag_xy` coordinates are local to that active window (`0..W`, `0..H`).\n",
        "- For `drag_xy` / `drag_cell`, `profile` may be \"spring\" (damped Hooke spring dynamics), \"bezier\", \"minimum_jerk\", or \"instant\".\n",
        "- When the active window is already a terminal (such as foot) and you want to run a command, click the terminal center if unfocused or directly emit inject_text (without newline) followed by {\"action\": \"key\", \"chord\": \"Return\"}.\n",
        "- Keep steps <= 48 and preserve exact text requested by the user."
    )
    .to_string()
}

fn append_audit_line(path: &Path, entry: &Value) -> anyhow::Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(file, "{}", serde_json::to_string(entry)?)?;
    Ok(())
}

fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    fn unique_temp_dir(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        std::env::temp_dir().join(format!("{prefix}-{}-{nanos}", std::process::id()))
    }

    #[tokio::test(flavor = "current_thread")]
    async fn daemon_unix_socket_roundtrip_status_skills_and_shutdown() {
        let tmp_dir = unique_temp_dir("jiku-daemon-test");
        fs::create_dir_all(&tmp_dir).expect("create test temp dir");
        let sock_path = tmp_dir.join("jiku.sock");

        let mut engine = JikuEngine::from_resolved(None, None, true).expect("engine init");
        assert_eq!(engine.config().planner_binding().1, "deepseek-v4.1-flash");

        let sock_for_server = sock_path.clone();
        let server_fut = run_daemon_loop_on_path(&mut engine, &sock_for_server);

        let client_fut = async {
            for _ in 0..50 {
                if sock_path.exists() {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }

            let status_resp = send_daemon_rpc(
                &sock_path,
                &DaemonRpcRequest {
                    id: 1,
                    method: "status".to_string(),
                    params: json!({}),
                },
            )
            .await
            .expect("status RPC");
            assert!(status_resp.error.is_none());
            let status_val = status_resp.result.expect("status result");
            assert_eq!(status_val["planner_model"], "deepseek-v4.1-flash");
            assert_eq!(status_val["planner_thinking_enabled"], false);

            let stage_resp = send_daemon_rpc(
                &sock_path,
                &DaemonRpcRequest {
                    id: 2,
                    method: "skills.stage".to_string(),
                    params: json!({
                        "skill_id": "test_skill",
                        "title": "Test Skill",
                        "body": "-- safe comment\nreturn true",
                    }),
                },
            )
            .await
            .expect("stage RPC");
            assert!(stage_resp.error.is_none());

            let diff_resp = send_daemon_rpc(
                &sock_path,
                &DaemonRpcRequest {
                    id: 3,
                    method: "skills.diff".to_string(),
                    params: json!({ "skill_id": "test_skill" }),
                },
            )
            .await
            .expect("diff RPC");
            assert!(
                diff_resp.result.expect("diff result")["diff"]
                    .as_str()
                    .unwrap_or_default()
                    .contains("safe comment")
            );

            let approve_resp = send_daemon_rpc(
                &sock_path,
                &DaemonRpcRequest {
                    id: 4,
                    method: "skills.approve".to_string(),
                    params: json!({ "skill_id": "test_skill" }),
                },
            )
            .await
            .expect("approve RPC");
            assert_eq!(
                approve_resp.result.expect("approve result")["record"]["status"],
                "approved"
            );

            let shutdown_resp = send_daemon_rpc(
                &sock_path,
                &DaemonRpcRequest {
                    id: 5,
                    method: "shutdown".to_string(),
                    params: json!({}),
                },
            )
            .await
            .expect("shutdown RPC");
            assert_eq!(
                shutdown_resp.result.expect("shutdown result")["shutting_down"],
                true
            );
        };

        let (server_res, ()) = tokio::join!(server_fut, client_fut);
        server_res.expect("daemon server clean exit");
        assert!(!sock_path.exists());
        let _ = fs::remove_dir_all(&tmp_dir);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn episode_completed_payload_passes_examples_init_lua_policy() {
        let policy_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples/init.lua");
        let mut engine = JikuEngine::from_resolved(None, Some(&policy_path), true)
            .expect("engine with examples/init.lua");
        let payload = JikuEngine::episode_completed_payload(
            1,
            "ws_01",
            "Verify duration_ms",
            "fast_path_koma",
            "done",
            false,
            None,
            2,
            6200,
        );
        engine
            .dispatch_lifecycle_event(
                "episode.completed",
                payload,
                AvatarEmotion::Success,
                Some("done"),
            )
            .await
            .expect("examples/init.lua episode.completed handler must accept duration_ms");
    }

    #[test]
    fn planned_koma_step_drag_and_wait_serde_defaults_and_settle_delays() {
        let raw = json!([
            {
                "action": "drag_xy",
                "from_x": 120.0,
                "from_y": 240.0,
                "to_x": 420.0,
                "to_y": 300.0
            },
            {
                "action": "drag_cell",
                "from_cell": "C4",
                "to_cell": "M9",
                "profile": "bezier",
                "duration_ms": 180
            },
            {
                "action": "wait_ms",
                "ms": 250
            }
        ]);
        let steps: Vec<PlannedKomaStep> = serde_json::from_value(raw).unwrap();
        assert_eq!(steps.len(), 3);
        assert_eq!(
            steps[0],
            PlannedKomaStep::DragXy {
                from_x: 120.0,
                from_y: 240.0,
                to_x: 420.0,
                to_y: 300.0,
                button: "left".to_string(),
                profile: "spring".to_string(),
                duration_ms: 120,
            }
        );
        assert_eq!(drag_settle_delay("spring", 120), Duration::from_millis(320));
        assert_eq!(drag_settle_delay("bezier", 180), Duration::from_millis(230));
        assert_eq!(drag_settle_delay("instant", 180), Duration::from_millis(25));
        assert_eq!(normalize_kinematic_profile("Hooke"), "spring");
        assert_eq!(MAX_TASK_ACTIONS, 48);

        let inv = vec![KomaWorkspaceSummary {
            ws_id: "ws_01".to_string(),
            name: "main".to_string(),
            layout_mode: "dwindle".to_string(),
            epoch: 3,
            is_human_preempted: false,
            viewport_width: 1280,
            viewport_height: 800,
            active_window: Some(crate::koma_ipc::KomaWindowSummary {
                win_id: "win_1_g1".to_string(),
                app_id: "koma-paint-probe".to_string(),
                title: "Koma Paint".to_string(),
                is_floating: false,
                is_sandboxed: false,
                rect_x: Some(12),
                rect_y: Some(12),
                rect_w: Some(1256),
                rect_h: Some(776),
            }),
        }];
        let ctx = build_runtime_context_summary(
            "ws_01",
            "Paint a sunlit house",
            &inv,
            "koma-paint-probe",
            "Koma Paint",
            "win_1_g1",
            Some(3),
            false,
            &[],
        );
        assert!(ctx.contains("Active window geometry: 1256x776 at (12, 12)"));
    }
}
