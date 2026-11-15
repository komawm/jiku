//! Native Unix domain socket IPC client for Koma (`koma-admin.sock` & `koma-agent.sock`).
//!
//! Implements 4-byte little-endian JSON framing, `SCM_RIGHTS` sealed `memfd` frame
//! reception (`File::read_exact`), PNG encoding, perceptual `dHash` computation, and
//! deterministic `Epoch`-fenced `vseat.*` dispatch.

use std::{
    collections::{BTreeMap, VecDeque},
    env,
    fs::File,
    io::{self, IoSlice, IoSliceMut, Read as _, Seek as _, SeekFrom},
    mem::MaybeUninit,
    os::{
        fd::{AsFd as _, BorrowedFd, OwnedFd},
        unix::net::UnixStream,
    },
    path::{Path, PathBuf},
};

use jiku_memory::compute_rgba_dhash64;
use rustix::{
    cmsg_space,
    io::Errno,
    net::{
        RecvAncillaryBuffer, RecvAncillaryMessage, RecvFlags, SendAncillaryBuffer,
        SendAncillaryMessage, SendFlags, recvmsg, sendmsg,
    },
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use thiserror::Error;

/// Canonical Koma wire protocol version.
pub const KOMA_PROTOCOL_VERSION: &str = "0.1";
const ADMIN_SOCKET_NAME: &str = "koma-admin.sock";
const AGENT_SOCKET_NAME: &str = "koma-agent.sock";
const MAX_FRAME_BYTES: usize = 16 * 1024 * 1024;
const MAX_SCM_FDS: usize = 4;

/// Koma wire IPC error symbols / numeric codes (`docs/IPC_SPEC.md`).
pub const ERR_CODE_PROTECTED_SURFACE: u16 = 4032;
/// Workspace epoch stale (`4090 ERR_STALE_EPOCH`): requires re-observation before mutating.
pub const ERR_CODE_STALE_EPOCH: u16 = 4090;
/// Human Mod+G takeover preempted workspace (`4091 ERR_HUMAN_PREEMPTED`).
pub const ERR_CODE_HUMAN_PREEMPTED: u16 = 4091;

/// Wire request envelope sent to Koma.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KomaRpcRequest {
    /// Monotonic client request ID.
    pub id: u64,
    /// RPC method name (`scene.inventory`, `capture.frame`, `vseat.click`, ...).
    pub method: String,
    /// Method parameter JSON object.
    #[serde(default)]
    pub params: Value,
}

/// Compositor delivery disposition reported on IPC errors (`KOM-0027`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeliveryDisposition {
    /// Delivery status could not be determined.
    #[default]
    Unknown,
    /// Compositor confirmed the request was rejected before applying input.
    NotDelivered,
    /// Input may or may not have reached the target surface.
    DeliveryUncertain,
    /// Input was injected without post-step visual confirmation.
    DeliveredUnverified,
    /// Input was injected and post-step verification succeeded.
    DeliveredVerified,
}

/// Retry safety guidance paired with [`DeliveryDisposition`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetryDisposition {
    /// Re-issuing the RPC after re-observation is safe.
    Safe,
    /// Blind retry risks duplicating side effects.
    Unsafe,
    /// Retry safety could not be determined.
    #[default]
    Unknown,
}

/// Combined delivery and retry semantics attached to [`KomaRpcError`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct DeliverySemantics {
    /// Whether the compositor applied the request prior to failing.
    pub delivery: DeliveryDisposition,
    /// Whether retrying the operation is considered safe.
    pub retry: RetryDisposition,
}

/// Structured error object returned inside [`KomaRpcResponse::error`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Error)]
#[error("[{code}] {message}")]
pub struct KomaRpcError {
    /// Numeric Koma IPC error code (`4000..=5040`).
    pub code: u16,
    /// Symbolic error string (e.g. `ERR_STALE_EPOCH`), if returned.
    #[serde(default)]
    pub symbol: Option<String>,
    /// Human-readable diagnostic message.
    pub message: String,
    /// Latest compositor workspace epoch when stale.
    #[serde(default)]
    pub current_epoch: Option<u64>,
    /// Whether retrying after re-observation is safe.
    #[serde(default)]
    pub retryable: Option<bool>,
    /// Optional delivery/retry semantics (`KOM-0027`).
    #[serde(default)]
    pub disposition: Option<DeliverySemantics>,
}

/// Wire response envelope returned by Koma.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KomaRpcResponse {
    /// Matching request ID.
    pub id: u64,
    /// Result payload on success.
    #[serde(default)]
    pub result: Option<Value>,
    /// Error descriptor on failure.
    #[serde(default)]
    pub error: Option<KomaRpcError>,
}

/// Errors raised when connecting to or exchanging frames with Koma.
#[derive(Debug, Error)]
pub enum KomaIpcError {
    /// No active `koma-agent.sock` runtime directory could be located.
    #[error("no active Koma runtime socket found: {0}")]
    NotFound(String),

    /// Missing authentication token and unreachable `koma-admin.sock`.
    #[error("Koma authentication unavailable: {0}")]
    Auth(String),

    /// Underlying Unix socket / `SCM_RIGHTS` / `memfd` I/O failure.
    #[error("Koma IPC I/O error: {0}")]
    Io(#[from] io::Error),

    /// JSON frame encode/decode error.
    #[error("Koma IPC JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Remote compositor returned an explicit IPC error envelope.
    #[error("Koma RPC error: {0}")]
    Rpc(#[from] KomaRpcError),
}

impl KomaIpcError {
    /// Returns structured delivery/retry semantics attached to a [`KomaRpcError`], if present.
    #[must_use]
    pub fn disposition(&self) -> Option<DeliverySemantics> {
        match self {
            Self::Rpc(rpc) => rpc.disposition,
            _ => None,
        }
    }
}

/// Compositor hit-test & modal occlusion verification result (`vseat.hit_test`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KomaHitTestResult {
    /// Whether sampled target coordinates reach the intended target window.
    pub reaches_target: bool,
    /// Target window identifier, if known.
    #[serde(default)]
    pub target_win_id: Option<String>,
    /// Target Wayland `app_id`, if known.
    #[serde(default)]
    pub target_app_id: Option<String>,
    /// Occluding/blocking window identifier, if blocked.
    #[serde(default)]
    pub blocking_win_id: Option<String>,
    /// Occluding window Wayland `app_id`, if blocked.
    #[serde(default)]
    pub blocking_app_id: Option<String>,
    /// Occluding window title, if blocked.
    #[serde(default)]
    pub blocking_title: Option<String>,
    /// Occluding window rect `[x, y, width, height]`, if blocked.
    #[serde(default)]
    pub blocking_rect: Option<[i32; 4]>,
    /// True if sampled window geometry remained stable during the hit-test pass.
    #[serde(default)]
    pub geometry_stable: bool,
    /// Coordinates `[x, y]` sampled across the target region.
    #[serde(default)]
    pub sampled_points: Vec<[f64; 2]>,
    /// Current cursor position `[x, y]`, if available.
    #[serde(default)]
    pub cursor_pos: Option<[f64; 2]>,
}

/// Window metadata summarized from `scene.inventory`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KomaWindowSummary {
    /// Stable generation-tagged window identifier (e.g. `"win_1_g1"`).
    pub win_id: String,
    /// Wayland `app_id`.
    pub app_id: String,
    /// Window title.
    pub title: String,
    /// Floating flag.
    pub is_floating: bool,
    /// Sandbox profile flag.
    pub is_sandboxed: bool,
    /// Logical window X coordinate when reported by `scene.inventory`.
    #[serde(default)]
    pub rect_x: Option<i32>,
    /// Logical window Y coordinate when reported by `scene.inventory`.
    #[serde(default)]
    pub rect_y: Option<i32>,
    /// Logical window width when reported by `scene.inventory`.
    #[serde(default)]
    pub rect_w: Option<u32>,
    /// Logical window height when reported by `scene.inventory`.
    #[serde(default)]
    pub rect_h: Option<u32>,
}

/// Workspace snapshot entry returned by `scene.inventory`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KomaWorkspaceSummary {
    /// Logical workspace identifier (`"ws_01"`).
    pub ws_id: String,
    /// Workspace name.
    pub name: String,
    /// Layout mode (`"dwindle"`, `"scrolling"`, `"deck"`).
    pub layout_mode: String,
    /// Current workspace concurrency epoch.
    pub epoch: u64,
    /// True when human `Mod+G` takeover is active.
    pub is_human_preempted: bool,
    /// Logical viewport width in pixels.
    pub viewport_width: u32,
    /// Logical viewport height in pixels.
    pub viewport_height: u32,
    /// Currently active window on this workspace, if any.
    pub active_window: Option<KomaWindowSummary>,
}

/// Zero-copy screenshot capture encoded to PNG + perceptual hash.
#[derive(Debug, Clone)]
pub struct KomaCapturedFrame {
    /// Workspace captured.
    pub ws_id: String,
    /// Observed workspace epoch at frame export.
    pub epoch: u64,
    /// Output pixel width (possibly downscaled to `max_edge`).
    pub width: u32,
    /// Output pixel height.
    pub height: u32,
    /// True if Ring 0 protected-surface blackout obscured sensitive surfaces.
    pub has_protected_blackout: bool,
    /// Number of overlay grid cells rendered by Koma (typically `225`).
    pub grid_cells: Option<u16>,
    /// 64-bit `dHash` computed directly over raw RGBA pixels.
    pub dhash64: u64,
    /// Encoded PNG image bytes ready for vision/decisions inference.
    pub png_bytes: Vec<u8>,
}

/// Waypoint anchor along a smooth `vseat.path` cursor trajectory.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum KomaPathWaypoint {
    /// Explicit window-local logical coordinate `(x, y)`.
    Xy {
        /// Horizontal logical pixel coordinate.
        x: f64,
        /// Vertical logical pixel coordinate.
        y: f64,
    },
    /// 15x15 `grid_225` cell label (e.g. `"H8"`).
    Grid225(String),
}

impl KomaPathWaypoint {
    /// Serializes this waypoint into the Koma `vseat.path` wire object shape.
    #[must_use]
    pub fn to_json_value(&self) -> Value {
        match self {
            Self::Xy { x, y } => json!({ "x": x, "y": y }),
            Self::Grid225(cell) => json!({ "grid_225": cell }),
        }
    }
}

/// Authenticated client connection to Koma's `koma-agent.sock`.
pub struct KomaIpcClient {
    stream: UnixStream,
    next_id: u64,
    instance_dir: PathBuf,
    session_id: String,
    read_buf: Vec<u8>,
    pending_fds: VecDeque<OwnedFd>,
}

impl KomaIpcClient {
    /// Discovers `$XDG_RUNTIME_DIR/koma/<instance>/`, authenticates using `$JIKU_KOMA_TOKEN`
    /// or `$KOMA_CAPABILITY_TOKEN` (minting an ephemeral local token via `koma-admin.sock`
    /// when `allow_admin_mint` is `true`), and completes Frame #1 `auth.authenticate`.
    pub fn connect_discovered(
        default_workspace: &str,
        allow_admin_mint: bool,
    ) -> Result<Self, KomaIpcError> {
        let dir = discover_koma_instance_dir()?;
        let token = match env::var("JIKU_KOMA_TOKEN")
            .ok()
            .or_else(|| env::var("KOMA_CAPABILITY_TOKEN").ok())
            .filter(|v| !v.trim().is_empty())
        {
            Some(t) => t.trim().to_string(),
            None if allow_admin_mint => {
                mint_local_token(&dir.join(ADMIN_SOCKET_NAME), default_workspace)?
            }
            None => {
                return Err(KomaIpcError::Auth(
                    "neither KOMA_CAPABILITY_TOKEN nor JIKU_KOMA_TOKEN is set".to_string(),
                ));
            }
        };

        Self::connect_with_token(&dir, &token)
    }

    /// Connects directly to `instance_dir/koma-agent.sock` and authenticates with `token`.
    pub fn connect_with_token(instance_dir: &Path, token: &str) -> Result<Self, KomaIpcError> {
        let agent_sock = instance_dir.join(AGENT_SOCKET_NAME);
        let stream = UnixStream::connect(&agent_sock).map_err(|e| {
            KomaIpcError::NotFound(format!("cannot connect to {}: {e}", agent_sock.display()))
        })?;

        let auth_req = KomaRpcRequest {
            id: 1,
            method: "auth.authenticate".to_string(),
            params: json!({
                "token": token,
                "client_name": "jikud",
                "protocol_version": KOMA_PROTOCOL_VERSION,
            }),
        };
        let (resp, _): (KomaRpcResponse, _) = sync_roundtrip(&stream, &auth_req, Vec::new())?;
        if let Some(err) = resp.error {
            return Err(KomaIpcError::Rpc(err));
        }
        let session_id = resp
            .result
            .as_ref()
            .and_then(|v| v.get("session_id"))
            .and_then(Value::as_str)
            .unwrap_or("sess_jiku")
            .to_string();

        Ok(Self {
            stream,
            next_id: 2,
            instance_dir: instance_dir.to_path_buf(),
            session_id,
            read_buf: Vec::new(),
            pending_fds: VecDeque::new(),
        })
    }

    /// Returns the discovered Koma instance directory (`.../koma/<sig>`).
    #[must_use]
    pub fn instance_dir(&self) -> &Path {
        &self.instance_dir
    }

    /// Returns the authenticated compositor session ID.
    #[must_use]
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    fn alloc_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    fn roundtrip_rpc(
        &mut self,
        req: &KomaRpcRequest,
        req_fds: Vec<OwnedFd>,
    ) -> io::Result<(KomaRpcResponse, Vec<OwnedFd>)> {
        let payload = serde_json::to_vec(req)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e.to_string()))?;
        send_frame_with_fds(&self.stream, &payload, req_fds)?;
        loop {
            let body =
                read_next_raw_frame(&self.stream, &mut self.read_buf, &mut self.pending_fds)?;
            let raw_val: Value = serde_json::from_slice(&body)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
            // Skip unsolicited Koma workspace EventEnvelope notifications (no matching request `id`).
            if raw_val.get("id").and_then(Value::as_u64) != Some(req.id) {
                continue;
            }
            let parsed: KomaRpcResponse = serde_json::from_value(raw_val)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
            let fds: Vec<OwnedFd> = self.pending_fds.drain(..).collect();
            return Ok((parsed, fds));
        }
    }

    /// Sends a generic JSON RPC over `koma-agent.sock` and returns the result JSON value.
    pub fn call_rpc(&mut self, method: &str, params: Value) -> Result<Value, KomaIpcError> {
        let req = KomaRpcRequest {
            id: self.alloc_id(),
            method: method.to_string(),
            params,
        };
        let (resp, _): (KomaRpcResponse, _) = self.roundtrip_rpc(&req, Vec::new())?;
        if let Some(err) = resp.error {
            return Err(KomaIpcError::Rpc(err));
        }
        Ok(resp.result.unwrap_or(Value::Null))
    }

    /// Queries `scene.inventory` and parses visible workspaces.
    pub fn inventory(&mut self) -> Result<Vec<KomaWorkspaceSummary>, KomaIpcError> {
        let val = self.call_rpc("scene.inventory", json!({}))?;
        let mut out = Vec::new();
        let Some(workspaces) = val.get("workspaces").and_then(Value::as_array) else {
            return Ok(out);
        };

        for ws in workspaces {
            let ws_id = ws
                .get("ws_id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let name = ws
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let layout_mode = ws
                .get("layout_mode")
                .and_then(Value::as_str)
                .unwrap_or("dwindle")
                .to_string();
            let epoch = ws.get("epoch").and_then(Value::as_u64).unwrap_or(1);
            let is_human_preempted = ws
                .get("is_human_preempted")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let viewport = ws.get("viewport");
            let vw = viewport
                .and_then(|v| v.get("width"))
                .and_then(Value::as_u64)
                .unwrap_or(1280) as u32;
            let vh = viewport
                .and_then(|v| v.get("height"))
                .and_then(Value::as_u64)
                .unwrap_or(800) as u32;

            let active_window = ws.get("active_window").filter(|v| !v.is_null()).map(|w| {
                let rect = w.get("rect");
                KomaWindowSummary {
                    win_id: w
                        .get("win_id")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    app_id: w
                        .get("app_id")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    title: w
                        .get("title")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    is_floating: w
                        .get("is_floating")
                        .and_then(Value::as_bool)
                        .unwrap_or(false),
                    is_sandboxed: w
                        .get("is_sandboxed")
                        .and_then(Value::as_bool)
                        .unwrap_or(false),
                    rect_x: rect
                        .and_then(|r| r.get("x"))
                        .and_then(Value::as_i64)
                        .and_then(|v| i32::try_from(v).ok()),
                    rect_y: rect
                        .and_then(|r| r.get("y"))
                        .and_then(Value::as_i64)
                        .and_then(|v| i32::try_from(v).ok()),
                    rect_w: rect
                        .and_then(|r| r.get("w"))
                        .and_then(Value::as_u64)
                        .and_then(|v| u32::try_from(v).ok()),
                    rect_h: rect
                        .and_then(|r| r.get("h"))
                        .and_then(Value::as_u64)
                        .and_then(|v| u32::try_from(v).ok()),
                }
            });

            out.push(KomaWorkspaceSummary {
                ws_id,
                name,
                layout_mode,
                epoch,
                is_human_preempted,
                viewport_width: vw,
                viewport_height: vh,
                active_window,
            });
        }

        Ok(out)
    }

    /// Captures `ws_id` via zero-copy `SCM_RIGHTS` sealed `memfd` (`shm_fd`, `raw_rgba8`),
    /// computes 64-bit `dHash`, and encodes compressed PNG bytes via `png::Encoder`.
    pub fn capture_frame(
        &mut self,
        ws_id: &str,
        overlay_kind: &str,
        max_edge: u32,
    ) -> Result<KomaCapturedFrame, KomaIpcError> {
        let req = KomaRpcRequest {
            id: self.alloc_id(),
            method: "capture.frame".to_string(),
            params: json!({
                "ws_id": ws_id,
                "win_id": null,
                "overlay_kind": overlay_kind,
                "max_edge": max_edge,
                "format": "raw_rgba8",
                "transport": "shm_fd",
            }),
        };
        let (resp, mut fds): (KomaRpcResponse, _) = self.roundtrip_rpc(&req, Vec::new())?;
        if let Some(err) = resp.error {
            return Err(KomaIpcError::Rpc(err));
        }

        let res = resp.result.unwrap_or(Value::Null);
        let epoch = res.get("epoch").and_then(Value::as_u64).unwrap_or(1);
        let width = res.get("width").and_then(Value::as_u64).unwrap_or(0) as u32;
        let height = res.get("height").and_then(Value::as_u64).unwrap_or(0) as u32;
        let byte_len = res.get("byte_len").and_then(Value::as_u64).unwrap_or(0) as usize;
        let has_protected_blackout = res
            .get("has_protected_blackout")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let grid_cells = res
            .get("grid_cells")
            .and_then(Value::as_u64)
            .map(|v| v as u16);

        let memfd = fds.pop().ok_or_else(|| {
            KomaIpcError::Io(io::Error::new(
                io::ErrorKind::InvalidData,
                "capture.frame reply omitted SCM_RIGHTS sealed memfd",
            ))
        })?;

        let (out_w, out_h, dhash64, png_bytes) =
            read_memfd_hash_and_encode_png(memfd, width, height, byte_len, max_edge)?;

        Ok(KomaCapturedFrame {
            ws_id: ws_id.to_string(),
            epoch,
            width: out_w,
            height: out_h,
            has_protected_blackout,
            grid_cells,
            dhash64,
            png_bytes,
        })
    }

    /// Clicks a `grid_225` cell label (e.g. `"G8"`) under `expected_epoch`, returning new epoch.
    pub fn click_grid(
        &mut self,
        ws_id: &str,
        expected_epoch: u64,
        cell: &str,
        button: &str,
        clicks: u8,
    ) -> Result<u64, KomaIpcError> {
        let res = self.call_rpc(
            "vseat.click",
            json!({
                "ws_id": ws_id,
                "expected_epoch": expected_epoch,
                "auto_acquire": true,
                "grid_225": cell,
                "button": button,
                "clicks": clicks,
            }),
        )?;
        Ok(res
            .get("epoch")
            .and_then(Value::as_u64)
            .unwrap_or(expected_epoch))
    }

    /// Clicks explicit pixel coordinates `(x, y)` under `expected_epoch`, returning new epoch.
    pub fn click_xy(
        &mut self,
        ws_id: &str,
        expected_epoch: u64,
        x: f64,
        y: f64,
        button: &str,
        clicks: u8,
    ) -> Result<u64, KomaIpcError> {
        let res = self.call_rpc(
            "vseat.click",
            json!({
                "ws_id": ws_id,
                "expected_epoch": expected_epoch,
                "auto_acquire": true,
                "x": x,
                "y": y,
                "button": button,
                "clicks": clicks,
            }),
        )?;
        Ok(res
            .get("epoch")
            .and_then(Value::as_u64)
            .unwrap_or(expected_epoch))
    }

    /// Drags the virtual pointer between two window-local logical coordinates via `vseat.drag`.
    #[allow(clippy::too_many_arguments)]
    pub fn drag_xy(
        &mut self,
        ws_id: &str,
        expected_epoch: u64,
        from_x: f64,
        from_y: f64,
        to_x: f64,
        to_y: f64,
        button: &str,
        profile: &str,
        duration_ms: u64,
    ) -> Result<u64, KomaIpcError> {
        self.drag_anchors(
            ws_id,
            expected_epoch,
            json!({ "x": from_x, "y": from_y }),
            json!({ "x": to_x, "y": to_y }),
            button,
            profile,
            duration_ms,
        )
    }

    /// Drags the virtual pointer between two 15x15 `grid_225` cells via `vseat.drag`.
    #[allow(clippy::too_many_arguments)]
    pub fn drag_grid(
        &mut self,
        ws_id: &str,
        expected_epoch: u64,
        from_cell: &str,
        to_cell: &str,
        button: &str,
        profile: &str,
        duration_ms: u64,
    ) -> Result<u64, KomaIpcError> {
        self.drag_anchors(
            ws_id,
            expected_epoch,
            json!({ "grid_225": from_cell }),
            json!({ "grid_225": to_cell }),
            button,
            profile,
            duration_ms,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn drag_anchors(
        &mut self,
        ws_id: &str,
        expected_epoch: u64,
        from: Value,
        to: Value,
        button: &str,
        profile: &str,
        duration_ms: u64,
    ) -> Result<u64, KomaIpcError> {
        let res = self.call_rpc(
            "vseat.drag",
            json!({
                "ws_id": ws_id,
                "expected_epoch": expected_epoch,
                "from": from,
                "to": to,
                "button": button,
                "kinematics": {
                    "profile": profile,
                    "duration_ms": duration_ms,
                },
            }),
        )?;
        Ok(res
            .get("epoch")
            .and_then(Value::as_u64)
            .unwrap_or(expected_epoch))
    }

    /// Taps a key chord (e.g. `"Return"`, `"Ctrl+l"`) under `expected_epoch`.
    pub fn tap_key(
        &mut self,
        ws_id: &str,
        expected_epoch: u64,
        chord: &str,
    ) -> Result<u64, KomaIpcError> {
        let res = self.call_rpc(
            "vseat.key",
            json!({
                "ws_id": ws_id,
                "expected_epoch": expected_epoch,
                "chord": chord,
            }),
        )?;
        Ok(res
            .get("epoch")
            .and_then(Value::as_u64)
            .unwrap_or(expected_epoch))
    }

    /// Injects UTF-8 text via Koma's VirtualSeat (`zwp_text_input_v3` / auto) under `expected_epoch`.
    pub fn inject_text(
        &mut self,
        ws_id: &str,
        expected_epoch: u64,
        text: &str,
    ) -> Result<u64, KomaIpcError> {
        let res = self.call_rpc(
            "vseat.inject_text",
            json!({
                "ws_id": ws_id,
                "expected_epoch": expected_epoch,
                "text": text,
                "strategy": "auto",
            }),
        )?;
        Ok(res
            .get("epoch")
            .and_then(Value::as_u64)
            .unwrap_or(expected_epoch))
    }

    /// Moves the virtual pointer (hover without button press) along a smooth multi-waypoint spline via `vseat.path`.
    pub fn move_path(
        &mut self,
        ws_id: &str,
        expected_epoch: u64,
        waypoints: &[KomaPathWaypoint],
        spline: &str,
        profile: &str,
        duration_ms: u64,
    ) -> Result<u64, KomaIpcError> {
        self.send_path(
            ws_id,
            expected_epoch,
            waypoints,
            None,
            spline,
            profile,
            duration_ms,
        )
    }

    /// Drags the virtual pointer along a smooth multi-waypoint spline via `vseat.path`.
    #[allow(clippy::too_many_arguments)]
    pub fn drag_path(
        &mut self,
        ws_id: &str,
        expected_epoch: u64,
        waypoints: &[KomaPathWaypoint],
        button: &str,
        spline: &str,
        profile: &str,
        duration_ms: u64,
    ) -> Result<u64, KomaIpcError> {
        self.send_path(
            ws_id,
            expected_epoch,
            waypoints,
            Some(button),
            spline,
            profile,
            duration_ms,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn send_path(
        &mut self,
        ws_id: &str,
        expected_epoch: u64,
        waypoints: &[KomaPathWaypoint],
        button: Option<&str>,
        spline: &str,
        profile: &str,
        duration_ms: u64,
    ) -> Result<u64, KomaIpcError> {
        let waypoint_values: Vec<Value> = waypoints
            .iter()
            .map(KomaPathWaypoint::to_json_value)
            .collect();
        let (hold_start_ms, hold_end_ms) = if button.is_some() { (20, 20) } else { (0, 0) };
        let res = self.call_rpc(
            "vseat.path",
            json!({
                "ws_id": ws_id,
                "expected_epoch": expected_epoch,
                "auto_acquire": true,
                "waypoints": waypoint_values,
                "spline": spline,
                "button": button,
                "hold_start_ms": hold_start_ms,
                "hold_end_ms": hold_end_ms,
                "kinematics": {
                    "profile": profile,
                    "duration_ms": duration_ms,
                },
            }),
        )?;
        Ok(res
            .get("epoch")
            .and_then(Value::as_u64)
            .unwrap_or(expected_epoch))
    }

    /// Smoothly moves the virtual pointer to a single `KomaPathWaypoint` via `vseat.pointer_move`.
    pub fn pointer_move_waypoint(
        &mut self,
        ws_id: &str,
        expected_epoch: u64,
        target: &KomaPathWaypoint,
        profile: &str,
        duration_ms: u64,
    ) -> Result<u64, KomaIpcError> {
        let res = self.call_rpc(
            "vseat.pointer_move",
            json!({
                "ws_id": ws_id,
                "expected_epoch": expected_epoch,
                "auto_acquire": true,
                "target": target.to_json_value(),
                "kinematics": {
                    "profile": profile,
                    "duration_ms": duration_ms,
                },
            }),
        )?;
        Ok(res
            .get("epoch")
            .and_then(Value::as_u64)
            .unwrap_or(expected_epoch))
    }

    /// Settles (or immediately releases) the virtual agent cursor on `ws_id` via `vseat.cursor_settle`.
    pub fn settle_cursor(&mut self, ws_id: &str, immediate: bool) -> Result<(), KomaIpcError> {
        self.call_rpc(
            "vseat.cursor_settle",
            json!({
                "ws_id": ws_id,
                "immediate": immediate,
            }),
        )?;
        Ok(())
    }

    /// Releases this session's exclusive control lease on `ws_id` via `lease.release`.
    pub fn release_lease(&mut self, ws_id: &str, expected_epoch: u64) -> Result<u64, KomaIpcError> {
        let res = self.call_rpc(
            "lease.release",
            json!({
                "ws_id": ws_id,
                "expected_epoch": expected_epoch,
            }),
        )?;
        Ok(res
            .get("epoch")
            .and_then(Value::as_u64)
            .unwrap_or(expected_epoch))
    }

    /// Emits a relative pointer motion delta via `vseat.relative_move`.
    pub fn relative_move(
        &mut self,
        ws_id: &str,
        expected_epoch: u64,
        dx: f64,
        dy: f64,
        modifiers: &[String],
    ) -> Result<Value, KomaIpcError> {
        self.call_rpc(
            "vseat.relative_move",
            json!({
                "ws_id": ws_id,
                "expected_epoch": expected_epoch,
                "dx": dx,
                "dy": dy,
                "modifiers": modifiers,
            }),
        )
    }

    /// Dispatches smooth/discrete scroll wheel deltas via `vseat.wheel`.
    #[allow(clippy::too_many_arguments)]
    pub fn wheel(
        &mut self,
        ws_id: &str,
        expected_epoch: u64,
        target: Option<Value>,
        dx: f64,
        dy: f64,
        discrete_x: i32,
        discrete_y: i32,
        modifiers: &[String],
    ) -> Result<Value, KomaIpcError> {
        let mut params = serde_json::Map::new();
        params.insert("ws_id".to_string(), json!(ws_id));
        params.insert("expected_epoch".to_string(), json!(expected_epoch));
        if let Some(target_val) = target {
            params.insert("target".to_string(), target_val);
        }
        params.insert("dx".to_string(), json!(dx));
        params.insert("dy".to_string(), json!(dy));
        params.insert("discrete_x".to_string(), json!(discrete_x));
        params.insert("discrete_y".to_string(), json!(discrete_y));
        params.insert("modifiers".to_string(), json!(modifiers));
        self.call_rpc("vseat.wheel", Value::Object(params))
    }

    /// Probes modal occlusion and surface stability at `target` via `vseat.hit_test`.
    pub fn hit_test(
        &mut self,
        ws_id: &str,
        target: Value,
    ) -> Result<KomaHitTestResult, KomaIpcError> {
        let res = self.call_rpc(
            "vseat.hit_test",
            json!({
                "ws_id": ws_id,
                "target": target,
            }),
        )?;
        Ok(serde_json::from_value(res)?)
    }

    /// Sets or clears the compositor agent cursor status pill via `vseat.cursor_label`.
    pub fn cursor_label(
        &mut self,
        ws_id: &str,
        label: Option<&str>,
        caret_rect: Option<[f64; 4]>,
    ) -> Result<Value, KomaIpcError> {
        let mut params = serde_json::Map::new();
        params.insert("ws_id".to_string(), json!(ws_id));
        if let Some(lbl) = label {
            params.insert("label".to_string(), json!(lbl));
        }
        if let Some(rect) = caret_rect {
            params.insert("caret_rect".to_string(), json!(rect));
        }
        self.call_rpc("vseat.cursor_label", Value::Object(params))
    }

    #[cfg(test)]
    pub(crate) fn from_test_stream(stream: UnixStream) -> Self {
        Self {
            stream,
            next_id: 1,
            instance_dir: PathBuf::new(),
            session_id: "sess_test".to_string(),
            read_buf: Vec::new(),
            pending_fds: VecDeque::new(),
        }
    }
}

/// Discovers the running Koma instance socket directory under `$XDG_RUNTIME_DIR/koma/`
/// (or `/tmp/koma-runtime-<uid>/`).
pub fn discover_koma_instance_dir() -> Result<PathBuf, KomaIpcError> {
    let base_runtime = env::var_os("XDG_RUNTIME_DIR")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(format!(
                "/tmp/koma-runtime-{}",
                rustix::process::getuid().as_raw()
            ))
        });
    let koma_root = base_runtime.join("koma");

    if let Ok(sig) = env::var("KOMA_INSTANCE_SIGNATURE")
        && !sig.trim().is_empty()
    {
        let dir = koma_root.join(sig.trim());
        if dir.join(AGENT_SOCKET_NAME).exists() {
            return Ok(dir);
        }
    }

    if koma_root.is_dir() {
        let mut candidates: Vec<PathBuf> = std::fs::read_dir(&koma_root)?
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| p.join(AGENT_SOCKET_NAME).exists())
            .collect();
        candidates.sort();
        if let Some(dir) = candidates.pop() {
            return Ok(dir);
        }
    }

    Err(KomaIpcError::NotFound(format!(
        "no {} socket under {}",
        AGENT_SOCKET_NAME,
        koma_root.display()
    )))
}

fn mint_local_token(admin_sock: &Path, default_ws: &str) -> Result<String, KomaIpcError> {
    let stream = UnixStream::connect(admin_sock).map_err(|e| {
        KomaIpcError::Auth(format!(
            "cannot mint token via {}: {e}",
            admin_sock.display()
        ))
    })?;

    let mut grants = BTreeMap::new();
    grants.insert("ws_01".to_string(), "control");
    if !default_ws.is_empty() {
        grants.insert(default_ws.to_string(), "control");
    }

    let req = KomaRpcRequest {
        id: 1,
        method: "auth.mint_token".to_string(),
        params: json!({
            "label": "jikud-local",
            "ttl_secs": 28_800,
            "can_create_headless": true,
            "max_owned_workspaces": 4,
            "workspace_grants": grants,
        }),
    };
    let (resp, _): (KomaRpcResponse, _) = sync_roundtrip(&stream, &req, Vec::new())?;
    if let Some(err) = resp.error {
        return Err(KomaIpcError::Rpc(err));
    }
    resp.result
        .as_ref()
        .and_then(|v| v.get("token"))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .ok_or_else(|| KomaIpcError::Auth("auth.mint_token omitted token".to_string()))
}

fn read_memfd_hash_and_encode_png(
    fd: OwnedFd,
    width: u32,
    height: u32,
    byte_len: usize,
    max_edge: u32,
) -> io::Result<(u32, u32, u64, Vec<u8>)> {
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|px| px.checked_mul(4))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "frame dimensions overflow"))?;
    if width == 0 || height == 0 || byte_len < expected || expected > isize::MAX as usize {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("memfd byte_len ({byte_len}) incompatible with {width}x{height} RGBA8"),
        ));
    }

    let mut f = File::from(fd);
    f.seek(SeekFrom::Start(0))?;
    let mut buf = vec![0_u8; expected];
    f.read_exact(&mut buf)?;
    hash_scale_and_encode_png(&buf, width, height, max_edge)
}

fn hash_scale_and_encode_png(
    rgba: &[u8],
    width: u32,
    height: u32,
    max_edge: u32,
) -> io::Result<(u32, u32, u64, Vec<u8>)> {
    let dhash64 = compute_rgba_dhash64(rgba, width, height)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;

    let (out_w, out_h, scaled_buf);
    let pixels: &[u8] = if max_edge > 0 && (width > max_edge || height > max_edge) {
        let scale = f64::from(max_edge) / f64::from(width.max(height));
        out_w = ((f64::from(width) * scale).round() as u32).max(1);
        out_h = ((f64::from(height) * scale).round() as u32).max(1);
        scaled_buf = downscale_rgba8(rgba, width, height, out_w, out_h);
        &scaled_buf
    } else {
        out_w = width;
        out_h = height;
        rgba
    };

    let mut png_bytes = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut png_bytes, out_w, out_h);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder
            .write_header()
            .map_err(|e| io::Error::other(format!("png header error: {e}")))?;
        writer
            .write_image_data(pixels)
            .map_err(|e| io::Error::other(format!("png encode error: {e}")))?;
    }

    Ok((out_w, out_h, dhash64, png_bytes))
}

fn downscale_rgba8(src: &[u8], src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> Vec<u8> {
    let mut out = vec![0_u8; (dst_w as usize) * (dst_h as usize) * 4];
    for dy in 0..dst_h {
        let sy = ((u64::from(dy) * u64::from(src_h)) / u64::from(dst_h)) as usize;
        for dx in 0..dst_w {
            let sx = ((u64::from(dx) * u64::from(src_w)) / u64::from(dst_w)) as usize;
            let s_off = (sy * (src_w as usize) + sx) * 4;
            let d_off = ((dy as usize) * (dst_w as usize) + (dx as usize)) * 4;
            out[d_off..d_off + 4].copy_from_slice(&src[s_off..s_off + 4]);
        }
    }
    out
}

/// Synchronously sends a length-prefixed JSON frame and reads one response frame
/// along with any attached `SCM_RIGHTS` file descriptors.
fn send_frame_with_fds(
    stream: &UnixStream,
    payload: &[u8],
    req_fds: Vec<OwnedFd>,
) -> io::Result<()> {
    if payload.is_empty() || payload.len() > MAX_FRAME_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "IPC request frame length out of bounds",
        ));
    }

    let mut frame = Vec::with_capacity(4 + payload.len());
    frame.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    frame.extend_from_slice(payload);

    let mut written = 0;
    let mut fds_opt = Some(req_fds);
    while written < frame.len() {
        let chunk = &frame[written..];
        let sent = if written == 0
            && let Some(fds) = fds_opt.take()
            && !fds.is_empty()
        {
            let borrowed: Vec<BorrowedFd<'_>> = fds.iter().map(|f| f.as_fd()).collect();
            send_with_fds(stream, chunk, &borrowed)?
        } else {
            send_with_fds(stream, chunk, &[])?
        };
        if sent == 0 {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "peer closed Unix socket during write",
            ));
        }
        written += sent;
    }
    Ok(())
}

fn read_next_raw_frame(
    stream: &UnixStream,
    buf: &mut Vec<u8>,
    pending_fds: &mut VecDeque<OwnedFd>,
) -> io::Result<Vec<u8>> {
    let mut scratch = [0_u8; 16 * 1024];
    loop {
        if buf.len() >= 4 {
            let body_len = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
            if body_len == 0 || body_len > MAX_FRAME_BYTES {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "invalid Koma IPC response frame length",
                ));
            }
            if buf.len() >= 4 + body_len {
                let body = buf[4..4 + body_len].to_vec();
                buf.drain(..4 + body_len);
                return Ok(body);
            }
        }
        let (n, fds) = recv_with_fds(stream, &mut scratch)?;
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "compositor closed IPC socket before reply",
            ));
        }
        buf.extend_from_slice(&scratch[..n]);
        pending_fds.extend(fds);
    }
}

pub fn sync_roundtrip<Req: Serialize, Resp: for<'de> Deserialize<'de>>(
    stream: &UnixStream,
    req: &Req,
    req_fds: Vec<OwnedFd>,
) -> io::Result<(Resp, Vec<OwnedFd>)> {
    let payload = serde_json::to_vec(req)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e.to_string()))?;
    let expected_id = serde_json::to_value(req)
        .ok()
        .and_then(|v| v.get("id").and_then(Value::as_u64));
    send_frame_with_fds(stream, &payload, req_fds)?;

    let mut buf = Vec::new();
    let mut pending_fds = VecDeque::new();
    loop {
        let body = read_next_raw_frame(stream, &mut buf, &mut pending_fds)?;
        if let Some(req_id) = expected_id {
            let val: Value = serde_json::from_slice(&body)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
            if val.get("id").and_then(Value::as_u64) != Some(req_id) {
                continue;
            }
        }
        let parsed: Resp = serde_json::from_slice(&body)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        let fds: Vec<OwnedFd> = pending_fds.drain(..).collect();
        return Ok((parsed, fds));
    }
}

fn send_with_fds(stream: &UnixStream, bytes: &[u8], fds: &[BorrowedFd<'_>]) -> io::Result<usize> {
    if fds.len() > MAX_SCM_FDS {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "too many SCM_RIGHTS FDs",
        ));
    }
    let iov = [IoSlice::new(bytes)];
    let res = if fds.is_empty() {
        let mut empty = SendAncillaryBuffer::default();
        sendmsg(stream, &iov, &mut empty, SendFlags::NOSIGNAL)
    } else {
        let mut space = [MaybeUninit::uninit(); cmsg_space!(ScmRights(MAX_SCM_FDS))];
        let mut cmsg_buf = SendAncillaryBuffer::new(&mut space);
        if !cmsg_buf.push(SendAncillaryMessage::ScmRights(fds)) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "insufficient ancillary cmsg space",
            ));
        }
        sendmsg(stream, &iov, &mut cmsg_buf, SendFlags::NOSIGNAL)
    };
    res.map_err(errno_to_io)
}

fn recv_with_fds(stream: &UnixStream, buf: &mut [u8]) -> io::Result<(usize, Vec<OwnedFd>)> {
    let mut iov = [IoSliceMut::new(buf)];
    let mut space = [MaybeUninit::uninit(); cmsg_space!(ScmRights(MAX_SCM_FDS))];
    let mut cmsg_buf = RecvAncillaryBuffer::new(&mut space);
    let out =
        recvmsg(stream, &mut iov, &mut cmsg_buf, RecvFlags::CMSG_CLOEXEC).map_err(errno_to_io)?;
    let mut fds = Vec::new();
    for msg in cmsg_buf.drain() {
        if let RecvAncillaryMessage::ScmRights(list) = msg {
            fds.extend(list);
        }
    }
    Ok((out.bytes, fds))
}

fn errno_to_io(err: Errno) -> io::Error {
    if err == Errno::AGAIN || err == Errno::WOULDBLOCK {
        io::Error::from(io::ErrorKind::WouldBlock)
    } else {
        io::Error::from_raw_os_error(err.raw_os_error())
    }
}

/// Inspects PNG width/height header chunks if valid, returning `(width, height)`.
#[must_use]
pub fn sniff_png_dimensions(png_bytes: &[u8]) -> Option<(u32, u32)> {
    let decoder = png::Decoder::new(std::io::Cursor::new(png_bytes));
    let reader = decoder.read_info().ok()?;
    let info = reader.info();
    Some((info.width, info.height))
}

#[cfg(test)]
pub(crate) fn read_test_rpc_request(
    stream: &UnixStream,
    buf: &mut Vec<u8>,
) -> io::Result<KomaRpcRequest> {
    let mut pending_fds = VecDeque::new();
    let body = read_next_raw_frame(stream, buf, &mut pending_fds)?;
    serde_json::from_slice(&body)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
}

#[cfg(test)]
pub(crate) fn write_test_rpc_response(
    stream: &UnixStream,
    resp: &KomaRpcResponse,
    fds: Vec<OwnedFd>,
) -> io::Result<()> {
    let body = serde_json::to_vec(resp)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e.to_string()))?;
    send_frame_with_fds(stream, &body, fds)
}

#[cfg(test)]
pub(crate) fn create_sealed_test_rgba_memfd(width: u32, height: u32) -> (OwnedFd, usize) {
    use rustix::fs::{MemfdFlags, SealFlags, fcntl_add_seals, memfd_create};
    use std::io::Write as _;

    let mut rgba = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            rgba.extend_from_slice(&[((x * 60) % 256) as u8, ((y * 60) % 256) as u8, 128, 255]);
        }
    }
    let cname = std::ffi::CString::new("jiku_test_memfd").unwrap();
    let fd = memfd_create(&cname, MemfdFlags::CLOEXEC | MemfdFlags::ALLOW_SEALING).unwrap();
    let mut file = File::from(fd);
    file.write_all(&rgba).unwrap();
    file.seek(SeekFrom::Start(0)).unwrap();
    let memfd = OwnedFd::from(file);
    fcntl_add_seals(
        &memfd,
        SealFlags::SHRINK | SealFlags::GROW | SealFlags::WRITE | SealFlags::SEAL,
    )
    .unwrap();
    (memfd, rgba.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustix::fs::{MemfdFlags, SealFlags, fcntl_add_seals, memfd_create};
    use std::io::Write as _;

    #[test]
    fn roundtrip_and_sealed_memfd_png_encode_work_over_unix_pair() {
        let (server, client) = UnixStream::pair().unwrap();

        let width = 4_u32;
        let height = 4_u32;
        let mut rgba = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height {
            for x in 0..width {
                rgba.extend_from_slice(&[(x * 60) as u8, (y * 60) as u8, 128, 255]);
            }
        }
        let cname = std::ffi::CString::new("jiku_test_memfd").unwrap();
        let fd = memfd_create(&cname, MemfdFlags::CLOEXEC | MemfdFlags::ALLOW_SEALING).unwrap();
        let mut file = File::from(fd);
        file.write_all(&rgba).unwrap();
        file.seek(SeekFrom::Start(0)).unwrap();
        let memfd = OwnedFd::from(file);
        fcntl_add_seals(
            &memfd,
            SealFlags::SHRINK | SealFlags::GROW | SealFlags::WRITE | SealFlags::SEAL,
        )
        .unwrap();

        let srv_handle = std::thread::spawn(move || {
            let mut req_buf = [0_u8; 512];
            let (n, _) = recv_with_fds(&server, &mut req_buf).unwrap();
            let req: KomaRpcRequest = serde_json::from_slice(&req_buf[4..n]).unwrap();
            assert_eq!(req.method, "capture.frame");

            let resp = KomaRpcResponse {
                id: req.id,
                result: Some(json!({
                    "epoch": 7,
                    "width": width,
                    "height": height,
                    "byte_len": rgba.len(),
                    "has_protected_blackout": false,
                    "grid_cells": 225,
                })),
                error: None,
            };
            let body = serde_json::to_vec(&resp).unwrap();
            let mut frame = (body.len() as u32).to_le_bytes().to_vec();
            frame.extend_from_slice(&body);
            send_with_fds(&server, &frame, &[memfd.as_fd()]).unwrap();
        });

        let req = KomaRpcRequest {
            id: 1,
            method: "capture.frame".to_string(),
            params: json!({ "ws_id": "ws_01" }),
        };
        let (resp, mut fds): (KomaRpcResponse, _) =
            sync_roundtrip(&client, &req, Vec::new()).unwrap();
        let fd = fds.pop().unwrap();
        let res = resp.result.unwrap();
        let (w, h, dhash, png) = read_memfd_hash_and_encode_png(
            fd,
            res["width"].as_u64().unwrap() as u32,
            res["height"].as_u64().unwrap() as u32,
            res["byte_len"].as_u64().unwrap() as usize,
            1920,
        )
        .unwrap();

        assert_eq!((w, h), (4, 4));
        assert_ne!(dhash, 0);
        assert_eq!(sniff_png_dimensions(&png), Some((4, 4)));
        srv_handle.join().unwrap();
    }

    #[test]
    fn drag_xy_and_drag_grid_send_expected_vseat_drag_payloads() {
        let (server, client) = UnixStream::pair().unwrap();
        let mut ipc = KomaIpcClient {
            stream: client,
            next_id: 10,
            instance_dir: PathBuf::new(),
            session_id: "sess_test".to_string(),
            read_buf: Vec::new(),
            pending_fds: VecDeque::new(),
        };

        let srv_handle = std::thread::spawn(move || {
            let mut captured = Vec::new();
            for expected_epoch in [5_u64, 6_u64] {
                let mut req_buf = [0_u8; 2048];
                let (n, _) = recv_with_fds(&server, &mut req_buf).unwrap();
                let req: KomaRpcRequest = serde_json::from_slice(&req_buf[4..n]).unwrap();
                captured.push(req.clone());
                let resp = KomaRpcResponse {
                    id: req.id,
                    result: Some(json!({ "ok": true, "epoch": expected_epoch + 1 })),
                    error: None,
                };
                let body = serde_json::to_vec(&resp).unwrap();
                let mut frame = (body.len() as u32).to_le_bytes().to_vec();
                frame.extend_from_slice(&body);
                send_with_fds(&server, &frame, &[]).unwrap();
            }
            captured
        });

        let epoch1 = ipc
            .drag_xy(
                "ws_01", 5, 100.0, 200.0, 350.0, 420.0, "left", "spring", 150,
            )
            .unwrap();
        let epoch2 = ipc
            .drag_grid("ws_01", epoch1, "C4", "M8", "left", "bezier", 120)
            .unwrap();
        assert_eq!(epoch1, 6);
        assert_eq!(epoch2, 7);

        let reqs = srv_handle.join().unwrap();
        assert_eq!(reqs[0].method, "vseat.drag");
        assert_eq!(reqs[0].params["from"], json!({ "x": 100.0, "y": 200.0 }));
        assert_eq!(reqs[0].params["to"], json!({ "x": 350.0, "y": 420.0 }));
        assert_eq!(reqs[0].params["button"], "left");
        assert_eq!(
            reqs[0].params["kinematics"],
            json!({ "profile": "spring", "duration_ms": 150 })
        );

        assert_eq!(reqs[1].method, "vseat.drag");
        assert_eq!(reqs[1].params["from"], json!({ "grid_225": "C4" }));
        assert_eq!(reqs[1].params["to"], json!({ "grid_225": "M8" }));
        assert_eq!(
            reqs[1].params["kinematics"],
            json!({ "profile": "bezier", "duration_ms": 120 })
        );
    }

    #[test]
    fn roundtrip_skips_coalesced_unsolicited_event_frames() {
        let (server, client) = UnixStream::pair().unwrap();
        let mut ipc = KomaIpcClient {
            stream: client,
            next_id: 1,
            instance_dir: PathBuf::new(),
            session_id: "sess_test".to_string(),
            read_buf: Vec::new(),
            pending_fds: VecDeque::new(),
        };

        let srv_handle = std::thread::spawn(move || {
            let mut req_buf = [0_u8; 2048];
            let (n, _) = recv_with_fds(&server, &mut req_buf).unwrap();
            let req1: KomaRpcRequest = serde_json::from_slice(&req_buf[4..n]).unwrap();
            assert_eq!(req1.id, 1);

            let event_body = serde_json::to_vec(&json!({
                "topic": "workspace_lease",
                "ws_id": "ws_01",
                "epoch": 2,
                "event": { "EpochChanged": { "new_epoch": 2 } }
            }))
            .unwrap();
            let resp1_body = serde_json::to_vec(&KomaRpcResponse {
                id: req1.id,
                result: Some(json!({ "ok": true, "epoch": 2 })),
                error: None,
            })
            .unwrap();
            // Coalesce an unsolicited event + RPC #1 response + a trailing event into ONE write.
            let mut combined = Vec::new();
            for body in [&event_body, &resp1_body, &event_body] {
                combined.extend_from_slice(&(body.len() as u32).to_le_bytes());
                combined.extend_from_slice(body);
            }
            send_with_fds(&server, &combined, &[]).unwrap();

            // Next RPC must still succeed after skipping the buffered trailing event.
            let (n2, _) = recv_with_fds(&server, &mut req_buf).unwrap();
            let req2: KomaRpcRequest = serde_json::from_slice(&req_buf[4..n2]).unwrap();
            assert_eq!(req2.id, 2);
            let resp2_body = serde_json::to_vec(&KomaRpcResponse {
                id: req2.id,
                result: Some(json!({ "ok": true, "epoch": 3 })),
                error: None,
            })
            .unwrap();
            let mut f2 = (resp2_body.len() as u32).to_le_bytes().to_vec();
            f2.extend_from_slice(&resp2_body);
            send_with_fds(&server, &f2, &[]).unwrap();
        });

        let r1 = ipc
            .drag_xy("ws_01", 1, 10.0, 20.0, 30.0, 40.0, "left", "spring", 120)
            .unwrap();
        let r2 = ipc.tap_key("ws_01", r1, "Return").unwrap();
        assert_eq!(r1, 2);
        assert_eq!(r2, 3);
        srv_handle.join().unwrap();
    }

    #[test]
    fn move_path_drag_path_and_settle_cursor_send_expected_wire_payloads() {
        let (server, client) = UnixStream::pair().unwrap();
        let mut ipc = KomaIpcClient::from_test_stream(client);

        let srv_handle = std::thread::spawn(move || {
            let mut captured = Vec::new();
            let mut buf = Vec::new();
            let replies = [
                json!({ "ok": true, "epoch": 4, "waypoints": 3 }),
                json!({ "ok": true, "epoch": 5, "waypoints": 2 }),
                json!({ "ok": true }),
                json!({ "ok": true }),
            ];
            for result in replies {
                let req = read_test_rpc_request(&server, &mut buf).unwrap();
                captured.push(req.clone());
                let resp = KomaRpcResponse {
                    id: req.id,
                    result: Some(result),
                    error: None,
                };
                write_test_rpc_response(&server, &resp, Vec::new()).unwrap();
            }
            captured
        });

        let move_waypoints = vec![
            KomaPathWaypoint::Xy { x: 120.0, y: 240.0 },
            KomaPathWaypoint::Grid225("H8".to_string()),
            KomaPathWaypoint::Xy { x: 360.0, y: 240.0 },
        ];
        let epoch_after_move = ipc
            .move_path("ws_01", 3, &move_waypoints, "catmull_rom", "spring", 260)
            .unwrap();
        assert_eq!(epoch_after_move, 4);

        let drag_waypoints = vec![
            KomaPathWaypoint::Grid225("C4".to_string()),
            KomaPathWaypoint::Grid225("M9".to_string()),
        ];
        let epoch_after_drag = ipc
            .drag_path(
                "ws_01",
                epoch_after_move,
                &drag_waypoints,
                "left",
                "bezier",
                "minimum_jerk",
                320,
            )
            .unwrap();
        assert_eq!(epoch_after_drag, 5);

        ipc.settle_cursor("ws_01", false).unwrap();
        ipc.settle_cursor("ws_01", true).unwrap();

        let reqs = srv_handle.join().unwrap();
        assert_eq!(reqs[0].method, "vseat.path");
        assert_eq!(
            reqs[0].params,
            json!({
                "ws_id": "ws_01",
                "expected_epoch": 3,
                "auto_acquire": true,
                "waypoints": [
                    { "x": 120.0, "y": 240.0 },
                    { "grid_225": "H8" },
                    { "x": 360.0, "y": 240.0 }
                ],
                "spline": "catmull_rom",
                "button": null,
                "hold_start_ms": 0,
                "hold_end_ms": 0,
                "kinematics": {
                    "profile": "spring",
                    "duration_ms": 260
                }
            })
        );

        assert_eq!(reqs[1].method, "vseat.path");
        assert_eq!(
            reqs[1].params,
            json!({
                "ws_id": "ws_01",
                "expected_epoch": 4,
                "auto_acquire": true,
                "waypoints": [
                    { "grid_225": "C4" },
                    { "grid_225": "M9" }
                ],
                "spline": "bezier",
                "button": "left",
                "hold_start_ms": 20,
                "hold_end_ms": 20,
                "kinematics": {
                    "profile": "minimum_jerk",
                    "duration_ms": 320
                }
            })
        );

        assert_eq!(reqs[2].method, "vseat.cursor_settle");
        assert_eq!(
            reqs[2].params,
            json!({ "ws_id": "ws_01", "immediate": false })
        );

        assert_eq!(reqs[3].method, "vseat.cursor_settle");
        assert_eq!(
            reqs[3].params,
            json!({ "ws_id": "ws_01", "immediate": true })
        );
    }

    #[test]
    fn pointer_move_waypoint_and_release_lease_send_expected_wire_payloads() {
        let (server, client) = UnixStream::pair().unwrap();
        let mut ipc = KomaIpcClient::from_test_stream(client);

        let srv_handle = std::thread::spawn(move || {
            let mut captured = Vec::new();
            let mut buf = Vec::new();
            let replies = [
                json!({ "ok": true, "epoch": 11 }),
                json!({ "ok": true, "epoch": 12 }),
                json!({ "epoch": 13 }),
            ];
            for result in replies {
                let req = read_test_rpc_request(&server, &mut buf).unwrap();
                captured.push(req.clone());
                let resp = KomaRpcResponse {
                    id: req.id,
                    result: Some(result),
                    error: None,
                };
                write_test_rpc_response(&server, &resp, Vec::new()).unwrap();
            }
            captured
        });

        let ep1 = ipc
            .pointer_move_waypoint(
                "ws_01",
                10,
                &KomaPathWaypoint::Grid225("A1".to_string()),
                "bezier",
                140,
            )
            .unwrap();
        assert_eq!(ep1, 11);

        let ep2 = ipc
            .pointer_move_waypoint(
                "ws_01",
                ep1,
                &KomaPathWaypoint::Xy { x: 420.0, y: 310.0 },
                "spring",
                180,
            )
            .unwrap();
        assert_eq!(ep2, 12);

        let ep3 = ipc.release_lease("ws_01", ep2).unwrap();
        assert_eq!(ep3, 13);

        let reqs = srv_handle.join().unwrap();
        assert_eq!(reqs[0].method, "vseat.pointer_move");
        assert_eq!(
            reqs[0].params,
            json!({
                "ws_id": "ws_01",
                "expected_epoch": 10,
                "auto_acquire": true,
                "target": { "grid_225": "A1" },
                "kinematics": { "profile": "bezier", "duration_ms": 140 }
            })
        );
        assert_eq!(reqs[1].method, "vseat.pointer_move");
        assert_eq!(
            reqs[1].params,
            json!({
                "ws_id": "ws_01",
                "expected_epoch": 11,
                "auto_acquire": true,
                "target": { "x": 420.0, "y": 310.0 },
                "kinematics": { "profile": "spring", "duration_ms": 180 }
            })
        );
        assert_eq!(reqs[2].method, "lease.release");
        assert_eq!(
            reqs[2].params,
            json!({
                "ws_id": "ws_01",
                "expected_epoch": 12
            })
        );
    }

    #[test]
    fn koma_ipc_supports_relative_move_wheel_hit_test_cursor_label_and_disposition() {
        let (client, server) = UnixStream::pair().unwrap();
        let mut ipc = KomaIpcClient::from_test_stream(client);

        let srv_handle = std::thread::spawn(move || {
            let mut buf = Vec::new();
            let mut captured = Vec::new();
            for step in 0..5 {
                let req = read_test_rpc_request(&server, &mut buf).unwrap();
                let resp = match req.method.as_str() {
                    "vseat.relative_move" => KomaRpcResponse {
                        id: req.id,
                        result: Some(json!({ "epoch": 21 })),
                        error: None,
                    },
                    "vseat.wheel" => KomaRpcResponse {
                        id: req.id,
                        result: Some(json!({ "epoch": 22 })),
                        error: None,
                    },
                    "vseat.hit_test" => KomaRpcResponse {
                        id: req.id,
                        result: Some(json!({
                            "reaches_target": false,
                            "target_win_id": "win_1_g1",
                            "target_app_id": "virtual-paint",
                            "blocking_win_id": "win_2_g1",
                            "blocking_app_id": "dialog-app",
                            "blocking_title": "Confirm Save",
                            "blocking_rect": [100, 100, 200, 120],
                            "geometry_stable": true,
                            "sampled_points": [[220.0, 180.0]],
                            "cursor_pos": [220.0, 180.0]
                        })),
                        error: None,
                    },
                    "vseat.cursor_label" => KomaRpcResponse {
                        id: req.id,
                        result: Some(json!({ "ok": true })),
                        error: None,
                    },
                    "vseat.click" => KomaRpcResponse {
                        id: req.id,
                        result: None,
                        error: Some(KomaRpcError {
                            code: ERR_CODE_STALE_EPOCH,
                            symbol: Some("ERR_STALE_EPOCH".to_string()),
                            message: format!("stale epoch at step {step}"),
                            current_epoch: Some(25),
                            retryable: Some(true),
                            disposition: Some(DeliverySemantics {
                                delivery: DeliveryDisposition::NotDelivered,
                                retry: RetryDisposition::Safe,
                            }),
                        }),
                    },
                    other => panic!("unexpected RPC method: {other}"),
                };
                captured.push(req);
                write_test_rpc_response(&server, &resp, Vec::new()).unwrap();
            }
            captured
        });

        let r1 = ipc
            .relative_move("ws_01", 20, -14.5, 8.0, &["shift".to_string()])
            .unwrap();
        assert_eq!(r1["epoch"], 21);

        let r2 = ipc
            .wheel(
                "ws_01",
                21,
                Some(json!({"x": 300.0, "y": 240.0})),
                0.0,
                -15.0,
                0,
                -1,
                &[],
            )
            .unwrap();
        assert_eq!(r2["epoch"], 22);

        let ht = ipc
            .hit_test("ws_01", json!({"x": 220.0, "y": 180.0}))
            .unwrap();
        assert!(!ht.reaches_target);
        assert_eq!(ht.blocking_app_id.as_deref(), Some("dialog-app"));
        assert_eq!(ht.blocking_rect, Some([100, 100, 200, 120]));
        assert!(ht.geometry_stable);

        let cl = ipc
            .cursor_label("ws_01", Some("Painting"), Some([10.0, 20.0, 2.0, 16.0]))
            .unwrap();
        assert_eq!(cl["ok"], true);

        let err = ipc
            .call_rpc("vseat.click", json!({"ws_id": "ws_01"}))
            .unwrap_err();
        assert_eq!(
            err.disposition(),
            Some(DeliverySemantics {
                delivery: DeliveryDisposition::NotDelivered,
                retry: RetryDisposition::Safe,
            })
        );

        let reqs = srv_handle.join().unwrap();
        assert_eq!(reqs.len(), 5);
        assert_eq!(reqs[0].method, "vseat.relative_move");
        assert_eq!(reqs[1].method, "vseat.wheel");
        assert_eq!(reqs[2].method, "vseat.hit_test");
        assert_eq!(reqs[3].method, "vseat.cursor_label");
    }
}
