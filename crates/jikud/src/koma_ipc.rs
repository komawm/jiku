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

/// Authenticated client connection to Koma's `koma-agent.sock`.
pub struct KomaIpcClient {
    stream: UnixStream,
    next_id: u64,
    instance_dir: PathBuf,
    session_id: String,
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

    /// Sends a generic JSON RPC over `koma-agent.sock` and returns the result JSON value.
    pub fn call_rpc(&mut self, method: &str, params: Value) -> Result<Value, KomaIpcError> {
        let req = KomaRpcRequest {
            id: self.alloc_id(),
            method: method.to_string(),
            params,
        };
        let (resp, _): (KomaRpcResponse, _) = sync_roundtrip(&self.stream, &req, Vec::new())?;
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

            let active_window =
                ws.get("active_window")
                    .filter(|v| !v.is_null())
                    .map(|w| KomaWindowSummary {
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
        let (resp, mut fds): (KomaRpcResponse, _) = sync_roundtrip(&self.stream, &req, Vec::new())?;
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
pub fn sync_roundtrip<Req: Serialize, Resp: for<'de> Deserialize<'de>>(
    stream: &UnixStream,
    req: &Req,
    req_fds: Vec<OwnedFd>,
) -> io::Result<(Resp, Vec<OwnedFd>)> {
    let payload = serde_json::to_vec(req)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e.to_string()))?;
    if payload.is_empty() || payload.len() > MAX_FRAME_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "IPC request frame length out of bounds",
        ));
    }

    let mut frame = Vec::with_capacity(4 + payload.len());
    frame.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    frame.extend_from_slice(&payload);

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

    let mut buf = Vec::new();
    let mut scratch = [0_u8; 16 * 1024];
    let mut pending_fds = VecDeque::new();
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
                let parsed: Resp = serde_json::from_slice(&buf[4..4 + body_len])
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
                let fds: Vec<OwnedFd> = pending_fds.drain(..).collect();
                return Ok((parsed, fds));
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
}
