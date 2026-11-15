//! Local HTTP + RFC 6455 WebSocket Companion Avatar stage server (`jiku::avatar_stage`).
//!
//! Serves a zero-external-asset transparent HTML5/Canvas2D 2.5D companion stage
//! (suitable for OBS Browser Source or transparent Wayland/Chromium overlays)
//! alongside `/health`, optional `/avatar.vrm` GLB binary hosting, and `/ws`
//! live streaming of [`AvatarStageMessage`] cues, lip-sync envelopes, `gaze_uv`,
//! and barge-in resets.

use std::{
    future::Future,
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use anyhow::{Context as _, anyhow};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64_STD};
use jiku_tts::{AudioEnvelopeFrame, AvatarCue, AvatarEmotion};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::{
    io::{AsyncRead, AsyncReadExt as _, AsyncWrite, AsyncWriteExt as _},
    net::{TcpListener, TcpStream},
    sync::{RwLock, broadcast, mpsc, oneshot},
    task::JoinHandle,
};

/// Default bind address for the companion avatar stage server.
pub const DEFAULT_AVATAR_STAGE_ADDR: &str = "127.0.0.1:47200";

const WS_GUID: &str = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
const MAX_HTTP_HEAD_BYTES: usize = 16 * 1024;
const MAX_WS_INCOMING_FRAME_BYTES: usize = 1024 * 1024;
const MAX_VRM_FILE_BYTES: u64 = 128 * 1024 * 1024;
const HTTP_READ_TIMEOUT: Duration = Duration::from_secs(5);

/// Server configuration for [`serve_avatar_stage`] and [`spawn_avatar_stage`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AvatarStageConfig {
    /// TCP socket address to bind (e.g. `"127.0.0.1:47200"` or `"127.0.0.1:0"`).
    pub bind_addr: String,
    /// Optional path to a `.vrm` binary model exposed at `GET /avatar.vrm`.
    pub vrm_path: Option<PathBuf>,
    /// Display name shown on the stage badge and `/health` JSON.
    pub companion_name: String,
}

impl Default for AvatarStageConfig {
    fn default() -> Self {
        Self {
            bind_addr: DEFAULT_AVATAR_STAGE_ADDR.to_string(),
            vrm_path: None,
            companion_name: "Jiku".to_string(),
        }
    }
}

/// Complete stage state snapshot sent to new `/ws` clients and kept synced across updates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AvatarStageSnapshot {
    /// Companion display name.
    pub companion_name: String,
    /// Latest [`AvatarCue`] (emotion, gaze, mora/PCM envelopes, interruption).
    pub cue: AvatarCue,
    /// Current subtitle/caption text shown on the overlay.
    pub caption: Option<String>,
    /// Whether active speech output is currently playing.
    pub speaking: bool,
    /// Whether a `.vrm` asset is configured at `/avatar.vrm`.
    pub vrm_available: bool,
}

impl Default for AvatarStageSnapshot {
    fn default() -> Self {
        Self {
            companion_name: "Jiku".to_string(),
            cue: AvatarCue::default(),
            caption: None,
            speaking: false,
            vrm_available: false,
        }
    }
}

impl AvatarStageSnapshot {
    /// Applies a broadcast [`AvatarStageMessage`] so late-joining WebSocket clients
    /// receive an up-to-date snapshot.
    pub fn apply_message(&mut self, msg: &AvatarStageMessage) {
        match msg {
            AvatarStageMessage::Snapshot(snapshot) => {
                *self = snapshot.clone();
            }
            AvatarStageMessage::Cue {
                cue,
                caption,
                speaking,
            } => {
                self.cue = cue.clone();
                self.speaking = *speaking;
                if let Some(text) = caption {
                    self.caption = (!text.is_empty()).then(|| text.clone());
                }
            }
            AvatarStageMessage::EnvelopeFrame { frame, .. } => {
                self.cue.pcm_envelope = Some(*frame);
                self.speaking = frame.rms >= 0.015;
            }
            AvatarStageMessage::Gaze { u, v } => {
                self.cue.gaze_uv = Some((*u, *v));
            }
            AvatarStageMessage::BargeInReset { emotion } => {
                self.cue.emotion = *emotion;
                self.cue.interrupted = true;
                self.cue.mora_envelope.clear();
                self.cue.pcm_envelope = None;
                self.speaking = false;
            }
            AvatarStageMessage::Caption { text } => {
                self.caption = text.clone();
            }
        }
    }
}

/// Real-time JSON message pushed to `/ws` companion avatar subscribers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AvatarStageMessage {
    /// Full state snapshot sent upon connect, lag recovery, or `snapshot_request`.
    Snapshot(AvatarStageSnapshot),
    /// Structured emotion/gaze/mora cue update.
    Cue {
        /// Companion cue payload.
        cue: AvatarCue,
        /// Optional live caption associated with this cue.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        caption: Option<String>,
        /// Whether speech playback is active.
        #[serde(default)]
        speaking: bool,
    },
    /// High-rate streaming PCM amplitude & spectral band frame for live lip-sync.
    EnvelopeFrame {
        /// Instantaneous amplitude/spectral bands.
        frame: AudioEnvelopeFrame,
        /// Optional estimated vowel hint (`"a"`, `"i"`, `"u"`, `"e"`, `"o"`, `"sil"`).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        vowel: Option<String>,
    },
    /// Gaze-target update tracking Jiku's virtual cursor (`vSeat`) on `[0.0, 1.0]^2`.
    Gaze {
        /// Normalized horizontal coordinate (`0.0` left .. `1.0` right).
        u: f32,
        /// Normalized vertical coordinate (`0.0` top .. `1.0` bottom).
        v: f32,
    },
    /// Immediate barge-in interruption clearing active lip-sync state.
    BargeInReset {
        /// Emotion to settle into after interruption.
        #[serde(default)]
        emotion: AvatarEmotion,
    },
    /// Live subtitle/caption banner update.
    Caption {
        /// Caption text, or `None` to clear.
        text: Option<String>,
    },
}

/// Running companion avatar stage server handle returned by [`spawn_avatar_stage`].
pub struct AvatarStageHandle {
    local_addr: SocketAddr,
    tx: broadcast::Sender<AvatarStageMessage>,
    shutdown_tx: Option<oneshot::Sender<()>>,
    join_handle: Option<JoinHandle<anyhow::Result<()>>>,
}

impl AvatarStageHandle {
    /// Returns the actual bound local socket address.
    #[must_use]
    pub const fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    /// Returns the HTTP root URL (`http://<addr>/`).
    #[must_use]
    pub fn http_url(&self) -> String {
        format!("http://{}/", self.local_addr)
    }

    /// Returns the WebSocket stream URL (`ws://<addr>/ws`).
    #[must_use]
    pub fn ws_url(&self) -> String {
        format!("ws://{}/ws", self.local_addr)
    }

    /// Returns the broadcast sender for pushing [`AvatarStageMessage`] updates.
    #[must_use]
    pub const fn sender(&self) -> &broadcast::Sender<AvatarStageMessage> {
        &self.tx
    }

    /// Signals the server to shut down and awaits task completion.
    pub async fn shutdown(mut self) -> anyhow::Result<()> {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        if let Some(handle) = self.join_handle.take() {
            handle.await.context("avatar stage task panicked")??;
        }
        Ok(())
    }
}

impl Drop for AvatarStageHandle {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
    }
}

/// Spawns [`serve_avatar_stage`] on the current Tokio runtime and waits until
/// the TCP listener is bound, returning an [`AvatarStageHandle`].
pub async fn spawn_avatar_stage(
    config: AvatarStageConfig,
    initial_snapshot: AvatarStageSnapshot,
) -> anyhow::Result<AvatarStageHandle> {
    let (tx, _) = broadcast::channel(256);
    let (ready_tx, ready_rx) = oneshot::channel();
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let tx_task = tx.clone();

    let join_handle = tokio::spawn(async move {
        serve_avatar_stage(config, initial_snapshot, tx_task, Some(ready_tx), async {
            let _ = shutdown_rx.await;
        })
        .await
    });

    let local_addr = ready_rx
        .await
        .context("avatar stage server terminated before binding listener")?;

    Ok(AvatarStageHandle {
        local_addr,
        tx,
        shutdown_tx: Some(shutdown_tx),
        join_handle: Some(join_handle),
    })
}

/// Runs the local HTTP + WebSocket Companion Avatar server until `shutdown` resolves.
///
/// Routes:
/// - `GET /` (or `/index.html`): transparent HTML5/Canvas2D + VRM-inspecting stage ([`AVATAR_STAGE_HTML`]).
/// - `GET /health`: JSON health & current state summary.
/// - `GET /avatar.vrm`: configured `.vrm` GLB binary (`model/gltf-binary`), or `404`.
/// - `GET /ws` (`Upgrade: websocket`): RFC 6455 JSON stream of [`AvatarStageMessage`].
pub async fn serve_avatar_stage(
    config: AvatarStageConfig,
    mut initial_snapshot: AvatarStageSnapshot,
    tx: broadcast::Sender<AvatarStageMessage>,
    ready_addr_tx: Option<oneshot::Sender<SocketAddr>>,
    shutdown: impl Future<Output = ()> + Send,
) -> anyhow::Result<()> {
    let listener = TcpListener::bind(&config.bind_addr)
        .await
        .with_context(|| format!("failed to bind avatar stage on `{}`", config.bind_addr))?;
    let local_addr = listener.local_addr()?;

    if initial_snapshot.companion_name.trim().is_empty()
        || (initial_snapshot.companion_name == "Jiku" && !config.companion_name.trim().is_empty())
    {
        initial_snapshot.companion_name = config.companion_name.clone();
    }
    initial_snapshot.vrm_available = config.vrm_path.as_deref().is_some_and(|p| p.is_file());

    if let Some(ready_tx) = ready_addr_tx {
        let _ = ready_tx.send(local_addr);
    }

    let snapshot = Arc::new(RwLock::new(initial_snapshot));
    let config = Arc::new(config);
    let mut state_rx = tx.subscribe();
    let mut shutdown_fut = std::pin::pin!(shutdown);

    loop {
        tokio::select! {
            _ = &mut shutdown_fut => {
                break;
            }
            msg_res = state_rx.recv() => {
                match msg_res {
                    Ok(msg) => {
                        snapshot.write().await.apply_message(&msg);
                    }
                    Err(broadcast::error::RecvError::Lagged(_)) => {}
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
            accept_res = listener.accept() => {
                match accept_res {
                    Ok((stream, _peer)) => {
                        let cfg = Arc::clone(&config);
                        let snap = Arc::clone(&snapshot);
                        let tx_conn = tx.clone();
                        tokio::spawn(async move {
                            let _ = handle_connection(stream, cfg, snap, tx_conn).await;
                        });
                    }
                    Err(err) => {
                        return Err(anyhow!("avatar stage accept failed: {err}"));
                    }
                }
            }
        }
    }

    Ok(())
}

struct ParsedHttpRequest {
    method: String,
    path: String,
    headers: Vec<(String, String)>,
}

impl ParsedHttpRequest {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
}

async fn read_http_request(stream: &mut TcpStream) -> anyhow::Result<ParsedHttpRequest> {
    let mut buf = Vec::with_capacity(1024);
    let mut tmp = [0_u8; 1024];

    let header_end = loop {
        let read_len = tokio::time::timeout(HTTP_READ_TIMEOUT, stream.read(&mut tmp))
            .await
            .context("timed out reading HTTP request")??;
        if read_len == 0 {
            return Err(anyhow!("unexpected EOF while reading HTTP request"));
        }
        if buf.len() + read_len > MAX_HTTP_HEAD_BYTES {
            return Err(anyhow!(
                "HTTP request header exceeds {MAX_HTTP_HEAD_BYTES} bytes"
            ));
        }
        buf.extend_from_slice(&tmp[..read_len]);
        if let Some(pos) = find_header_end(&buf) {
            break pos;
        }
    };

    let head_str = std::str::from_utf8(&buf[..header_end]).context("non-UTF-8 HTTP header")?;
    let mut lines = head_str.split("\r\n");
    let request_line = lines
        .next()
        .ok_or_else(|| anyhow!("missing HTTP request line"))?;
    let mut parts = request_line.split_whitespace();
    let method = parts
        .next()
        .ok_or_else(|| anyhow!("missing HTTP method"))?
        .to_ascii_uppercase();
    let raw_target = parts
        .next()
        .ok_or_else(|| anyhow!("missing HTTP request target"))?;
    let path = raw_target.split('?').next().unwrap_or("/").to_string();

    let mut headers = Vec::new();
    for line in lines {
        if line.is_empty() {
            continue;
        }
        if let Some((k, v)) = line.split_once(':') {
            headers.push((k.trim().to_string(), v.trim().to_string()));
        }
    }

    Ok(ParsedHttpRequest {
        method,
        path,
        headers,
    })
}

fn find_header_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n")
}

async fn handle_connection(
    mut stream: TcpStream,
    config: Arc<AvatarStageConfig>,
    snapshot: Arc<RwLock<AvatarStageSnapshot>>,
    tx: broadcast::Sender<AvatarStageMessage>,
) -> anyhow::Result<()> {
    let req = match read_http_request(&mut stream).await {
        Ok(r) => r,
        Err(_) => {
            let _ = write_http_response(
                &mut stream,
                400,
                "Bad Request",
                "text/plain; charset=utf-8",
                b"400 Bad Request\n",
                false,
            )
            .await;
            return Ok(());
        }
    };

    let is_ws_upgrade = req.method == "GET"
        && req.path == "/ws"
        && req
            .header("upgrade")
            .is_some_and(|v| v.eq_ignore_ascii_case("websocket"))
        && req
            .header("connection")
            .is_some_and(|v| v.to_ascii_lowercase().contains("upgrade"));

    if is_ws_upgrade {
        let Some(ws_key) = req.header("sec-websocket-key").map(str::trim) else {
            write_http_response(
                &mut stream,
                400,
                "Bad Request",
                "text/plain; charset=utf-8",
                b"missing Sec-WebSocket-Key\n",
                false,
            )
            .await?;
            return Ok(());
        };
        let accept = compute_ws_accept_key(ws_key);
        let handshake = format!(
            "HTTP/1.1 101 Switching Protocols\r\n\
             Upgrade: websocket\r\n\
             Connection: Upgrade\r\n\
             Sec-WebSocket-Accept: {accept}\r\n\r\n"
        );
        stream.write_all(handshake.as_bytes()).await?;
        stream.flush().await?;
        return run_ws_client(stream, snapshot, tx).await;
    }

    if req.method != "GET" && req.method != "HEAD" {
        write_http_response(
            &mut stream,
            405,
            "Method Not Allowed",
            "text/plain; charset=utf-8",
            b"Method Not Allowed\n",
            false,
        )
        .await?;
        return Ok(());
    }
    let head_only = req.method == "HEAD";

    match req.path.as_str() {
        "/" | "/index.html" => {
            write_http_response(
                &mut stream,
                200,
                "OK",
                "text/html; charset=utf-8",
                AVATAR_STAGE_HTML.as_bytes(),
                head_only,
            )
            .await?;
        }
        "/health" => {
            let snap = snapshot.read().await.clone();
            let vrm_available = config.vrm_path.as_deref().is_some_and(Path::is_file);
            let payload = json!({
                "status": "ok",
                "companion_name": snap.companion_name,
                "emotion": snap.cue.emotion,
                "speaking": snap.speaking,
                "gaze_uv": snap.cue.gaze_uv,
                "vrm_available": vrm_available,
            });
            let body = serde_json::to_vec(&payload)?;
            write_http_response(&mut stream, 200, "OK", "application/json", &body, head_only)
                .await?;
        }
        "/avatar.vrm" => {
            if let Some(vrm_path) = config.vrm_path.as_deref()
                && let Ok(meta) = tokio::fs::metadata(vrm_path).await
                && meta.is_file()
                && meta.len() <= MAX_VRM_FILE_BYTES
                && let Ok(bytes) = tokio::fs::read(vrm_path).await
            {
                write_http_response(
                    &mut stream,
                    200,
                    "OK",
                    "model/gltf-binary",
                    &bytes,
                    head_only,
                )
                .await?;
                return Ok(());
            }
            write_http_response(
                &mut stream,
                404,
                "Not Found",
                "text/plain; charset=utf-8",
                b"avatar.vrm not configured\n",
                head_only,
            )
            .await?;
        }
        _ => {
            write_http_response(
                &mut stream,
                404,
                "Not Found",
                "text/plain; charset=utf-8",
                b"404 Not Found\n",
                head_only,
            )
            .await?;
        }
    }

    Ok(())
}

async fn write_http_response(
    stream: &mut TcpStream,
    status: u16,
    reason: &str,
    content_type: &str,
    body: &[u8],
    head_only: bool,
) -> anyhow::Result<()> {
    let head = format!(
        "HTTP/1.1 {status} {reason}\r\n\
         Content-Type: {content_type}\r\n\
         Content-Length: {}\r\n\
         Cache-Control: no-store\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Connection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes()).await?;
    if !head_only && !body.is_empty() {
        stream.write_all(body).await?;
    }
    stream.flush().await?;
    Ok(())
}

enum WsOutgoing {
    Text(String),
    Pong(Vec<u8>),
    Close,
}

async fn run_ws_client(
    mut stream: TcpStream,
    snapshot: Arc<RwLock<AvatarStageSnapshot>>,
    tx: broadcast::Sender<AvatarStageMessage>,
) -> anyhow::Result<()> {
    let mut rx = tx.subscribe();
    let initial = AvatarStageMessage::Snapshot(snapshot.read().await.clone());
    let initial_json = serde_json::to_string(&initial)?;
    write_ws_frame(&mut stream, 0x1, initial_json.as_bytes(), None).await?;

    let (mut reader, mut writer) = stream.into_split();
    let (ctrl_tx, mut ctrl_rx) = mpsc::channel::<WsOutgoing>(16);

    let snap_reader = Arc::clone(&snapshot);
    let reader_task = tokio::spawn(async move {
        while let Ok(Some(frame)) = read_ws_frame(&mut reader, MAX_WS_INCOMING_FRAME_BYTES).await {
            match frame.opcode {
                0x8 => {
                    let _ = ctrl_tx.send(WsOutgoing::Close).await;
                    break;
                }
                0x9 => {
                    if ctrl_tx.send(WsOutgoing::Pong(frame.payload)).await.is_err() {
                        break;
                    }
                }
                0x1 => {
                    if let Ok(text) = std::str::from_utf8(&frame.payload)
                        && let Ok(val) = serde_json::from_str::<Value>(text)
                    {
                        match val.get("type").and_then(Value::as_str) {
                            Some("ping") => {
                                let _ = ctrl_tx
                                    .send(WsOutgoing::Text(r#"{"type":"pong"}"#.to_string()))
                                    .await;
                            }
                            Some("snapshot_request") => {
                                let msg =
                                    AvatarStageMessage::Snapshot(snap_reader.read().await.clone());
                                if let Ok(json) = serde_json::to_string(&msg) {
                                    let _ = ctrl_tx.send(WsOutgoing::Text(json)).await;
                                }
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }
    });

    loop {
        tokio::select! {
            ctrl = ctrl_rx.recv() => {
                match ctrl {
                    Some(WsOutgoing::Text(text)) => {
                        if write_ws_frame(&mut writer, 0x1, text.as_bytes(), None).await.is_err() {
                            break;
                        }
                    }
                    Some(WsOutgoing::Pong(payload)) => {
                        if write_ws_frame(&mut writer, 0xA, &payload, None).await.is_err() {
                            break;
                        }
                    }
                    Some(WsOutgoing::Close) | None => {
                        let _ = write_ws_frame(&mut writer, 0x8, &[], None).await;
                        break;
                    }
                }
            }
            msg_res = rx.recv() => {
                match msg_res {
                    Ok(msg) => {
                        let json = serde_json::to_string(&msg)?;
                        if write_ws_frame(&mut writer, 0x1, json.as_bytes(), None).await.is_err() {
                            break;
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(_)) => {
                        let resync = AvatarStageMessage::Snapshot(snapshot.read().await.clone());
                        let json = serde_json::to_string(&resync)?;
                        if write_ws_frame(&mut writer, 0x1, json.as_bytes(), None).await.is_err() {
                            break;
                        }
                    }
                    Err(broadcast::error::RecvError::Closed) => {
                        let _ = write_ws_frame(&mut writer, 0x8, &[], None).await;
                        break;
                    }
                }
            }
        }
    }

    reader_task.abort();
    Ok(())
}

/// Computes the RFC 6455 `Sec-WebSocket-Accept` header value from a client's `Sec-WebSocket-Key`.
#[must_use]
pub fn compute_ws_accept_key(client_key: &str) -> String {
    let mut input = Vec::with_capacity(client_key.trim().len() + WS_GUID.len());
    input.extend_from_slice(client_key.trim().as_bytes());
    input.extend_from_slice(WS_GUID.as_bytes());
    let digest = sha1_digest(&input);
    BASE64_STD.encode(digest)
}

/// Pure-Rust FIPS 180-4 / RFC 3174 SHA-1 digest used solely for RFC 6455 WebSocket
/// `Sec-WebSocket-Accept` handshake validation.
fn sha1_digest(data: &[u8]) -> [u8; 20] {
    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut padded = Vec::with_capacity(data.len() + 72);
    padded.extend_from_slice(data);
    padded.push(0x80);
    while (padded.len() % 64) != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    let mut h0 = 0x6745_2301_u32;
    let mut h1 = 0xEFCD_AB89_u32;
    let mut h2 = 0x98BA_DCFE_u32;
    let mut h3 = 0x1032_5476_u32;
    let mut h4 = 0xC3D2_E1F0_u32;

    let (blocks, _) = padded.as_chunks::<64>();
    for block in blocks {
        let mut w = [0_u32; 80];
        let (words, _) = block.as_chunks::<4>();
        for (idx, &word) in words.iter().enumerate() {
            w[idx] = u32::from_be_bytes(word);
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }

        let (mut a, mut b, mut c, mut d, mut e) = (h0, h1, h2, h3, h4);
        for (i, &wi) in w.iter().enumerate() {
            let (f, k) = match i {
                0..=19 => ((b & c) | ((!b) & d), 0x5A82_7999_u32),
                20..=39 => (b ^ c ^ d, 0x6ED9_EBA1_u32),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1B_BCDC_u32),
                _ => (b ^ c ^ d, 0xCA62_C1D6_u32),
            };
            let temp = a
                .rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(wi);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = temp;
        }

        h0 = h0.wrapping_add(a);
        h1 = h1.wrapping_add(b);
        h2 = h2.wrapping_add(c);
        h3 = h3.wrapping_add(d);
        h4 = h4.wrapping_add(e);
    }

    let mut out = [0_u8; 20];
    out[0..4].copy_from_slice(&h0.to_be_bytes());
    out[4..8].copy_from_slice(&h1.to_be_bytes());
    out[8..12].copy_from_slice(&h2.to_be_bytes());
    out[12..16].copy_from_slice(&h3.to_be_bytes());
    out[16..20].copy_from_slice(&h4.to_be_bytes());
    out
}

struct WsFrame {
    opcode: u8,
    payload: Vec<u8>,
}

async fn read_ws_frame<R: AsyncRead + Unpin>(
    reader: &mut R,
    max_bytes: usize,
) -> anyhow::Result<Option<WsFrame>> {
    let mut assembled_opcode: Option<u8> = None;
    let mut assembled_payload: Vec<u8> = Vec::new();

    loop {
        let mut head = [0_u8; 2];
        match reader.read_exact(&mut head).await {
            Ok(_) => {}
            Err(err) if err.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(err) => return Err(err.into()),
        }

        let fin = (head[0] & 0x80) != 0;
        let rsv = head[0] & 0x70;
        if rsv != 0 {
            return Err(anyhow!("non-zero WebSocket RSV bits"));
        }
        let opcode = head[0] & 0x0F;
        let masked = (head[1] & 0x80) != 0;
        let len7 = head[1] & 0x7F;

        let payload_len: usize = match len7 {
            0..=125 => usize::from(len7),
            126 => {
                let mut ext = [0_u8; 2];
                reader.read_exact(&mut ext).await?;
                usize::from(u16::from_be_bytes(ext))
            }
            _ => {
                let mut ext = [0_u8; 8];
                reader.read_exact(&mut ext).await?;
                let raw = u64::from_be_bytes(ext);
                if (raw & (1_u64 << 63)) != 0 {
                    return Err(anyhow!("invalid WebSocket 64-bit length MSB"));
                }
                usize::try_from(raw).map_err(|_| anyhow!("WebSocket payload overflows usize"))?
            }
        };

        if payload_len > max_bytes
            || assembled_payload.len().saturating_add(payload_len) > max_bytes
        {
            return Err(anyhow!(
                "WebSocket frame exceeds max allowed size {max_bytes}"
            ));
        }

        let mask_key = if masked {
            let mut key = [0_u8; 4];
            reader.read_exact(&mut key).await?;
            Some(key)
        } else {
            None
        };

        let mut payload = vec![0_u8; payload_len];
        if payload_len > 0 {
            reader.read_exact(&mut payload).await?;
            if let Some(key) = mask_key {
                for (idx, byte) in payload.iter_mut().enumerate() {
                    *byte ^= key[idx & 3];
                }
            }
        }

        // Control frames (0x8..=0xF) may interleave fragmented data frames per RFC 6455 §5.4.
        if opcode >= 0x8 {
            return Ok(Some(WsFrame { opcode, payload }));
        }

        if opcode == 0x0 {
            if assembled_opcode.is_none() {
                return Err(anyhow!("unexpected WebSocket continuation frame"));
            }
            assembled_payload.extend_from_slice(&payload);
        } else {
            if assembled_opcode.is_some() {
                return Err(anyhow!("nested WebSocket fragmented data frame"));
            }
            if fin {
                return Ok(Some(WsFrame { opcode, payload }));
            }
            assembled_opcode = Some(opcode);
            assembled_payload = payload;
        }

        if fin && let Some(op) = assembled_opcode {
            return Ok(Some(WsFrame {
                opcode: op,
                payload: assembled_payload,
            }));
        }
    }
}

async fn write_ws_frame<W: AsyncWrite + Unpin>(
    writer: &mut W,
    opcode: u8,
    payload: &[u8],
    mask_key: Option<[u8; 4]>,
) -> anyhow::Result<()> {
    let mut header = Vec::with_capacity(14);
    header.push(0x80 | (opcode & 0x0F));
    let mask_bit = if mask_key.is_some() { 0x80_u8 } else { 0x00_u8 };

    let len = payload.len();
    if len <= 125 {
        header.push(mask_bit | (len as u8));
    } else if len <= usize::from(u16::MAX) {
        header.push(mask_bit | 126);
        header.extend_from_slice(&(len as u16).to_be_bytes());
    } else {
        header.push(mask_bit | 127);
        header.extend_from_slice(&(len as u64).to_be_bytes());
    }

    if let Some(key) = mask_key {
        header.extend_from_slice(&key);
        writer.write_all(&header).await?;
        let mut masked = payload.to_vec();
        for (idx, byte) in masked.iter_mut().enumerate() {
            *byte ^= key[idx & 3];
        }
        if !masked.is_empty() {
            writer.write_all(&masked).await?;
        }
    } else {
        writer.write_all(&header).await?;
        if !payload.is_empty() {
            writer.write_all(payload).await?;
        }
    }
    writer.flush().await?;
    Ok(())
}

/// Embedded zero-external-asset HTML5/Canvas2D + optional VRM-inspecting companion
/// stage served at `GET /`.
///
/// Query parameters:
/// - `?captions=0`: hides the bottom live caption/status strip.
/// - `?hud=1`: shows compact acoustic RMS/low/mid/high + viseme telemetry meters.
/// - `?bg=dark`: adds a subtle dark studio backdrop when previewing outside OBS.
/// - `?ws=ws://...`: overrides the default `/ws` WebSocket endpoint.
pub const AVATAR_STAGE_HTML: &str = r####"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Jiku Companion Avatar Stage</title>
<style>
  :root {
    --badge-bg: rgba(15, 20, 32, 0.72);
    --badge-border: rgba(148, 184, 255, 0.34);
    --text-main: #eef4ff;
    --text-muted: #9eb4d6;
    --accent: #67d6ff;
  }
  html, body {
    margin: 0;
    padding: 0;
    width: 100%;
    height: 100%;
    overflow: hidden;
    background: transparent;
    font-family: Inter, "Noto Sans CJK SC", system-ui, -apple-system, sans-serif;
    user-select: none;
  }
  body.preview-dark {
    background: radial-gradient(circle at 50% 42%, #192236 0%, #0b0f19 72%);
  }
  #stage-wrap {
    position: relative;
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  #stage {
    width: 100%;
    height: 100%;
    display: block;
  }
  #overlay-top {
    position: fixed;
    top: 14px;
    left: 50%;
    transform: translateX(-50%);
    display: flex;
    align-items: center;
    gap: 8px;
    pointer-events: none;
  }
  .pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 5px 12px;
    border-radius: 999px;
    background: var(--badge-bg);
    border: 1px solid var(--badge-border);
    backdrop-filter: blur(10px);
    color: var(--text-main);
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.03em;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.28);
  }
  .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 10px var(--accent);
  }
  #vrm-badge {
    display: none;
    color: #b8f3ff;
  }
  #hud {
    position: fixed;
    top: 50px;
    right: 14px;
    display: none;
    flex-direction: column;
    gap: 4px;
    padding: 8px 10px;
    border-radius: 10px;
    background: var(--badge-bg);
    border: 1px solid var(--badge-border);
    color: var(--text-muted);
    font-size: 11px;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }
  #caption-bar {
    position: fixed;
    bottom: 18px;
    left: 50%;
    transform: translateX(-50%);
    max-width: min(86vw, 680px);
    padding: 8px 16px;
    border-radius: 14px;
    background: var(--badge-bg);
    border: 1px solid var(--badge-border);
    backdrop-filter: blur(12px);
    color: var(--text-main);
    font-size: 14px;
    line-height: 1.45;
    text-align: center;
    box-shadow: 0 10px 28px rgba(0, 0, 0, 0.34);
    transition: opacity 0.2s ease, transform 0.2s ease;
  }
  #caption-bar.hidden {
    opacity: 0;
    pointer-events: none;
    transform: translateX(-50%) translateY(6px);
  }
  #drop-hint {
    position: fixed;
    inset: 12px;
    border: 2px dashed rgba(103, 214, 255, 0.75);
    border-radius: 18px;
    background: rgba(12, 18, 30, 0.55);
    color: #e6f7ff;
    display: none;
    align-items: center;
    justify-content: center;
    font-size: 15px;
    font-weight: 600;
    pointer-events: none;
  }
</style>
</head>
<body>
<div id="stage-wrap">
  <canvas id="stage"></canvas>
  <div id="overlay-top">
    <div class="pill"><span class="dot" id="status-dot"></span><span id="name-label">JIKU</span><span id="emotion-label">NEUTRAL</span></div>
    <div class="pill" id="vrm-badge">VRM</div>
  </div>
  <div id="hud"></div>
  <div id="caption-bar" class="hidden"></div>
  <div id="drop-hint">Drop a .vrm file to inspect VRM metadata</div>
</div>
<script>
(() => {
  const params = new URLSearchParams(window.location.search);
  const showCaptions = params.get("captions") !== "0";
  const showHud = params.get("hud") === "1";
  if (params.get("bg") === "dark") {
    document.body.classList.add("preview-dark");
  }

  const canvas = document.getElementById("stage");
  const ctx = canvas.getContext("2d");
  const nameLabel = document.getElementById("name-label");
  const emotionLabel = document.getElementById("emotion-label");
  const statusDot = document.getElementById("status-dot");
  const vrmBadge = document.getElementById("vrm-badge");
  const hud = document.getElementById("hud");
  const captionBar = document.getElementById("caption-bar");
  const dropHint = document.getElementById("drop-hint");
  if (showHud) hud.style.display = "flex";

  const EMOTIONS = {
    neutral:       { ring: "#6dd3ff", glow: "rgba(109, 211, 255, 0.34)", iris: "#38b6ff", hairTint: "#8fd4ff", browTilt: 0.0,  browLift: 0.0,  smile: 0.28, label: "NEUTRAL" },
    idle:          { ring: "#8fa8ff", glow: "rgba(143, 168, 255, 0.28)", iris: "#6f8cff", hairTint: "#a9baff", browTilt: -0.04,browLift: 0.04, smile: 0.18, label: "IDLE" },
    thinking:      { ring: "#c084fc", glow: "rgba(192, 132, 252, 0.38)", iris: "#a855f7", hairTint: "#d8b4fe", browTilt: 0.22, browLift: 0.12, smile: -0.05,label: "THINKING" },
    acting:        { ring: "#22d3ee", glow: "rgba(34, 211, 238, 0.42)", iris: "#06b6d4", hairTint: "#67e8f9", browTilt: -0.18,browLift: -0.06,smile: 0.14, label: "ACTING" },
    speaking:      { ring: "#38bdf8", glow: "rgba(56, 189, 248, 0.48)", iris: "#0ea5e9", hairTint: "#7dd3fc", browTilt: 0.04, browLift: 0.08, smile: 0.35, label: "SPEAKING" },
    waiting_human: { ring: "#fbbf24", glow: "rgba(251, 191, 36, 0.45)",  iris: "#f59e0b", hairTint: "#fde68a", browTilt: 0.18, browLift: 0.22, smile: 0.02, label: "WAIT HUMAN" },
    success:       { ring: "#34d399", glow: "rgba(52, 211, 153, 0.46)",  iris: "#10b981", hairTint: "#6ee7b7", browTilt: 0.06, browLift: 0.16, smile: 0.72, label: "SUCCESS" },
    alert:         { ring: "#fb7185", glow: "rgba(251, 113, 133, 0.54)", iris: "#f43f5e", hairTint: "#fda4af", browTilt: -0.32,browLift: -0.14,smile: -0.38,label: "ALERT" }
  };

  const VISEME_SHAPES = {
    sil: { open: 0.0,  wide: 0.35, round: 0.0 },
    N:   { open: 0.04, wide: 0.40, round: 0.0 },
    a:   { open: 0.92, wide: 0.72, round: 0.15 },
    i:   { open: 0.28, wide: 0.96, round: 0.0 },
    u:   { open: 0.38, wide: 0.24, round: 0.95 },
    e:   { open: 0.58, wide: 0.86, round: 0.05 },
    o:   { open: 0.74, wide: 0.42, round: 0.88 }
  };

  const state = {
    companionName: "Jiku",
    emotion: "neutral",
    speaking: false,
    connected: false,
    caption: null,
    gazeTarget: { u: 0.5, v: 0.48 },
    gazePos: { x: 0.0, y: 0.0, vx: 0.0, vy: 0.0 },
    mouthOpen: { pos: 0.0, vel: 0.0 },
    mouthWide: { pos: 0.4, vel: 0.0 },
    mouthRound: { pos: 0.0, vel: 0.0 },
    pcm: { rms: 0.0, low_band: 0.0, mid_band: 0.0, high_band: 0.0 },
    pcmDecayAt: 0,
    moraQueue: [],
    moraStartMs: 0,
    activeVowel: "sil",
    blinkNextAt: performance.now() + 2600,
    blinkProgress: -1,
    doubleBlink: false,
    ws: null,
    reconnectDelay: 500
  };

  function resizeCanvas() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2.5);
    const w = Math.max(1, window.innerWidth);
    const h = Math.max(1, window.innerHeight);
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  window.addEventListener("resize", resizeCanvas);
  resizeCanvas();

  function stepSpring(spring, target, stiffness, damping, dt) {
    const force = (target - spring.pos) * stiffness;
    spring.vel = (spring.vel + force * dt) * Math.exp(-damping * dt);
    spring.pos += spring.vel * dt;
  }

  function updateCaption(text) {
    state.caption = text || null;
    if (!showCaptions || !state.caption) {
      captionBar.classList.add("hidden");
      captionBar.textContent = "";
    } else {
      captionBar.textContent = state.caption;
      captionBar.classList.remove("hidden");
    }
  }

  function applyCue(cue, caption, speaking) {
    if (!cue) return;
    if (cue.emotion && EMOTIONS[cue.emotion]) {
      state.emotion = cue.emotion;
    }
    if (Array.isArray(cue.gaze_uv) && cue.gaze_uv.length === 2) {
      state.gazeTarget.u = Math.max(0, Math.min(1, Number(cue.gaze_uv[0]) || 0.5));
      state.gazeTarget.v = Math.max(0, Math.min(1, Number(cue.gaze_uv[1]) || 0.5));
    }
    if (Array.isArray(cue.mora_envelope) && cue.mora_envelope.length > 0) {
      state.moraQueue = cue.mora_envelope;
      state.moraStartMs = performance.now();
    } else if (cue.interrupted) {
      state.moraQueue = [];
    }
    if (cue.pcm_envelope) {
      state.pcm = cue.pcm_envelope;
      state.pcmDecayAt = performance.now() + 260;
    }
    if (typeof speaking === "boolean") {
      state.speaking = speaking;
    }
    if (caption !== undefined && caption !== null) {
      updateCaption(caption);
    }
  }

  function handleMessage(msg) {
    if (!msg || !msg.type) return;
    switch (msg.type) {
      case "snapshot":
        state.companionName = msg.companion_name || "Jiku";
        nameLabel.textContent = state.companionName.toUpperCase();
        applyCue(msg.cue, msg.caption, !!msg.speaking);
        if (msg.vrm_available) {
          probeVrmFromServer();
        }
        break;
      case "cue":
        applyCue(msg.cue, msg.caption, msg.speaking);
        break;
      case "envelope_frame":
        if (msg.frame) {
          state.pcm = msg.frame;
          state.pcmDecayAt = performance.now() + 180;
          state.speaking = (msg.frame.rms || 0) >= 0.015;
          if (msg.vowel) state.activeVowel = msg.vowel;
        }
        break;
      case "gaze":
        state.gazeTarget.u = Math.max(0, Math.min(1, Number(msg.u) || 0.5));
        state.gazeTarget.v = Math.max(0, Math.min(1, Number(msg.v) || 0.5));
        break;
      case "barge_in_reset":
        state.emotion = EMOTIONS[msg.emotion] ? msg.emotion : "idle";
        state.speaking = false;
        state.moraQueue = [];
        state.activeVowel = "sil";
        state.pcm = { rms: 0, low_band: 0, mid_band: 0, high_band: 0 };
        break;
      case "caption":
        updateCaption(msg.text || null);
        break;
    }
    const theme = EMOTIONS[state.emotion] || EMOTIONS.neutral;
    emotionLabel.textContent = theme.label;
    statusDot.style.background = theme.ring;
    statusDot.style.boxShadow = `0 0 10px ${theme.ring}`;
  }

  function parseVrmGlbBuffer(arrayBuffer, sourceLabel) {
    if (arrayBuffer.byteLength < 20) return null;
    const view = new DataView(arrayBuffer);
    if (view.getUint32(0, true) !== 0x46546c67) return null; // "glTF"
    const chunkLen = view.getUint32(12, true);
    const chunkType = view.getUint32(16, true);
    if (chunkType !== 0x4e4f534a || 20 + chunkLen > arrayBuffer.byteLength) return null;
    const jsonBytes = new Uint8Array(arrayBuffer, 20, chunkLen);
    const gltf = JSON.parse(new TextDecoder().decode(jsonBytes));
    const ext = gltf.extensions || {};
    let title = sourceLabel;
    let exprCount = 0;
    let spec = "VRM";
    if (ext.VRMC_vrm && ext.VRMC_vrm.meta) {
      spec = "VRM 1.0";
      title = ext.VRMC_vrm.meta.name || title;
      const preset = ext.VRMC_vrm.expressions && ext.VRMC_vrm.expressions.preset;
      if (preset) exprCount = Object.keys(preset).length;
    } else if (ext.VRM && ext.VRM.meta) {
      spec = "VRM 0.x";
      title = ext.VRM.meta.title || title;
      const groups = ext.VRM.blendShapeMaster && ext.VRM.blendShapeMaster.blendShapeGroups;
      if (Array.isArray(groups)) exprCount = groups.length;
    }
    return { spec, title, exprCount };
  }

  async function probeVrmFromServer() {
    try {
      const resp = await fetch("/avatar.vrm", { cache: "no-store" });
      if (!resp.ok) return;
      const buf = await resp.arrayBuffer();
      const info = parseVrmGlbBuffer(buf, "avatar.vrm");
      vrmBadge.style.display = "inline-flex";
      vrmBadge.textContent = info
        ? `${info.spec}: ${info.title} (${info.exprCount} expr)`
        : `VRM: ${(buf.byteLength / 1024).toFixed(0)} KiB`;
    } catch (_) {}
  }

  window.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropHint.style.display = "flex";
  });
  window.addEventListener("dragleave", () => {
    dropHint.style.display = "none";
  });
  window.addEventListener("drop", async (e) => {
    e.preventDefault();
    dropHint.style.display = "none";
    const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (!file) return;
    const buf = await file.arrayBuffer();
    const info = parseVrmGlbBuffer(buf, file.name);
    vrmBadge.style.display = "inline-flex";
    vrmBadge.textContent = info
      ? `${info.spec}: ${info.title} (${info.exprCount} expr)`
      : `GLB: ${file.name} (${(buf.byteLength / 1024).toFixed(0)} KiB)`;
  });

  function connectWs() {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = params.get("ws") || `${proto}//${window.location.host}/ws`;
    try {
      const ws = new WebSocket(wsUrl);
      state.ws = ws;
      ws.onopen = () => {
        state.connected = true;
        state.reconnectDelay = 500;
      };
      ws.onmessage = (ev) => {
        try {
          handleMessage(JSON.parse(ev.data));
        } catch (_) {}
      };
      ws.onclose = () => {
        state.connected = false;
        setTimeout(connectWs, state.reconnectDelay);
        state.reconnectDelay = Math.min(5000, Math.round(state.reconnectDelay * 1.5));
      };
      ws.onerror = () => ws.close();
    } catch (_) {
      setTimeout(connectWs, 1500);
    }
  }
  connectWs();
  probeVrmFromServer();

  function activeMoraAt(now) {
    if (!state.moraQueue.length) return null;
    const elapsed = now - state.moraStartMs;
    const last = state.moraQueue[state.moraQueue.length - 1];
    const totalMs = (last.start_ms || 0) + (last.duration_ms || 0);
    if (elapsed > totalMs + 80) {
      state.moraQueue = [];
      return null;
    }
    for (let i = 0; i < state.moraQueue.length; i++) {
      const m = state.moraQueue[i];
      if (elapsed >= m.start_ms && elapsed <= m.start_ms + m.duration_ms) {
        return m.vowel || "a";
      }
    }
    return "sil";
  }

  let prevTs = performance.now();
  function render(now) {
    const dt = Math.min(0.05, Math.max(0.001, (now - prevTs) / 1000));
    prevTs = now;

    if (now > state.pcmDecayAt && !state.moraQueue.length) {
      state.pcm.rms *= Math.exp(-10 * dt);
      state.pcm.low_band *= Math.exp(-10 * dt);
      state.pcm.mid_band *= Math.exp(-10 * dt);
      state.pcm.high_band *= Math.exp(-10 * dt);
      if (state.pcm.rms < 0.01) state.speaking = false;
    }

    const moraVowel = activeMoraAt(now);
    const vowelKey = moraVowel || (state.pcm.rms >= 0.02 ? state.activeVowel : "sil");
    const shape = VISEME_SHAPES[vowelKey] || VISEME_SHAPES.a;
    const pcmBoost = Math.min(1.0, (state.pcm.rms || 0) * 2.4 + (state.pcm.mid_band || 0) * 0.7);

    let targetOpen = 0.0;
    let targetWide = 0.45;
    let targetRound = 0.0;
    if (moraVowel && moraVowel !== "sil") {
      const amp = pcmBoost > 0.05 ? 0.55 + 0.45 * pcmBoost : 0.82;
      targetOpen = shape.open * amp;
      targetWide = shape.wide;
      targetRound = shape.round;
    } else if (state.pcm.rms >= 0.02) {
      targetOpen = Math.min(0.95, pcmBoost);
      targetWide = 0.45 + 0.45 * Math.min(1, (state.pcm.high_band || 0) * 2.2);
      targetRound = Math.min(0.85, (state.pcm.low_band || 0) * 1.8);
    }

    stepSpring(state.mouthOpen, targetOpen, 220, 18, dt);
    stepSpring(state.mouthWide, targetWide, 180, 16, dt);
    stepSpring(state.mouthRound, targetRound, 180, 16, dt);

    const gxTarget = (state.gazeTarget.u - 0.5) * 2.0;
    const gyTarget = (state.gazeTarget.v - 0.5) * 2.0;
    const gxForce = (gxTarget - state.gazePos.x) * 85;
    const gyForce = (gyTarget - state.gazePos.y) * 85;
    state.gazePos.vx = (state.gazePos.vx + gxForce * dt) * Math.exp(-12 * dt);
    state.gazePos.vy = (state.gazePos.vy + gyForce * dt) * Math.exp(-12 * dt);
    state.gazePos.x += state.gazePos.vx * dt;
    state.gazePos.y += state.gazePos.vy * dt;

    if (now >= state.blinkNextAt && state.blinkProgress < 0) {
      state.blinkProgress = 0;
      state.doubleBlink = Math.random() < 0.22;
    }
    let blinkClose = 0.0;
    if (state.blinkProgress >= 0) {
      state.blinkProgress += dt / 0.14;
      if (state.blinkProgress >= 1.0) {
        if (state.doubleBlink) {
          state.doubleBlink = false;
          state.blinkProgress = 0;
        } else {
          state.blinkProgress = -1;
          state.blinkNextAt = now + 2400 + Math.random() * 3200;
        }
      } else {
        blinkClose = Math.sin(state.blinkProgress * Math.PI);
      }
    }

    const w = window.innerWidth;
    const h = window.innerHeight;
    ctx.clearRect(0, 0, w, h);

    const theme = EMOTIONS[state.emotion] || EMOTIONS.neutral;
    const scale = Math.min(w, h) / 360;
    const cx = w * 0.5;
    const cy = h * 0.52;
    const t = now * 0.001;
    const breath = Math.sin(t * 2.1) * 4.2 * scale;
    const headBob = Math.sin(t * 2.1 - 0.5) * 3.0 * scale + state.mouthOpen.pos * 3.0 * scale;
    const tilt = state.gazePos.x * 0.07 + (state.emotion === "thinking" ? 0.075 : 0);

    // Outer Aura & Harmonic Equalizer Ring
    ctx.save();
    ctx.translate(cx, cy + breath * 0.5);
    const auraR = 118 * scale * (1 + Math.sin(t * 2.6) * 0.025 + state.mouthOpen.pos * 0.06);
    const auraGrad = ctx.createRadialGradient(0, 0, auraR * 0.35, 0, 0, auraR * 1.35);
    auraGrad.addColorStop(0, theme.glow);
    auraGrad.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = auraGrad;
    ctx.beginPath();
    ctx.arc(0, 0, auraR * 1.35, 0, Math.PI * 2);
    ctx.fill();

    ctx.strokeStyle = theme.ring;
    ctx.lineWidth = (state.emotion === "alert" ? 3.2 : 2.0) * scale;
    ctx.globalAlpha = 0.65;
    if (state.emotion === "thinking") {
      ctx.setLineDash([7 * scale, 7 * scale]);
      ctx.rotate(t * 0.6);
    }
    ctx.beginPath();
    ctx.arc(0, 0, auraR, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);

    if (state.speaking || state.pcm.rms > 0.02) {
      const bars = 24;
      for (let i = 0; i < bars; i++) {
        const ang = (i / bars) * Math.PI * 2 + t * 0.9;
        const band = i % 3 === 0 ? state.pcm.low_band : i % 3 === 1 ? state.pcm.mid_band : state.pcm.high_band;
        const mag = (0.06 + Math.min(1, band * 2.5 + state.mouthOpen.pos * 0.45)) * 18 * scale;
        ctx.beginPath();
        ctx.moveTo(Math.cos(ang) * (auraR + 4 * scale), Math.sin(ang) * (auraR + 4 * scale));
        ctx.lineTo(Math.cos(ang) * (auraR + 4 * scale + mag), Math.sin(ang) * (auraR + 4 * scale + mag));
        ctx.strokeStyle = theme.ring;
        ctx.lineWidth = 2.2 * scale;
        ctx.stroke();
      }
    }
    ctx.restore();

    // Character Body + Shoulders
    ctx.save();
    ctx.translate(cx + state.gazePos.x * 4 * scale, cy + breath);
    ctx.fillStyle = "#1d2742";
    ctx.strokeStyle = theme.ring;
    ctx.lineWidth = 2 * scale;
    ctx.beginPath();
    ctx.moveTo(-58 * scale, 108 * scale);
    ctx.quadraticCurveTo(-48 * scale, 52 * scale, 0, 50 * scale);
    ctx.quadraticCurveTo(48 * scale, 52 * scale, 58 * scale, 108 * scale);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // Glowing collar crest
    ctx.fillStyle = theme.ring;
    ctx.beginPath();
    ctx.moveTo(0, 62 * scale);
    ctx.lineTo(9 * scale, 74 * scale);
    ctx.lineTo(0, 86 * scale);
    ctx.lineTo(-9 * scale, 74 * scale);
    ctx.closePath();
    ctx.fill();
    ctx.restore();

    // Character Head & 2.5D Parallax Face
    ctx.save();
    ctx.translate(cx + state.gazePos.x * 7 * scale, cy - 10 * scale + headBob);
    ctx.rotate(tilt);

    // Back Hair Silhouette
    ctx.fillStyle = "#182138";
    ctx.beginPath();
    ctx.ellipse(0, -6 * scale, 74 * scale, 80 * scale, 0, 0, Math.PI * 2);
    ctx.fill();

    // Spirit Ear Fin Accents
    for (const dir of [-1, 1]) {
      ctx.save();
      ctx.translate(dir * 66 * scale, -18 * scale);
      ctx.rotate(dir * (0.28 + Math.sin(t * 2.4) * 0.04));
      ctx.fillStyle = "#223055";
      ctx.strokeStyle = theme.ring;
      ctx.lineWidth = 1.8 * scale;
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.quadraticCurveTo(dir * 26 * scale, -26 * scale, dir * 16 * scale, -48 * scale);
      ctx.quadraticCurveTo(dir * 4 * scale, -28 * scale, -dir * 4 * scale, -10 * scale);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
      ctx.restore();
    }

    // Face Contour
    const px = state.gazePos.x * 6.5 * scale;
    const py = state.gazePos.y * 4.5 * scale;
    ctx.fillStyle = "#fff1eb";
    ctx.beginPath();
    ctx.moveTo(-54 * scale, -14 * scale);
    ctx.quadraticCurveTo(-56 * scale, 32 * scale, 0, 52 * scale);
    ctx.quadraticCurveTo(56 * scale, 32 * scale, 54 * scale, -14 * scale);
    ctx.quadraticCurveTo(52 * scale, -64 * scale, 0, -64 * scale);
    ctx.quadraticCurveTo(-52 * scale, -64 * scale, -54 * scale, -14 * scale);
    ctx.closePath();
    ctx.fill();

    // Soft Cheek Blush
    for (const dir of [-1, 1]) {
      ctx.fillStyle = state.emotion === "success" ? "rgba(255, 128, 156, 0.36)" : "rgba(255, 148, 168, 0.22)";
      ctx.beginPath();
      ctx.ellipse(px + dir * 32 * scale, py + 15 * scale, 11 * scale, 6 * scale, 0, 0, Math.PI * 2);
      ctx.fill();
    }

    // Eyes
    for (const dir of [-1, 1]) {
      const ex = px + dir * 23 * scale;
      const ey = py - 2 * scale;

      if (state.emotion === "success") {
        ctx.strokeStyle = "#1c233a";
        ctx.lineWidth = 3.2 * scale;
        ctx.lineCap = "round";
        ctx.beginPath();
        ctx.arc(ex, ey + 3 * scale, 10 * scale, Math.PI * 1.15, Math.PI * 1.85);
        ctx.stroke();
      } else {
        const openH = Math.max(0.08, 1.0 - blinkClose);
        ctx.save();
        ctx.beginPath();
        ctx.ellipse(ex, ey, 11.5 * scale, 13.5 * scale * openH, 0, 0, Math.PI * 2);
        ctx.fillStyle = "#ffffff";
        ctx.fill();
        ctx.clip();

        const ix = ex + state.gazePos.x * 4.5 * scale;
        const iy = ey + state.gazePos.y * 3.5 * scale;
        const irisGrad = ctx.createLinearGradient(ix, iy - 11 * scale, ix, iy + 11 * scale);
        irisGrad.addColorStop(0, "#101b38");
        irisGrad.addColorStop(0.55, theme.iris);
        irisGrad.addColorStop(1, theme.hairTint);
        ctx.fillStyle = irisGrad;
        ctx.beginPath();
        ctx.ellipse(ix, iy, 8.2 * scale, 10.6 * scale, 0, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = "#0b1024";
        ctx.beginPath();
        ctx.arc(ix, iy, 3.6 * scale, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = "#ffffff";
        ctx.beginPath();
        ctx.arc(ix - 3.0 * scale, iy - 3.6 * scale, 2.5 * scale, 0, Math.PI * 2);
        ctx.arc(ix + 2.8 * scale, iy + 3.2 * scale, 1.3 * scale, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();

        // Upper eyelash arc
        ctx.strokeStyle = "#182138";
        ctx.lineWidth = 3.0 * scale;
        ctx.lineCap = "round";
        ctx.beginPath();
        ctx.ellipse(ex, ey - (1 - openH) * 2 * scale, 12.2 * scale, 13.5 * scale * openH, 0, Math.PI * 1.08, Math.PI * 1.92);
        ctx.stroke();
      }

      // Eyebrow
      ctx.save();
      ctx.translate(ex, ey - 19 * scale - theme.browLift * 6 * scale);
      ctx.rotate(dir * theme.browTilt);
      ctx.strokeStyle = "#232e4f";
      ctx.lineWidth = 2.4 * scale;
      ctx.lineCap = "round";
      ctx.beginPath();
      ctx.moveTo(-9 * scale, 0);
      ctx.quadraticCurveTo(0, -2.5 * scale, 9 * scale, 0);
      ctx.stroke();
      ctx.restore();
    }

    // Mouth (Mora viseme + 3-band PCM envelope)
    const mx = px;
    const my = py + 26 * scale;
    const openAmt = Math.max(0, Math.min(1, state.mouthOpen.pos));
    const wideAmt = Math.max(0.2, Math.min(1, state.mouthWide.pos));
    const roundAmt = Math.max(0, Math.min(1, state.mouthRound.pos));
    const mw = (6.5 + wideAmt * 9.5 - roundAmt * 3.5) * scale;
    const mh = openAmt * 11.5 * scale;

    if (mh < 1.3 * scale) {
      ctx.strokeStyle = "#6e3b46";
      ctx.lineWidth = 2.2 * scale;
      ctx.lineCap = "round";
      ctx.beginPath();
      ctx.moveTo(mx - mw * 0.75, my);
      ctx.quadraticCurveTo(mx, my + theme.smile * 6 * scale, mx + mw * 0.75, my);
      ctx.stroke();
    } else {
      ctx.fillStyle = "#5c1f2c";
      ctx.strokeStyle = "#8a4654";
      ctx.lineWidth = 1.6 * scale;
      ctx.beginPath();
      ctx.ellipse(mx, my + mh * 0.25, mw * 0.72, mh * 0.65, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = "#f28496";
      ctx.beginPath();
      ctx.ellipse(mx, my + mh * 0.55, mw * 0.44, mh * 0.25, 0, 0, Math.PI * 2);
      ctx.fill();
    }

    // Front Hair Bangs & Ahoge
    ctx.fillStyle = "#233052";
    ctx.beginPath();
    ctx.moveTo(-58 * scale, -18 * scale);
    ctx.quadraticCurveTo(-48 * scale, -72 * scale, 0, -72 * scale);
    ctx.quadraticCurveTo(48 * scale, -72 * scale, 58 * scale, -18 * scale);
    ctx.quadraticCurveTo(38 * scale, -40 * scale, 22 * scale, -20 * scale);
    ctx.quadraticCurveTo(10 * scale, -44 * scale, 0, -18 * scale);
    ctx.quadraticCurveTo(-12 * scale, -42 * scale, -24 * scale, -22 * scale);
    ctx.quadraticCurveTo(-40 * scale, -40 * scale, -58 * scale, -18 * scale);
    ctx.closePath();
    ctx.fill();

    ctx.strokeStyle = theme.hairTint;
    ctx.lineWidth = 2.4 * scale;
    ctx.lineCap = "round";
    ctx.beginPath();
    const sway = Math.sin(t * 2.5) * 4 * scale;
    ctx.moveTo(0, -70 * scale);
    ctx.quadraticCurveTo(10 * scale + sway, -96 * scale, 22 * scale + sway, -84 * scale);
    ctx.stroke();

    // Emotion sigil near crown
    if (state.emotion === "waiting_human") {
      ctx.fillStyle = "#fbbf24";
      ctx.font = `700 ${Math.round(22 * scale)}px system-ui`;
      ctx.fillText("?", 56 * scale, -52 * scale);
    } else if (state.emotion === "alert") {
      ctx.fillStyle = "#fb7185";
      ctx.font = `800 ${Math.round(24 * scale)}px system-ui`;
      ctx.fillText("!", 56 * scale, -52 * scale);
    } else if (state.emotion === "success") {
      ctx.fillStyle = "#34d399";
      ctx.font = `700 ${Math.round(18 * scale)}px system-ui`;
      ctx.fillText("✦", 56 * scale, -52 * scale);
    }
    ctx.restore();

    if (showHud) {
      hud.textContent =
        `ws:${state.connected ? "up" : "retry"} vowel:${vowelKey} ` +
        `rms:${state.pcm.rms.toFixed(2)} L:${state.pcm.low_band.toFixed(2)} ` +
        `M:${state.pcm.mid_band.toFixed(2)} H:${state.pcm.high_band.toFixed(2)}`;
    }

    requestAnimationFrame(render);
  }
  requestAnimationFrame(render);
})();
</script>
</body>
</html>
"####;

#[cfg(test)]
mod tests {
    use super::*;
    use jiku_tts::{
        IrodoriHttpProvider, MoraViseme, OpenAiSpeechProvider, TtsProvider, VoicevoxProvider,
        synthesize_test_tone_wav,
    };

    #[test]
    fn rfc6455_sha1_and_accept_key_match_spec_vectors() {
        assert_eq!(
            sha1_digest(b""),
            [
                0xda, 0x39, 0xa3, 0xee, 0x5e, 0x6b, 0x4b, 0x0d, 0x32, 0x55, 0xbf, 0xef, 0x95, 0x60,
                0x18, 0x90, 0xaf, 0xd8, 0x07, 0x09
            ]
        );
        assert_eq!(
            sha1_digest(b"abc"),
            [
                0xa9, 0x99, 0x3e, 0x36, 0x47, 0x06, 0x81, 0x6a, 0xba, 0x3e, 0x25, 0x71, 0x78, 0x50,
                0xc2, 0x6c, 0x9c, 0xd0, 0xd8, 0x9d
            ]
        );
        // RFC 6455 Section 1.3 handshake test vector.
        assert_eq!(
            compute_ws_accept_key("dGhlIHNhbXBsZSBub25jZQ=="),
            "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="
        );
    }

    async fn http_get(addr: SocketAddr, path: &str) -> (String, Vec<u8>) {
        let mut stream = TcpStream::connect(addr).await.unwrap();
        let req = format!("GET {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n");
        stream.write_all(req.as_bytes()).await.unwrap();
        let mut raw = Vec::new();
        stream.read_to_end(&mut raw).await.unwrap();
        let split = find_header_end(&raw).expect("valid HTTP response headers");
        let head = String::from_utf8_lossy(&raw[..split]).into_owned();
        let body = raw[split + 4..].to_vec();
        (head, body)
    }

    async fn connect_test_ws(addr: SocketAddr) -> TcpStream {
        let mut stream = TcpStream::connect(addr).await.unwrap();
        let client_key = "dGhlIHNhbXBsZSBub25jZQ==";
        let req = format!(
            "GET /ws HTTP/1.1\r\n\
             Host: {addr}\r\n\
             Upgrade: websocket\r\n\
             Connection: Upgrade\r\n\
             Sec-WebSocket-Key: {client_key}\r\n\
             Sec-WebSocket-Version: 13\r\n\r\n"
        );
        stream.write_all(req.as_bytes()).await.unwrap();
        let mut buf = Vec::new();
        let mut byte = [0_u8; 1];
        loop {
            stream.read_exact(&mut byte).await.unwrap();
            buf.push(byte[0]);
            if buf.ends_with(b"\r\n\r\n") {
                break;
            }
        }
        let head = String::from_utf8(buf).unwrap();
        assert!(head.starts_with("HTTP/1.1 101 Switching Protocols"));
        assert!(head.contains("Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo="));
        stream
    }

    #[tokio::test]
    async fn avatar_stage_serves_html_health_and_vrm_over_http() {
        let temp_dir = std::env::temp_dir();
        let vrm_file = temp_dir.join(format!("jiku-test-avatar-{}.vrm", std::process::id()));
        let fake_glb = b"glTF\x02\x00\x00\x00\x14\x00\x00\x00\x00\x00\x00\x00JSON";
        tokio::fs::write(&vrm_file, fake_glb).await.unwrap();

        let cfg = AvatarStageConfig {
            bind_addr: "127.0.0.1:0".to_string(),
            vrm_path: Some(vrm_file.clone()),
            companion_name: "JikuTest".to_string(),
        };
        let handle = spawn_avatar_stage(cfg, AvatarStageSnapshot::default())
            .await
            .unwrap();
        let addr = handle.local_addr();

        let (head_html, body_html) = http_get(addr, "/?captions=0").await;
        assert!(head_html.starts_with("HTTP/1.1 200 OK"));
        assert!(head_html.contains("Content-Type: text/html; charset=utf-8"));
        let html = String::from_utf8(body_html).unwrap();
        assert!(html.contains("background: transparent"));
        assert!(html.contains("waiting_human"));
        assert!(html.contains("/avatar.vrm"));
        assert!(html.contains("gaze_uv"));

        let (head_health, body_health) = http_get(addr, "/health").await;
        assert!(head_health.starts_with("HTTP/1.1 200 OK"));
        let health: Value = serde_json::from_slice(&body_health).unwrap();
        assert_eq!(health["status"], "ok");
        assert_eq!(health["companion_name"], "JikuTest");
        assert_eq!(health["vrm_available"], true);

        let (head_vrm, body_vrm) = http_get(addr, "/avatar.vrm").await;
        assert!(head_vrm.starts_with("HTTP/1.1 200 OK"));
        assert!(head_vrm.contains("Content-Type: model/gltf-binary"));
        assert_eq!(body_vrm, fake_glb);

        let (head_404, _) = http_get(addr, "/does-not-exist").await;
        assert!(head_404.starts_with("HTTP/1.1 404 Not Found"));

        handle.shutdown().await.unwrap();
        let _ = tokio::fs::remove_file(&vrm_file).await;
    }

    #[tokio::test]
    async fn avatar_stage_ws_streams_snapshot_cues_gaze_and_barge_in() {
        let cfg = AvatarStageConfig {
            bind_addr: "127.0.0.1:0".to_string(),
            vrm_path: None,
            companion_name: "JikuLive".to_string(),
        };
        let handle = spawn_avatar_stage(cfg, AvatarStageSnapshot::default())
            .await
            .unwrap();
        let addr = handle.local_addr();
        let tx = handle.sender().clone();

        let mut ws = connect_test_ws(addr).await;

        // 1. Initial snapshot frame.
        let f1 = read_ws_frame(&mut ws, MAX_WS_INCOMING_FRAME_BYTES)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(f1.opcode, 0x1);
        let msg1: AvatarStageMessage = serde_json::from_slice(&f1.payload).unwrap();
        assert!(
            matches!(msg1, AvatarStageMessage::Snapshot(ref s) if s.companion_name == "JikuLive")
        );

        // 2. Broadcast Cue, EnvelopeFrame, Gaze, and BargeInReset.
        tx.send(AvatarStageMessage::Cue {
            cue: AvatarCue {
                emotion: AvatarEmotion::Speaking,
                mora_envelope: vec![MoraViseme {
                    vowel: "a".to_string(),
                    start_ms: 0,
                    duration_ms: 140,
                }],
                ..AvatarCue::default()
            },
            caption: Some("Checking layout".to_string()),
            speaking: true,
        })
        .unwrap();

        let f2 = read_ws_frame(&mut ws, MAX_WS_INCOMING_FRAME_BYTES)
            .await
            .unwrap()
            .unwrap();
        let msg2: AvatarStageMessage = serde_json::from_slice(&f2.payload).unwrap();
        assert!(matches!(
            msg2,
            AvatarStageMessage::Cue { speaking: true, .. }
        ));

        tx.send(AvatarStageMessage::EnvelopeFrame {
            frame: AudioEnvelopeFrame {
                rms: 0.42,
                low_band: 0.25,
                mid_band: 0.35,
                high_band: 0.12,
            },
            vowel: Some("a".to_string()),
        })
        .unwrap();
        let f3 = read_ws_frame(&mut ws, MAX_WS_INCOMING_FRAME_BYTES)
            .await
            .unwrap()
            .unwrap();
        let msg3: AvatarStageMessage = serde_json::from_slice(&f3.payload).unwrap();
        assert!(matches!(
            msg3,
            AvatarStageMessage::EnvelopeFrame {
                vowel: Some(ref v),
                ..
            } if v == "a"
        ));

        tx.send(AvatarStageMessage::Gaze { u: 0.85, v: 0.25 })
            .unwrap();
        let f4 = read_ws_frame(&mut ws, MAX_WS_INCOMING_FRAME_BYTES)
            .await
            .unwrap()
            .unwrap();
        let msg4: AvatarStageMessage = serde_json::from_slice(&f4.payload).unwrap();
        assert!(
            matches!(msg4, AvatarStageMessage::Gaze { u, v } if (u - 0.85).abs() < 1e-5 && (v - 0.25).abs() < 1e-5)
        );

        tx.send(AvatarStageMessage::BargeInReset {
            emotion: AvatarEmotion::Idle,
        })
        .unwrap();
        let f5 = read_ws_frame(&mut ws, MAX_WS_INCOMING_FRAME_BYTES)
            .await
            .unwrap()
            .unwrap();
        let msg5: AvatarStageMessage = serde_json::from_slice(&f5.payload).unwrap();
        assert!(matches!(
            msg5,
            AvatarStageMessage::BargeInReset {
                emotion: AvatarEmotion::Idle
            }
        ));

        // 3. Client Ping -> Server Pong.
        write_ws_frame(&mut ws, 0x9, b"hi", Some([1, 2, 3, 4]))
            .await
            .unwrap();
        let pong = read_ws_frame(&mut ws, MAX_WS_INCOMING_FRAME_BYTES)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(pong.opcode, 0xA);
        assert_eq!(pong.payload, b"hi");

        // 4. Client snapshot_request -> reflects latest state (Idle + gaze + cleared speaking).
        write_ws_frame(
            &mut ws,
            0x1,
            br#"{"type":"snapshot_request"}"#,
            Some([9, 8, 7, 6]),
        )
        .await
        .unwrap();
        let resync = read_ws_frame(&mut ws, MAX_WS_INCOMING_FRAME_BYTES)
            .await
            .unwrap()
            .unwrap();
        let resync_msg: AvatarStageMessage = serde_json::from_slice(&resync.payload).unwrap();
        match resync_msg {
            AvatarStageMessage::Snapshot(s) => {
                assert_eq!(s.cue.emotion, AvatarEmotion::Idle);
                assert!(!s.speaking);
                assert!(s.cue.interrupted);
                assert_eq!(s.cue.gaze_uv, Some((0.85, 0.25)));
            }
            other => panic!("expected Snapshot resync, got {other:?}"),
        }

        let _ = write_ws_frame(&mut ws, 0x8, &[], Some([4, 3, 2, 1])).await;
        handle.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn tts_http_providers_populate_wav_timelines_against_mock_server() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let wav = synthesize_test_tone_wav(120, 440.0);
        let wav_clone = wav.clone();

        let server = tokio::spawn(async move {
            for _ in 0..5 {
                let Ok((mut stream, _)) = listener.accept().await else {
                    break;
                };
                let mut req_buf = vec![0_u8; 4096];
                let n = stream.read(&mut req_buf).await.unwrap();
                let req_str = String::from_utf8_lossy(&req_buf[..n]);
                let first_line = req_str.lines().next().unwrap_or("");
                if first_line.contains("/audio_query") {
                    let json_body = serde_json::json!({
                        "speedScale": 1.0,
                        "prePhonemeLength": 0.02,
                        "accent_phrases": [{
                            "moras": [{ "vowel": "a", "consonant_length": 0.02, "vowel_length": 0.08 }]
                        }]
                    })
                    .to_string();
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        json_body.len(),
                        json_body
                    );
                    stream.write_all(resp.as_bytes()).await.unwrap();
                } else {
                    let head = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: audio/wav\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        wav_clone.len()
                    );
                    stream.write_all(head.as_bytes()).await.unwrap();
                    stream.write_all(&wav_clone).await.unwrap();
                }
            }
        });

        let base = format!("http://{addr}");
        let irodori = IrodoriHttpProvider::new(format!("{base}/synthesize")).unwrap();
        let iro_res = irodori
            .synthesize("準備できました", AvatarEmotion::Success)
            .await
            .unwrap();
        assert_eq!(iro_res.provider, "irodori");
        assert_eq!(iro_res.duration_ms, 120);
        assert_eq!(iro_res.envelope_timeline.len(), 6);
        assert!(iro_res.cue.pcm_envelope.is_some());
        assert!(!iro_res.cue.mora_envelope.is_empty());

        let iro_openai = IrodoriHttpProvider::new(&base)
            .unwrap()
            .with_caption("soft cheerful tone");
        let iro_oai_res = iro_openai
            .synthesize("準備できました", AvatarEmotion::Thinking)
            .await
            .unwrap();
        assert_eq!(iro_oai_res.duration_ms, 120);
        assert!(iro_oai_res.cue.pcm_envelope.is_some());

        let openai = OpenAiSpeechProvider::new(&base, "", "gpt-4o-mini-tts", "alloy").unwrap();
        let oai_res = openai
            .synthesize("Hello", AvatarEmotion::Speaking)
            .await
            .unwrap();
        assert_eq!(oai_res.duration_ms, 120);
        assert!(!oai_res.envelope_timeline.is_empty());
        assert!(oai_res.cue.pcm_envelope.is_some());
        assert!(!oai_res.cue.mora_envelope.is_empty());

        let voicevox = VoicevoxProvider::new(&base, 3).unwrap();
        let vv_res = voicevox
            .synthesize("おはよう", AvatarEmotion::Neutral)
            .await
            .unwrap();
        assert_eq!(vv_res.duration_ms, 120);
        assert!(!vv_res.envelope_timeline.is_empty());
        assert!(vv_res.cue.pcm_envelope.is_some());
        assert_eq!(vv_res.cue.mora_envelope[0].vowel, "a");

        server.await.unwrap();
    }
}
