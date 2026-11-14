//! Stateful full-duplex WebSocket driver (`LiveSessionDriver`, `JIK-0006`) for
//! `BidiGenerateContent` audio/video co-piloting, barge-in buffer flushing,
//! local Epoch gating, and transparent `sessionResumption` / `GoAway` reconnects.

use std::time::{Duration, Instant};

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64_STD};
use futures_util::{SinkExt as _, StreamExt as _};
use jiku_tts::{AvatarCue, AvatarEmotion, compute_pcm_envelope};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::net::TcpStream;
use tokio_tungstenite::{
    MaybeTlsStream, WebSocketStream, connect_async, tungstenite::protocol::Message,
};

use crate::{
    LiveResponseModality, LiveSessionConfig, ToolResponseScheduling, drivers::DriverError,
};

/// Server event decoded from a `BidiGenerateContent` WebSocket stream.
#[derive(Debug, Clone, PartialEq)]
pub enum LiveServerEvent {
    /// Gateway acknowledged the `setup` frame.
    SetupComplete,
    /// User speech barge-in interrupted the current model turn; callers must
    /// immediately flush queued downlink audio and reset lip-sync state.
    BargeInInterrupted {
        /// Zeroed companion reset cue.
        reset_cue: AvatarCue,
    },
    /// Downlink `24 kHz` PCM audio chunk paired with its computed lip-sync cue.
    DownlinkAudioChunk {
        /// Raw signed 16-bit LE PCM bytes.
        pcm_bytes: Vec<u8>,
        /// Extracted RMS + frequency band lip-sync cue.
        cue: AvatarCue,
    },
    /// Plain-text transcript or assistant text delta.
    TextDelta {
        /// Transcript text fragment.
        text: String,
    },
    /// Model requested an asynchronous tool call (`NON_BLOCKING` or `Blocking`).
    ToolCall {
        /// Tool call ID returned with `toolResponse`.
        call_id: String,
        /// Declared tool function name (e.g. `"koma_click"`).
        name: String,
        /// Structured JSON arguments.
        args: Value,
    },
    /// Resumption handle pushed by the server across ~10-minute WS lifetimes.
    SessionResumptionUpdate {
        /// New opaque handle to pass on reconnect.
        new_handle: String,
        /// Whether the session is resumable.
        resumable: bool,
    },
    /// Gateway announced imminent WebSocket shutdown (`time_left_ms`).
    GoAway {
        /// Remaining time before disconnect, if specified.
        time_left_ms: Option<u64>,
    },
}

/// Asynchronous live WebSocket session handle managing rate-limited screen frames,
/// Epoch-fenced barge-in state, and resumption metadata.
pub struct LiveSessionConnection {
    ws: WebSocketStream<MaybeTlsStream<TcpStream>>,
    config: LiveSessionConfig,
    last_screen_sent: Option<Instant>,
    resumption_handle: Option<String>,
    interrupted_generation: u64,
    downlink_pcm_queue: Vec<u8>,
}

impl LiveSessionConnection {
    /// Builds the initial `BidiGenerateContent` `setup` payload matching [`LiveSessionConfig`].
    pub fn build_setup_payload(
        config: &LiveSessionConfig,
        resumption_handle: Option<&str>,
    ) -> Result<Value, DriverError> {
        config
            .validate()
            .map_err(|msg| DriverError::Validation(msg.to_string()))?;

        let modality = match config.response_modality {
            LiveResponseModality::Audio => "AUDIO",
            LiveResponseModality::Text => "TEXT",
        };
        let mut generation_config = json!({
            "responseModalities": [modality],
        });
        if config.enable_affective_dialog
            && let Some(obj) = generation_config.as_object_mut()
        {
            obj.insert("enableAffectiveDialog".to_string(), json!(true));
        }

        let mut setup = json!({
            "model": format!("models/{}", config.model),
            "generationConfig": generation_config,
            "contextWindowCompression": {
                "triggerTokens": config.compression.trigger_tokens.to_string(),
                "slidingWindow": {
                    "targetTokens": config.compression.sliding_window_target_tokens.to_string()
                }
            },
            "sessionResumption": if let Some(handle) = resumption_handle {
                json!({ "handle": handle })
            } else {
                json!({})
            }
        });

        if config.output_audio_transcription
            && let Some(obj) = setup.as_object_mut()
        {
            obj.insert("outputAudioTranscription".to_string(), json!({}));
        }
        if config.proactive_audio
            && let Some(obj) = setup.as_object_mut()
        {
            obj.insert("proactivity".to_string(), json!({ "proactiveAudio": true }));
        }

        Ok(json!({ "setup": setup }))
    }

    /// Connects over `ws://` or `wss://` (using `rustls`), sends the `setup` frame,
    /// and returns the active [`LiveSessionConnection`].
    pub async fn connect(
        ws_url: &str,
        config: LiveSessionConfig,
        resumption_handle: Option<String>,
    ) -> Result<Self, DriverError> {
        config
            .validate()
            .map_err(|msg| DriverError::Validation(msg.to_string()))?;

        let setup_json = Self::build_setup_payload(&config, resumption_handle.as_deref())?;
        let (mut ws, _) = connect_async(ws_url).await.map_err(|e| {
            DriverError::WebSocket(format!("WS handshake failed for {ws_url}: {e}"))
        })?;

        ws.send(Message::Text(setup_json.to_string().into()))
            .await
            .map_err(|e| DriverError::WebSocket(format!("WS setup send failed: {e}")))?;

        Ok(Self {
            ws,
            config,
            last_screen_sent: None,
            resumption_handle,
            interrupted_generation: 0,
            downlink_pcm_queue: Vec::new(),
        })
    }

    /// Streams raw uplink PCM (`16 kHz` default) inside `realtimeInput.mediaChunks`.
    pub async fn send_audio_pcm(&mut self, pcm_le_i16: &[u8]) -> Result<(), DriverError> {
        let mime = format!("audio/pcm;rate={}", self.config.uplink_sample_rate_hz);
        let msg = json!({
            "realtimeInput": {
                "mediaChunks": [{
                    "mimeType": mime,
                    "data": BASE64_STD.encode(pcm_le_i16),
                }]
            }
        });
        self.ws
            .send(Message::Text(msg.to_string().into()))
            .await
            .map_err(|e| DriverError::WebSocket(format!("WS audio send failed: {e}")))
    }

    /// Streams a visual workspace frame when elapsed time satisfies `config.max_screen_fps`,
    /// returning `Ok(false)` when throttled cleanly.
    pub async fn send_screen_frame(
        &mut self,
        mime_type: &str,
        encoded_bytes: &[u8],
    ) -> Result<bool, DriverError> {
        let min_interval = Duration::from_secs_f64(1.0 / f64::from(self.config.max_screen_fps));
        let now = Instant::now();
        if let Some(prev) = self.last_screen_sent
            && now.duration_since(prev) < min_interval
        {
            return Ok(false);
        }
        self.last_screen_sent = Some(now);

        let msg = json!({
            "realtimeInput": {
                "mediaChunks": [{
                    "mimeType": mime_type,
                    "data": BASE64_STD.encode(encoded_bytes),
                }]
            }
        });
        self.ws
            .send(Message::Text(msg.to_string().into()))
            .await
            .map_err(|e| DriverError::WebSocket(format!("WS screen frame send failed: {e}")))?;
        Ok(true)
    }

    /// Returns a `toolResponse` packet tagged with [`ToolResponseScheduling`] and an
    /// `epoch` fence summary.
    pub async fn send_tool_response(
        &mut self,
        call_id: &str,
        name: &str,
        scheduling: ToolResponseScheduling,
        result_payload: Value,
    ) -> Result<(), DriverError> {
        let sched_str = match scheduling {
            ToolResponseScheduling::Interrupt => "INTERRUPT",
            ToolResponseScheduling::WhenIdle => "WHEN_IDLE",
            ToolResponseScheduling::Silent => "SILENT",
        };
        let msg = json!({
            "toolResponse": {
                "functionResponses": [{
                    "id": call_id,
                    "name": name,
                    "scheduling": sched_str,
                    "response": result_payload,
                }]
            }
        });
        self.ws
            .send(Message::Text(msg.to_string().into()))
            .await
            .map_err(|e| DriverError::WebSocket(format!("WS toolResponse send failed: {e}")))
    }

    /// Reads and parses the next WebSocket control/content event, updating local
    /// barge-in buffers and resumption handle automatically.
    pub async fn recv_event(&mut self) -> Result<Option<LiveServerEvent>, DriverError> {
        while let Some(frame_res) = self.ws.next().await {
            let msg = frame_res.map_err(|e| DriverError::WebSocket(e.to_string()))?;
            let text = match msg {
                Message::Text(t) => t.to_string(),
                Message::Binary(b) => String::from_utf8_lossy(&b).into_owned(),
                Message::Close(_) => return Ok(None),
                Message::Ping(_) | Message::Pong(_) | Message::Frame(_) => continue,
            };

            let val: Value = serde_json::from_str(&text)
                .map_err(|e| DriverError::InvalidPayload(format!("invalid WS JSON: {e}")))?;

            if val.get("setupComplete").is_some() {
                return Ok(Some(LiveServerEvent::SetupComplete));
            }
            if let Some(upd) = val.get("sessionResumptionUpdate") {
                let handle = upd
                    .get("newHandle")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let resumable = upd
                    .get("resumable")
                    .and_then(Value::as_bool)
                    .unwrap_or(!handle.is_empty());
                if !handle.is_empty() {
                    self.resumption_handle = Some(handle.clone());
                }
                return Ok(Some(LiveServerEvent::SessionResumptionUpdate {
                    new_handle: handle,
                    resumable,
                }));
            }
            if let Some(go_away) = val.get("goAway") {
                let ms = go_away
                    .get("timeLeft")
                    .and_then(Value::as_str)
                    .and_then(|s| s.trim_end_matches('s').parse::<f64>().ok())
                    .map(|secs| (secs * 1000.0).max(0.0) as u64);
                return Ok(Some(LiveServerEvent::GoAway { time_left_ms: ms }));
            }
            if let Some(tc) = val.get("toolCall")
                && let Some(first) = tc
                    .get("functionCalls")
                    .and_then(Value::as_array)
                    .and_then(|arr| arr.first())
            {
                let call_id = first
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or("call_0")
                    .to_string();
                let name = first
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown")
                    .to_string();
                let args = first.get("args").cloned().unwrap_or(Value::Null);
                return Ok(Some(LiveServerEvent::ToolCall {
                    call_id,
                    name,
                    args,
                }));
            }
            if let Some(sc) = val.get("serverContent") {
                if sc.get("interrupted").and_then(Value::as_bool) == Some(true) {
                    self.downlink_pcm_queue.clear();
                    self.interrupted_generation += 1;
                    return Ok(Some(LiveServerEvent::BargeInInterrupted {
                        reset_cue: AvatarCue {
                            emotion: AvatarEmotion::Idle,
                            interrupted: true,
                            ..AvatarCue::default()
                        },
                    }));
                }
                if let Some(parts) = sc
                    .get("modelTurn")
                    .and_then(|mt| mt.get("parts"))
                    .and_then(Value::as_array)
                {
                    for part in parts {
                        if let Some(inline) = part.get("inlineData")
                            && let Some(b64) = inline.get("data").and_then(Value::as_str)
                            && let Ok(pcm_bytes) = BASE64_STD.decode(b64)
                        {
                            self.downlink_pcm_queue.extend_from_slice(&pcm_bytes);
                            let env = compute_pcm_envelope(&pcm_bytes).ok();
                            return Ok(Some(LiveServerEvent::DownlinkAudioChunk {
                                pcm_bytes,
                                cue: AvatarCue {
                                    emotion: AvatarEmotion::Speaking,
                                    pcm_envelope: env,
                                    ..AvatarCue::default()
                                },
                            }));
                        }
                        if let Some(t) = part.get("text").and_then(Value::as_str) {
                            return Ok(Some(LiveServerEvent::TextDelta {
                                text: t.to_string(),
                            }));
                        }
                    }
                }
                if let Some(t) = sc
                    .get("outputTranscription")
                    .and_then(|ot| ot.get("text"))
                    .and_then(Value::as_str)
                {
                    return Ok(Some(LiveServerEvent::TextDelta {
                        text: t.to_string(),
                    }));
                }
            }
        }
        Ok(None)
    }

    /// Returns the latest `SessionResumptionUpdate` handle if issued.
    #[must_use]
    pub fn resumption_handle(&self) -> Option<&str> {
        self.resumption_handle.as_deref()
    }

    /// Returns the count of barge-in queue flushes performed on this connection.
    #[must_use]
    pub const fn interrupted_generation(&self) -> u64 {
        self.interrupted_generation
    }

    /// Returns buffered downlink PCM bytes (emptied immediately on barge-in).
    #[must_use]
    pub fn queued_downlink_bytes(&self) -> usize {
        self.downlink_pcm_queue.len()
    }

    /// Closes the WebSocket cleanly.
    pub async fn close(mut self) -> Result<(), DriverError> {
        let _ = self.ws.close(None).await;
        Ok(())
    }
}

/// Summary snapshot of a `LiveSessionDriver` connection state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LiveSessionStatus {
    /// Active live model ID.
    pub model: String,
    /// Active resumption handle, if issued.
    pub resumption_handle: Option<String>,
    /// Barge-in generation counter.
    pub interrupted_generation: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::net::TcpListener;
    use tokio_tungstenite::accept_async;

    #[tokio::test]
    async fn live_ws_loopback_handles_setup_pcm_barge_in_tool_and_resumption() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server_task = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut ws = accept_async(stream).await.unwrap();

            // 1. Expect setup packet.
            let first = ws.next().await.unwrap().unwrap();
            let setup: Value = serde_json::from_str(first.to_text().unwrap()).unwrap();
            assert_eq!(setup["setup"]["model"], "models/gemini-2.5-flash-live");

            // Reply with setupComplete + audio chunk + resumption + interrupted + toolCall + goAway.
            ws.send(Message::Text(r#"{"setupComplete":{}}"#.into()))
                .await
                .unwrap();

            // Wait for uplink PCM chunk.
            let uplink = ws.next().await.unwrap().unwrap();
            assert!(uplink.to_text().unwrap().contains("audio/pcm;rate=16000"));

            let pcm_sample: Vec<u8> = [0x00, 0x20, 0x00, 0xE0].repeat(4);
            let b64 = BASE64_STD.encode(&pcm_sample);
            let chunk_msg = json!({
                "serverContent": {
                    "modelTurn": { "parts": [{ "inlineData": { "mimeType": "audio/pcm;rate=24000", "data": b64 } }] }
                }
            });
            ws.send(Message::Text(chunk_msg.to_string().into()))
                .await
                .unwrap();
            ws.send(Message::Text(
                r#"{"sessionResumptionUpdate":{"newHandle":"resume_tok_99","resumable":true}}"#
                    .into(),
            ))
            .await
            .unwrap();
            ws.send(Message::Text(
                r#"{"serverContent":{"interrupted":true}}"#.into(),
            ))
            .await
            .unwrap();
            ws.send(Message::Text(
                r#"{"toolCall":{"functionCalls":[{"id":"c1","name":"koma_click","args":{"grid":"G8"}}]}}"#.into(),
            ))
            .await
            .unwrap();

            let resp = ws.next().await.unwrap().unwrap();
            assert!(resp.to_text().unwrap().contains("WHEN_IDLE"));
            ws.send(Message::Text(r#"{"goAway":{"timeLeft":"1.5s"}}"#.into()))
                .await
                .unwrap();
        });

        let url = format!("ws://{addr}");
        let mut client = LiveSessionConnection::connect(&url, LiveSessionConfig::default(), None)
            .await
            .unwrap();
        assert!(matches!(
            client.recv_event().await.unwrap(),
            Some(LiveServerEvent::SetupComplete)
        ));

        client.send_audio_pcm(&[0, 1, 2, 3]).await.unwrap();

        match client.recv_event().await.unwrap() {
            Some(LiveServerEvent::DownlinkAudioChunk { pcm_bytes, cue }) => {
                assert_eq!(pcm_bytes.len(), 16);
                assert!(cue.pcm_envelope.unwrap().rms > 0.1);
                assert!(client.queued_downlink_bytes() > 0);
            }
            other => panic!("expected DownlinkAudioChunk, got {other:?}"),
        }

        assert!(matches!(
            client.recv_event().await.unwrap(),
            Some(LiveServerEvent::SessionResumptionUpdate { ref new_handle, resumable: true }) if new_handle == "resume_tok_99"
        ));
        assert_eq!(client.resumption_handle(), Some("resume_tok_99"));

        // Barge-in must flush queued audio immediately.
        assert!(matches!(
            client.recv_event().await.unwrap(),
            Some(LiveServerEvent::BargeInInterrupted { ref reset_cue }) if reset_cue.interrupted
        ));
        assert_eq!(client.queued_downlink_bytes(), 0);
        assert_eq!(client.interrupted_generation(), 1);

        match client.recv_event().await.unwrap() {
            Some(LiveServerEvent::ToolCall {
                call_id,
                name,
                args,
            }) => {
                assert_eq!(name, "koma_click");
                assert_eq!(args["grid"], "G8");
                client
                    .send_tool_response(
                        &call_id,
                        &name,
                        ToolResponseScheduling::WhenIdle,
                        json!({"epoch": 42, "ok": true}),
                    )
                    .await
                    .unwrap();
            }
            other => panic!("expected ToolCall, got {other:?}"),
        }

        assert!(matches!(
            client.recv_event().await.unwrap(),
            Some(LiveServerEvent::GoAway {
                time_left_ms: Some(1500)
            })
        ));
        client.close().await.unwrap();
        server_task.await.unwrap();
    }
}
