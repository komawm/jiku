//! JSON-RPC 2.0 Model Context Protocol (MCP) stdio bridge exposing Jiku tools.

use std::{
    io::{BufRead, Write},
    path::PathBuf,
};

use jikud::daemon::RpcDecisionQuestion;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::client::JikuClient;

/// MCP protocol revision advertised during `initialize`.
pub const MCP_PROTOCOL_VERSION: &str = "2024-11-05";
const MAX_MCP_MESSAGE_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum McpFraming {
    Ndjson,
    ContentLength,
}

#[derive(Debug, Deserialize)]
struct JsonRpcIncoming {
    #[serde(default)]
    jsonrpc: Option<String>,
    #[serde(default)]
    id: Option<Value>,
    method: String,
    #[serde(default)]
    params: Value,
}

#[derive(Debug, Serialize)]
struct JsonRpcOutgoing {
    jsonrpc: &'static str,
    id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcErrObj>,
}

#[derive(Debug, Serialize)]
struct JsonRpcErrObj {
    code: i32,
    message: String,
}

/// Returns the canonical MCP `tools/list` schema array for Jiku.
#[must_use]
pub fn tool_definitions() -> Value {
    json!([
        {
            "name": "jiku_status",
            "description": "Inspect jikud config, model driver bindings, memory DB counts, and live Koma connectivity.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "compact": { "type": "boolean", "default": true }
                },
                "additionalProperties": false
            }
        },
        {
            "name": "jiku_run_task",
            "description": "Execute a desktop orchestration goal inside Koma using DecisionRouter gating, TurnModel planning, and Epoch-fenced vseat calls.",
            "inputSchema": {
                "type": "object",
                "required": ["goal"],
                "properties": {
                    "goal": { "type": "string" },
                    "workspace": { "type": "string" },
                    "dry_run": { "type": "boolean" },
                    "force_emulate": { "type": "boolean" },
                    "compact": { "type": "boolean", "default": true }
                }
            }
        },
        {
            "name": "jiku_decide",
            "description": "Evaluate zero-decode calibrated Noul/Choice decisions with gemini-decision-1 (or structured planner fallback).",
            "inputSchema": {
                "type": "object",
                "required": ["prompt"],
                "properties": {
                    "prompt": { "type": "string" },
                    "image_path": { "type": "string" },
                    "force_emulate": { "type": "boolean" },
                    "questions": { "type": "array" },
                    "compact": { "type": "boolean", "default": true }
                }
            }
        },
        {
            "name": "jiku_plan",
            "description": "Generate a reasoning or visual-language plan using the configured TurnModelDriver (deepseek-v4.1-flash on OpenCode Go).",
            "inputSchema": {
                "type": "object",
                "required": ["prompt"],
                "properties": {
                    "prompt": { "type": "string" },
                    "image_path": { "type": "string" },
                    "system": { "type": "string" },
                    "json_object": { "type": "boolean" }
                }
            }
        },
        {
            "name": "jiku_box_classify",
            "description": "Run multimodal DecisionsDriver triage across a dropzone directory (default ~/JikuBox).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "directory": { "type": "string" },
                    "force_emulate": { "type": "boolean" }
                }
            }
        },
        {
            "name": "jiku_memory_search",
            "description": "Search persisted L2 episodes using SQLite FTS5 BM25 ranking.",
            "inputSchema": {
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": { "type": "string" },
                    "limit": { "type": "integer", "minimum": 1, "maximum": 100 }
                }
            }
        },
        {
            "name": "jiku_skills_list",
            "description": "List staged, approved, and rejected Lua/SOP skills awaiting or completing review.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "compact": { "type": "boolean", "default": true }
                },
                "additionalProperties": false
            }
        },
        {
            "name": "jiku_a11y_snapshot",
            "description": "Capture normalized AT-SPI UI tree with snapshot-scoped @snap:eN handles.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "skeleton": { "type": "boolean" },
                    "root_ref": { "type": "string" },
                    "app_id": { "type": "string" },
                    "window_title": { "type": "string" },
                    "compact": { "type": "boolean", "default": true }
                }
            }
        },
        {
            "name": "jiku_a11y_act",
            "description": "Invoke a semantic AT-SPI action (click, focus, set-value, type, read-value) on a @snap:eN element.",
            "inputSchema": {
                "type": "object",
                "required": ["action", "target_ref"],
                "properties": {
                    "action": { "type": "string" },
                    "target_ref": { "type": "string" },
                    "text": { "type": "string" },
                    "compact": { "type": "boolean", "default": true }
                }
            }
        },
        {
            "name": "jiku_tts_say",
            "description": "Synthesize companion speech (and optional WAV export/playback) updating AvatarCue lip-sync state.",
            "inputSchema": {
                "type": "object",
                "required": ["text"],
                "properties": {
                    "text": { "type": "string" },
                    "emotion": { "type": "string" },
                    "play": { "type": "boolean" },
                    "wav_out": { "type": "string" },
                    "compact": { "type": "boolean", "default": true }
                }
            }
        },
        {
            "name": "jiku_context_inspect",
            "description": "Inspect small-context token budgeting config, last compaction telemetry, and waterfall simulation.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "max_tokens": { "type": "integer", "minimum": 256 },
                    "sample_text": { "type": "string" },
                    "compact": { "type": "boolean", "default": true }
                }
            }
        },
        {
            "name": "jiku_bench_run",
            "description": "Run the 2x2 Decision x Grounding VirtualPaint ablation benchmark (offline deterministic or live VLM).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "offline": { "type": "boolean", "default": true },
                    "max_context_tokens": { "type": "integer", "minimum": 256 },
                    "out_dir": { "type": "string" },
                    "endpoint": { "type": "string" },
                    "model": { "type": "string" },
                    "provider": { "type": "string" },
                    "compact": { "type": "boolean", "default": true }
                }
            }
        }
    ])
}

/// Processes a single JSON-RPC 2.0 request line, returning `Some(serialized_response)`
/// when `id` is present or `None` for notifications.
pub async fn process_jsonrpc_line(client: &mut JikuClient, line: &str) -> Option<String> {
    let parsed: JsonRpcIncoming = match serde_json::from_str(line) {
        Ok(req) => req,
        Err(e) => {
            let err = JsonRpcOutgoing {
                jsonrpc: "2.0",
                id: Value::Null,
                result: None,
                error: Some(JsonRpcErrObj {
                    code: -32700,
                    message: format!("parse error: {e}"),
                }),
            };
            return serde_json::to_string(&err).ok();
        }
    };

    if parsed.jsonrpc.as_deref().is_some_and(|v| v != "2.0") {
        let err = JsonRpcOutgoing {
            jsonrpc: "2.0",
            id: parsed.id.unwrap_or(Value::Null),
            result: None,
            error: Some(JsonRpcErrObj {
                code: -32600,
                message: "invalid jsonrpc version (expected 2.0)".to_string(),
            }),
        };
        return serde_json::to_string(&err).ok();
    }

    let Some(id) = parsed.id else {
        // Notification (e.g. `notifications/initialized`) requires no response frame.
        return None;
    };

    let resp = match parsed.method.as_str() {
        "initialize" => JsonRpcOutgoing {
            jsonrpc: "2.0",
            id,
            result: Some(json!({
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {
                    "tools": { "listChanged": false },
                    "resources": { "subscribe": false, "listChanged": false },
                    "prompts": { "listChanged": false }
                },
                "serverInfo": {
                    "name": "jiku",
                    "version": env!("CARGO_PKG_VERSION"),
                }
            })),
            error: None,
        },
        "ping" => JsonRpcOutgoing {
            jsonrpc: "2.0",
            id,
            result: Some(json!({})),
            error: None,
        },
        "tools/list" => JsonRpcOutgoing {
            jsonrpc: "2.0",
            id,
            result: Some(json!({ "tools": tool_definitions() })),
            error: None,
        },
        "resources/list" => JsonRpcOutgoing {
            jsonrpc: "2.0",
            id,
            result: Some(json!({ "resources": [] })),
            error: None,
        },
        "resources/templates/list" => JsonRpcOutgoing {
            jsonrpc: "2.0",
            id,
            result: Some(json!({ "resourceTemplates": [] })),
            error: None,
        },
        "prompts/list" => JsonRpcOutgoing {
            jsonrpc: "2.0",
            id,
            result: Some(json!({ "prompts": [] })),
            error: None,
        },
        "tools/call" => {
            let tool_name = parsed
                .params
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("");
            let args = parsed
                .params
                .get("arguments")
                .cloned()
                .unwrap_or_else(|| json!({}));

            match call_tool(client, tool_name, &args).await {
                Ok(val) => {
                    let text = serde_json::to_string_pretty(&val).unwrap_or_default();
                    JsonRpcOutgoing {
                        jsonrpc: "2.0",
                        id,
                        result: Some(json!({
                            "content": [{ "type": "text", "text": text }],
                            "structuredContent": val,
                            "isError": false
                        })),
                        error: None,
                    }
                }
                Err(e) => JsonRpcOutgoing {
                    jsonrpc: "2.0",
                    id,
                    result: Some(json!({
                        "content": [{ "type": "text", "text": format!("{e:#}") }],
                        "isError": true
                    })),
                    error: None,
                },
            }
        }
        other => JsonRpcOutgoing {
            jsonrpc: "2.0",
            id,
            result: None,
            error: Some(JsonRpcErrObj {
                code: -32601,
                message: format!("method not found: `{other}`"),
            }),
        },
    };

    serde_json::to_string(&resp).ok()
}

async fn call_tool(client: &mut JikuClient, name: &str, args: &Value) -> anyhow::Result<Value> {
    let compact = args.get("compact").and_then(Value::as_bool).unwrap_or(true);

    match name {
        "jiku_status" => {
            let status = client.status().await?;
            if compact {
                Ok(json!({
                    "planner_provider": status.planner_provider,
                    "planner_model": status.planner_model,
                    "decision_mode": status.decision_mode,
                    "koma_connected": status.koma_connected,
                    "episode_count": status.episode_count,
                    "skill_count": status.skill_count,
                    "tts_enabled": status.tts_enabled,
                    "tts_provider": status.tts_provider,
                }))
            } else {
                Ok(serde_json::to_value(status)?)
            }
        }
        "jiku_run_task" => {
            let goal = args.get("goal").and_then(Value::as_str).unwrap_or("");
            let ws = args.get("workspace").and_then(Value::as_str);
            let dry_run = args
                .get("dry_run")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let force_emulate = args
                .get("force_emulate")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let report = client.run_task(ws, goal, dry_run, force_emulate).await?;
            if compact {
                Ok(json!({
                    "workspace": report.workspace,
                    "route": report.route,
                    "planner_summary": report.planner_summary,
                    "steps": report.executed_steps.len(),
                    "halt_reason": report.halt_reason,
                    "duration_ms": report.duration_ms,
                }))
            } else {
                Ok(serde_json::to_value(report)?)
            }
        }
        "jiku_decide" => {
            let prompt = args.get("prompt").and_then(Value::as_str).unwrap_or("");
            let img = args
                .get("image_path")
                .and_then(Value::as_str)
                .map(PathBuf::from);
            let force_emulate = args
                .get("force_emulate")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let questions: Vec<RpcDecisionQuestion> = args
                .get("questions")
                .cloned()
                .map(serde_json::from_value)
                .transpose()?
                .unwrap_or_default();
            let resp = client
                .decide(img.as_deref(), prompt, &questions, force_emulate)
                .await?;
            if compact {
                Ok(json!({
                    "execution_mode": resp.execution_mode,
                    "answers": resp.answers,
                }))
            } else {
                Ok(serde_json::to_value(resp)?)
            }
        }
        "jiku_plan" => {
            let prompt = args.get("prompt").and_then(Value::as_str).unwrap_or("");
            let img = args
                .get("image_path")
                .and_then(Value::as_str)
                .map(PathBuf::from);
            let system = args.get("system").and_then(Value::as_str);
            let json_object = args
                .get("json_object")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            Ok(serde_json::to_value(
                client
                    .plan(img.as_deref(), system, prompt, json_object, false)
                    .await?,
            )?)
        }
        "jiku_box_classify" => {
            let dir = args
                .get("directory")
                .and_then(Value::as_str)
                .map(PathBuf::from);
            let force_emulate = args
                .get("force_emulate")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            Ok(serde_json::to_value(
                client.box_classify(dir.as_deref(), force_emulate).await?,
            )?)
        }
        "jiku_memory_search" => {
            let query = args.get("query").and_then(Value::as_str).unwrap_or("");
            let limit = args.get("limit").and_then(Value::as_u64).unwrap_or(10) as usize;
            Ok(serde_json::to_value(
                client.memory_search(query, limit).await?,
            )?)
        }
        "jiku_skills_list" => {
            let list = client.skills_list().await?;
            if compact {
                let slim: Vec<Value> = list
                    .into_iter()
                    .map(|s| {
                        json!({
                            "skill_id": s.skill_id,
                            "status": s.status,
                            "title": s.title,
                        })
                    })
                    .collect();
                Ok(json!(slim))
            } else {
                Ok(serde_json::to_value(list)?)
            }
        }
        "jiku_a11y_snapshot" => {
            let skeleton = args
                .get("skeleton")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let root_ref = args.get("root_ref").and_then(Value::as_str);
            let app_id = args.get("app_id").and_then(Value::as_str).unwrap_or("");
            let window_title = args
                .get("window_title")
                .and_then(Value::as_str)
                .unwrap_or("");
            let res = client
                .a11y_snapshot(skeleton, root_ref, app_id, window_title)
                .await?;
            if compact {
                Ok(json!({
                    "snapshot_id": res["snapshot"]["snapshot_id"],
                    "entries_count": res["snapshot"]["entries"].as_array().map_or(0, Vec::len),
                    "truncated": res["snapshot"]["truncated"],
                    "text_tree": res["text_tree"],
                }))
            } else {
                Ok(res)
            }
        }
        "jiku_a11y_act" => {
            let action = args.get("action").and_then(Value::as_str).unwrap_or("");
            let target_ref = args.get("target_ref").and_then(Value::as_str).unwrap_or("");
            let text = args.get("text").and_then(Value::as_str);
            let res = client.a11y_act(action, target_ref, text).await?;
            if compact {
                Ok(json!({
                    "ok": res["ok"],
                    "ref_id": res["ref_id"],
                    "action": res["action"],
                    "value": res["value"],
                    "message": res["message"],
                }))
            } else {
                Ok(res)
            }
        }
        "jiku_tts_say" => {
            let text = args.get("text").and_then(Value::as_str).unwrap_or("");
            let emotion = args.get("emotion").and_then(Value::as_str);
            let play = args.get("play").and_then(Value::as_bool).unwrap_or(true);
            let wav_out = args
                .get("wav_out")
                .and_then(Value::as_str)
                .map(PathBuf::from);
            let res = client
                .tts_say(text, emotion, play, wav_out.as_deref())
                .await?;
            if compact {
                Ok(json!({
                    "ok": res["ok"],
                    "provider": res["provider"],
                    "duration_ms": res["duration_ms"],
                    "wav_bytes_len": res["wav_bytes_len"],
                    "played": res["played"],
                    "wav_out": res["wav_out"],
                }))
            } else {
                Ok(res)
            }
        }
        "jiku_context_inspect" => {
            let max_tokens = args
                .get("max_tokens")
                .and_then(Value::as_u64)
                .and_then(|v| u32::try_from(v).ok());
            let sample_text = args.get("sample_text").and_then(Value::as_str);
            let res = client.context_inspect(max_tokens, sample_text).await?;
            if compact {
                Ok(json!({
                    "simulated_report": res["simulated_report"],
                    "last_report": res["last_report"],
                }))
            } else {
                Ok(res)
            }
        }
        "jiku_bench_run" => {
            let offline = args.get("offline").and_then(Value::as_bool).unwrap_or(true);
            let max_context_tokens = args
                .get("max_context_tokens")
                .and_then(Value::as_u64)
                .and_then(|v| u32::try_from(v).ok());
            let out_dir = args
                .get("out_dir")
                .and_then(Value::as_str)
                .map(PathBuf::from);
            let endpoint = args.get("endpoint").and_then(Value::as_str);
            let model = args.get("model").and_then(Value::as_str);
            let provider = args.get("provider").and_then(Value::as_str);
            let res = client
                .bench_run(
                    offline,
                    max_context_tokens,
                    out_dir.as_deref(),
                    provider,
                    model,
                    endpoint,
                )
                .await?;
            if compact {
                let cells: Vec<Value> = res["report"]["cells"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .map(|c| {
                                json!({
                                    "label": c["label"],
                                    "pass_count": c["pass_count"],
                                    "total_tasks": c["total_tasks"],
                                    "avg_hit_error_px": c["avg_hit_error_px"],
                                    "avg_chamfer_error_px": c["avg_chamfer_error_px"],
                                    "total_wasted_strokes": c["total_wasted_strokes"],
                                })
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                Ok(json!({
                    "executor": res["report"]["executor_name"],
                    "cells": cells,
                    "artifacts": res["artifacts"],
                }))
            } else {
                Ok(res)
            }
        }
        other => anyhow::bail!("unknown MCP tool `{other}`"),
    }
}

/// Runs the dual-framing (`Content-Length` HTTP headers or NDJSON `{...}` auto-detected
/// per message) JSON-RPC 2.0 MCP loop reading from `reader` and writing to `writer`.
pub async fn run_mcp_stdio<R: BufRead, W: Write>(
    client: &mut JikuClient,
    reader: R,
    writer: W,
    stateless: bool,
) -> anyhow::Result<()> {
    run_mcp_loop(client, reader, writer, stateless).await
}

/// Runs the JSON-RPC 2.0 MCP stdio loop with per-message framing auto-detection
/// (supporting both `Content-Length` HTTP header blocks and NDJSON lines).
pub async fn run_mcp_loop<R: BufRead, W: Write>(
    client: &mut JikuClient,
    mut reader: R,
    mut writer: W,
    stateless: bool,
) -> anyhow::Result<()> {
    while let Some((framing, payload)) = read_next_mcp_frame(&mut reader)? {
        if let Some(reply) = process_jsonrpc_line(client, payload.trim()).await {
            match framing {
                McpFraming::Ndjson => {
                    writeln!(writer, "{reply}")?;
                }
                McpFraming::ContentLength => {
                    write!(writer, "Content-Length: {}\r\n\r\n{}", reply.len(), reply)?;
                }
            }
            writer.flush()?;
        }
        if stateless && payload.contains("\"tools/call\"") {
            // Keep draining until EOF so multi-message stateless pipes work cleanly.
            continue;
        }
    }
    Ok(())
}

fn read_next_mcp_frame<R: BufRead>(reader: &mut R) -> anyhow::Result<Option<(McpFraming, String)>> {
    let mut line = String::new();
    loop {
        line.clear();
        let bytes = reader.read_line(&mut line)?;
        if bytes == 0 {
            return Ok(None);
        }
        if !line.trim().is_empty() {
            break;
        }
    }

    let first_trimmed = line.trim();
    if is_http_header_line(first_trimmed) {
        let mut content_length = parse_content_length_header(first_trimmed);
        let mut hdr = String::new();
        loop {
            hdr.clear();
            let bytes = reader.read_line(&mut hdr)?;
            if bytes == 0 {
                anyhow::bail!("unexpected EOF while reading MCP Content-Length header block");
            }
            let h_trimmed = hdr.trim();
            if h_trimmed.is_empty() {
                break;
            }
            if let Some(len) = parse_content_length_header(h_trimmed) {
                content_length = Some(len);
            }
        }
        let len = content_length
            .ok_or_else(|| anyhow::anyhow!("missing Content-Length header in MCP message"))?;
        if len == 0 || len > MAX_MCP_MESSAGE_BYTES {
            anyhow::bail!("MCP Content-Length {len} exceeds bounds");
        }
        let mut body = vec![0_u8; len];
        reader.read_exact(&mut body)?;
        let body_text = String::from_utf8(body)?;
        Ok(Some((McpFraming::ContentLength, body_text)))
    } else {
        Ok(Some((McpFraming::Ndjson, first_trimmed.to_string())))
    }
}

fn is_http_header_line(line: &str) -> bool {
    if line.starts_with('{') || line.starts_with('[') {
        return false;
    }
    line.split_once(':').is_some_and(|(key, _)| {
        let k = key.trim();
        k.eq_ignore_ascii_case("content-length") || k.eq_ignore_ascii_case("content-type")
    })
}

fn parse_content_length_header(line: &str) -> Option<usize> {
    let (key, value) = line.split_once(':')?;
    if key.trim().eq_ignore_ascii_case("content-length") {
        value.trim().parse::<usize>().ok()
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use crate::client::JikuClientOptions;

    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn mcp_stdio_roundtrip_initialize_list_and_call() {
        let mut client = JikuClient::new(JikuClientOptions {
            in_memory_db: true,
            ..JikuClientOptions::default()
        })
        .expect("client init");

        let input = concat!(
            "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{}}\n",
            "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}\n",
            "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/list\",\"params\":{}}\n",
            "{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"tools/call\",\"params\":{\"name\":\"jiku_status\",\"arguments\":{}}}\n",
            "{\"jsonrpc\":\"2.0\",\"id\":4,\"method\":\"tools/call\",\"params\":{\"name\":\"jiku_skills_list\",\"arguments\":{}}}\n"
        );

        let reader = Cursor::new(input.as_bytes());
        let mut output = Vec::<u8>::new();
        run_mcp_loop(&mut client, reader, &mut output, true)
            .await
            .expect("MCP loop");

        let text = String::from_utf8(output).expect("valid utf8 output");
        let lines: Vec<Value> = text
            .lines()
            .map(|l| serde_json::from_str(l).expect("valid JSON reply"))
            .collect();
        // 4 replies (notification produces no reply).
        assert_eq!(lines.len(), 4);
        assert_eq!(lines[0]["id"], 1);
        assert_eq!(lines[0]["result"]["protocolVersion"], MCP_PROTOCOL_VERSION);

        assert_eq!(lines[1]["id"], 2);
        let tools = lines[1]["result"]["tools"].as_array().expect("tools array");
        assert!(tools.iter().any(|t| t["name"] == "jiku_status"));
        assert!(tools.iter().any(|t| t["name"] == "jiku_run_task"));

        assert_eq!(lines[2]["id"], 3);
        assert_eq!(lines[2]["result"]["isError"], false);
        assert_eq!(
            lines[2]["result"]["structuredContent"]["planner_model"],
            "deepseek-v4.1-flash"
        );

        assert_eq!(lines[3]["id"], 4);
        assert_eq!(lines[3]["result"]["isError"], false);
    }

    fn frame_content_length(body: &str) -> String {
        format!("Content-Length: {}\r\n\r\n{}", body.len(), body)
    }

    fn parse_content_length_stream(raw: &[u8]) -> Vec<Value> {
        let mut cursor = 0usize;
        let mut out = Vec::new();
        while cursor < raw.len() {
            let rest = &raw[cursor..];
            let Some(hdr_end_rel) = rest.windows(4).position(|w| w == b"\r\n\r\n") else {
                panic!("expected Content-Length framing at offset {cursor}");
            };
            let hdr = std::str::from_utf8(&rest[..hdr_end_rel]).unwrap();
            let len: usize = hdr
                .lines()
                .find_map(|l| parse_content_length_header(l.trim()))
                .expect("Content-Length header");
            let body_start = cursor + hdr_end_rel + 4;
            let body_end = body_start + len;
            let val: Value = serde_json::from_slice(&raw[body_start..body_end]).unwrap();
            out.push(val);
            cursor = body_end;
        }
        out
    }

    #[tokio::test(flavor = "current_thread")]
    async fn mcp_supports_content_length_resources_prompts_notifications_and_compact_tools() {
        let mut client = JikuClient::new(JikuClientOptions {
            socket_path: Some(PathBuf::from("/tmp/jiku-nonexistent-mcp-test.sock")),
            in_memory_db: true,
            ..JikuClientOptions::default()
        })
        .expect("client init");

        let req_init = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#;
        let notif = r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#;
        let req_res = r#"{"jsonrpc":"2.0","id":2,"method":"resources/list","params":{}}"#;
        let req_tpl = r#"{"jsonrpc":"2.0","id":3,"method":"resources/templates/list","params":{}}"#;
        let req_prompts = r#"{"jsonrpc":"2.0","id":4,"method":"prompts/list","params":{}}"#;
        let req_tools = r#"{"jsonrpc":"2.0","id":5,"method":"tools/list","params":{}}"#;
        let req_ctx = r#"{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"jiku_context_inspect","arguments":{"max_tokens":4096,"compact":true}}}"#;
        let req_tts = r#"{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"jiku_tts_say","arguments":{"text":"Compact check","play":false,"compact":true}}}"#;
        let req_bench = r#"{"jsonrpc":"2.0","id":8,"method":"tools/call","params":{"name":"jiku_bench_run","arguments":{"offline":true,"compact":true}}}"#;

        let mut wire_in = String::new();
        for body in [
            req_init,
            notif,
            req_res,
            req_tpl,
            req_prompts,
            req_tools,
            req_ctx,
            req_tts,
            req_bench,
        ] {
            wire_in.push_str(&frame_content_length(body));
        }

        let reader = Cursor::new(wire_in.as_bytes());
        let mut output = Vec::<u8>::new();
        run_mcp_stdio(&mut client, reader, &mut output, false)
            .await
            .expect("Content-Length MCP loop");

        let responses = parse_content_length_stream(&output);
        // 8 replies (notification skipped silently).
        assert_eq!(responses.len(), 8);
        assert_eq!(responses[0]["id"], 1);
        assert_eq!(responses[1]["result"]["resources"], json!([]));
        assert_eq!(responses[2]["result"]["resourceTemplates"], json!([]));
        assert_eq!(responses[3]["result"]["prompts"], json!([]));

        let tools = responses[4]["result"]["tools"].as_array().unwrap();
        for expected in [
            "jiku_a11y_snapshot",
            "jiku_a11y_act",
            "jiku_tts_say",
            "jiku_context_inspect",
            "jiku_bench_run",
        ] {
            assert!(
                tools.iter().any(|t| t["name"] == expected),
                "missing tool {expected}"
            );
        }

        let ctx_sc = &responses[5]["result"]["structuredContent"];
        assert!(ctx_sc.get("config").is_none(), "compact mode omits config");
        assert!(
            ctx_sc["simulated_report"]["compacted_prompt_tokens"]
                .as_u64()
                .is_some()
        );

        let tts_sc = &responses[6]["result"]["structuredContent"];
        assert_eq!(tts_sc["ok"], true);
        assert!(tts_sc.get("pcm_envelope").is_none());

        let bench_sc = &responses[7]["result"]["structuredContent"];
        assert_eq!(bench_sc["cells"].as_array().unwrap().len(), 4);
        assert!(bench_sc["cells"][0].get("task_scores").is_none());
    }
}
