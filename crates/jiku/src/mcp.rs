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
            "description": "Inspect jikud config paths, planner/decision driver bindings, memory DB counts, and live Koma IPC connectivity.",
            "inputSchema": {
                "type": "object",
                "properties": {},
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
                    "force_emulate": { "type": "boolean" }
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
                    "questions": { "type": "array" }
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
                "properties": {},
                "additionalProperties": false
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
                "capabilities": { "tools": { "listChanged": false } },
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
    match name {
        "jiku_status" => Ok(serde_json::to_value(client.status().await?)?),
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
            Ok(serde_json::to_value(
                client.run_task(ws, goal, dry_run, force_emulate).await?,
            )?)
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
            Ok(serde_json::to_value(
                client
                    .decide(img.as_deref(), prompt, &questions, force_emulate)
                    .await?,
            )?)
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
        "jiku_skills_list" => Ok(serde_json::to_value(client.skills_list().await?)?),
        other => anyhow::bail!("unknown MCP tool `{other}`"),
    }
}

/// Runs the newline-delimited JSON-RPC 2.0 MCP loop reading from `reader` and writing to `writer`.
pub async fn run_mcp_loop<R: BufRead, W: Write>(
    client: &mut JikuClient,
    mut reader: R,
    mut writer: W,
    stateless: bool,
) -> anyhow::Result<()> {
    let mut line = String::new();
    loop {
        line.clear();
        let bytes = reader.read_line(&mut line)?;
        if bytes == 0 {
            break;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(reply) = process_jsonrpc_line(client, trimmed).await {
            writeln!(writer, "{reply}")?;
            writer.flush()?;
        }
        if stateless && trimmed.contains("\"tools/call\"") {
            // In stateless mode callers can send a single initialize+tools/call stream or EOF;
            // keep draining until EOF so multi-message stateless pipes work seamlessly.
            continue;
        }
    }
    Ok(())
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
}
