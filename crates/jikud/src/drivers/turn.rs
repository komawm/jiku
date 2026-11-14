//! Autoregressive `TurnModelDriver` supporting OpenCode Go (`deepseek-v4.1-flash`),
//! OpenAI-compatible, Google Gemini (`generateContent`), and Anthropic endpoints.

use std::time::Duration;

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64_STD};
use futures_util::StreamExt as _;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{
    config::{JikuConfig, ProviderKind, ProviderSpec},
    drivers::{DriverError, DriverMediaInput, build_http_client},
};

/// Single chat message passed to [`TurnModelClient::generate`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TurnMessage {
    /// OpenAI-style role (`"user"`, `"assistant"`).
    pub role: String,
    /// Primary plain text content.
    pub text: String,
    /// Optional multimodal attachments (primarily PNG/JPEG frames).
    pub media: Vec<DriverMediaInput>,
}

/// Request specification for [`TurnModelClient::generate`].
#[derive(Debug, Clone)]
pub struct TurnRequest {
    /// Logical provider ID in [`JikuConfig::providers`].
    pub provider_id: String,
    /// Target model identifier (e.g. `"deepseek-v4.1-flash"`).
    pub model: String,
    /// System instruction prefix.
    pub system_prompt: String,
    /// Conversation messages (`>= 1`).
    pub messages: Vec<TurnMessage>,
    /// Optional sampling temperature.
    pub temperature: Option<f32>,
    /// Optional output token ceiling.
    pub max_tokens: Option<u32>,
    /// Whether to request JSON-object structured output when supported.
    pub json_object: bool,
    /// Whether to use SSE streaming (`stream: true`) on OpenAI-compatible gateways.
    pub stream: bool,
    /// Whether extended thinking / reasoning tokens are enabled (`false` sends
    /// `"thinking": {"type": "disabled"}` on OpenCode Go / compatible endpoints).
    pub thinking_enabled: bool,
}

/// Completed generation outcome returned by [`TurnModelClient::generate`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TurnResponse {
    /// Logical provider ID used.
    pub provider_id: String,
    /// Executed model identifier.
    pub model: String,
    /// Final assistant text output.
    pub text: String,
    /// Reasoning text if the model produced any (`None` when thinking is disabled).
    pub reasoning_text: Option<String>,
    /// Input/prompt token count if reported by the gateway.
    pub prompt_tokens: u64,
    /// Completion token count if reported by the gateway.
    pub completion_tokens: u64,
    /// Reasoning token count (`0` when thinking mode is disabled).
    pub reasoning_tokens: u64,
    /// Deterministic `x-opencode-session` UUID sent on OpenCode Go calls, if applicable.
    pub session_id: Option<String>,
}

/// Derives a deterministic 128-bit v4-shaped UUID matching OpenCode Go's
/// `uuidFromString` algorithm from UTF-16 code units.
///
/// Time complexity: $O(L)$ where $L$ is `input.len()` in UTF-16 code units.
/// Space complexity: $O(1)$ fixed 36-byte UUID string allocation.
#[must_use]
pub fn derive_opencode_session_uuid(input: &str) -> String {
    let mut h1: u32 = 0x811c_9dc5;
    let mut h2: u32 = 0x9e37_79b9;
    let mut h3: u32 = 0x85eb_ca6b;
    let mut h4: u32 = 0xc2b2_ae35;

    for code_unit in input.encode_utf16() {
        let c = u32::from(code_unit);
        h1 = (h1 ^ c).wrapping_mul(0x0100_0193);
        h2 = (h2 ^ c).wrapping_mul(0x85eb_ca6b);
        h3 = (h3 ^ c).wrapping_mul(0xc2b2_ae35);
        h4 = (h4 ^ c).wrapping_mul(0x27d4_eb2f);
    }

    let raw = format!("{h1:08x}{h2:08x}{h3:08x}{h4:08x}");
    let mut chars: Vec<char> = raw.chars().collect();
    chars[12] = '4';
    chars[16] = '8';
    let hex: String = chars.into_iter().collect();
    format!(
        "{}-{}-{}-{}-{}",
        &hex[0..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    )
}

/// Computes the stable `x-opencode-session` UUID for a prompt + message history.
#[must_use]
pub fn opencode_session_id_for_turn(system_prompt: &str, messages: &[TurnMessage]) -> String {
    let first_user_text = messages
        .iter()
        .find(|m| m.role == "user")
        .map(|m| m.text.as_str())
        .unwrap_or("");
    derive_opencode_session_uuid(&format!("{system_prompt}\0{first_user_text}"))
}

/// Async client implementing `TurnModelDriver` across heterogeneous gateways.
#[derive(Debug, Clone)]
pub struct TurnModelClient {
    http: reqwest::Client,
}

impl TurnModelClient {
    /// Creates a new [`TurnModelClient`] with a shared `rustls-tls` connection pool.
    pub fn new(timeout: Duration) -> Result<Self, DriverError> {
        Ok(Self {
            http: build_http_client(timeout)?,
        })
    }

    /// Executes an autoregressive turn request against the provider configured in `config`.
    pub async fn generate(
        &self,
        config: &JikuConfig,
        req: &TurnRequest,
    ) -> Result<TurnResponse, DriverError> {
        let provider = config.providers.get(&req.provider_id).ok_or_else(|| {
            DriverError::Config(format!("unknown provider `{}`", req.provider_id))
        })?;
        let api_key = config
            .read_provider_api_key(&req.provider_id)
            .map_err(|e| DriverError::Config(e.to_string()))?;

        match provider.kind {
            ProviderKind::OpenCodeGo | ProviderKind::OpenAiCompatible => {
                self.call_openai_family(provider, api_key.as_deref(), req)
                    .await
            }
            ProviderKind::GoogleGemini => {
                self.call_gemini_generate_content(provider, api_key.as_deref(), req)
                    .await
            }
            ProviderKind::Anthropic => {
                self.call_anthropic_messages(provider, api_key.as_deref(), req)
                    .await
            }
        }
    }

    async fn call_openai_family(
        &self,
        provider: &ProviderSpec,
        api_key: Option<&str>,
        req: &TurnRequest,
    ) -> Result<TurnResponse, DriverError> {
        let base = provider.base_url.trim_end_matches('/');
        let url = format!("{base}/chat/completions");

        let is_opencode = provider.kind == ProviderKind::OpenCodeGo || base.contains("opencode.ai");
        let session_id = if is_opencode {
            Some(opencode_session_id_for_turn(
                &req.system_prompt,
                &req.messages,
            ))
        } else {
            None
        };

        let mut wire_messages = Vec::with_capacity(req.messages.len() + 1);
        if !req.system_prompt.trim().is_empty() {
            wire_messages.push(json!({
                "role": "system",
                "content": req.system_prompt,
            }));
        }
        for msg in &req.messages {
            if msg.media.is_empty() {
                wire_messages.push(json!({
                    "role": msg.role,
                    "content": msg.text,
                }));
            } else {
                let mut parts = Vec::with_capacity(msg.media.len() + 1);
                parts.push(json!({ "type": "text", "text": msg.text }));
                for attachment in &msg.media {
                    if let DriverMediaInput::Inline { mime_type, bytes } = attachment {
                        let b64 = BASE64_STD.encode(bytes);
                        parts.push(json!({
                            "type": "image_url",
                            "image_url": {
                                "url": format!("data:{mime_type};base64,{b64}")
                            }
                        }));
                    }
                }
                wire_messages.push(json!({
                    "role": msg.role,
                    "content": parts,
                }));
            }
        }

        let mut body = json!({
            "model": req.model,
            "messages": wire_messages,
            "stream": req.stream,
        });
        if let Some(obj) = body.as_object_mut() {
            if let Some(temp) = req.temperature {
                obj.insert("temperature".to_string(), json!(temp));
            }
            if let Some(max_tok) = req.max_tokens {
                obj.insert("max_tokens".to_string(), json!(max_tok));
            }
            if req.json_object {
                obj.insert(
                    "response_format".to_string(),
                    json!({ "type": "json_object" }),
                );
            }
            if !req.thinking_enabled {
                // Verified live against OpenCode Go `deepseek-v4.1-flash`:
                // returns `reasoning_content: null` and `reasoning_tokens: 0`.
                obj.insert("thinking".to_string(), json!({ "type": "disabled" }));
            }
        }

        let mut http_req = self.http.post(&url).json(&body);
        if let Some(key) = api_key {
            http_req = http_req.bearer_auth(key);
        }
        if let Some(sid) = &session_id {
            http_req = http_req.header("x-opencode-session", sid);
        }

        let resp = http_req.send().await?;
        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let err_body = resp.text().await.unwrap_or_default();
            return Err(DriverError::RemoteStatus {
                status,
                endpoint: url,
                body: err_body,
            });
        }

        if req.stream {
            return parse_openai_sse_stream(resp, req, session_id).await;
        }

        let parsed: Value = resp.json().await?;
        let message = parsed
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|first| first.get("message"))
            .ok_or_else(|| {
                DriverError::InvalidPayload(
                    "OpenAI response missing choices[0].message".to_string(),
                )
            })?;

        let text = extract_openai_message_text(message.get("content"));
        let reasoning_text = message
            .get("reasoning_content")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(ToOwned::to_owned);

        let usage = parsed.get("usage");
        let prompt_tokens = usage
            .and_then(|u| u.get("prompt_tokens"))
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let completion_tokens = usage
            .and_then(|u| u.get("completion_tokens"))
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let reasoning_tokens = usage
            .and_then(|u| u.get("completion_tokens_details"))
            .and_then(|d| d.get("reasoning_tokens"))
            .and_then(Value::as_u64)
            .unwrap_or(0);

        Ok(TurnResponse {
            provider_id: req.provider_id.clone(),
            model: req.model.clone(),
            text,
            reasoning_text,
            prompt_tokens,
            completion_tokens,
            reasoning_tokens,
            session_id,
        })
    }

    async fn call_gemini_generate_content(
        &self,
        provider: &ProviderSpec,
        api_key: Option<&str>,
        req: &TurnRequest,
    ) -> Result<TurnResponse, DriverError> {
        let base = provider.base_url.trim_end_matches('/');
        let url = format!("{base}/models/{}:generateContent", req.model);
        let key = api_key.ok_or_else(|| {
            DriverError::Config(format!("missing Gemini API key for `{}`", req.provider_id))
        })?;

        let mut contents = Vec::with_capacity(req.messages.len());
        for msg in &req.messages {
            let role = if msg.role == "assistant" {
                "model"
            } else {
                "user"
            };
            let mut parts = Vec::new();
            for attachment in &msg.media {
                match attachment {
                    DriverMediaInput::Inline { mime_type, bytes } => {
                        parts.push(json!({
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": BASE64_STD.encode(bytes),
                            }
                        }));
                    }
                    DriverMediaInput::FilesUri { uri } => {
                        parts.push(json!({
                            "file_data": { "file_uri": uri }
                        }));
                    }
                }
            }
            if !msg.text.is_empty() {
                parts.push(json!({ "text": msg.text }));
            }
            contents.push(json!({ "role": role, "parts": parts }));
        }

        let mut gen_cfg = json!({});
        if let Some(obj) = gen_cfg.as_object_mut() {
            if let Some(temp) = req.temperature {
                obj.insert("temperature".to_string(), json!(temp));
            }
            if let Some(max_tok) = req.max_tokens {
                obj.insert("maxOutputTokens".to_string(), json!(max_tok));
            }
            if req.json_object {
                obj.insert("responseMimeType".to_string(), json!("application/json"));
            }
            if !req.thinking_enabled {
                obj.insert("thinkingConfig".to_string(), json!({ "thinkingBudget": 0 }));
            }
        }

        let mut body = json!({
            "contents": contents,
            "generationConfig": gen_cfg,
        });
        if !req.system_prompt.trim().is_empty()
            && let Some(obj) = body.as_object_mut()
        {
            obj.insert(
                "systemInstruction".to_string(),
                json!({ "parts": [{ "text": req.system_prompt }] }),
            );
        }

        let resp = self
            .http
            .post(&url)
            .header("x-goog-api-key", key)
            .json(&body)
            .send()
            .await?;
        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let err_body = resp.text().await.unwrap_or_default();
            return Err(DriverError::RemoteStatus {
                status,
                endpoint: url,
                body: err_body,
            });
        }

        let parsed: Value = resp.json().await?;
        let mut text = String::new();
        if let Some(parts) = parsed
            .get("candidates")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("content"))
            .and_then(|c| c.get("parts"))
            .and_then(Value::as_array)
        {
            for p in parts {
                if let Some(t) = p.get("text").and_then(Value::as_str) {
                    text.push_str(t);
                }
            }
        }
        let usage = parsed.get("usageMetadata");
        Ok(TurnResponse {
            provider_id: req.provider_id.clone(),
            model: req.model.clone(),
            text,
            reasoning_text: None,
            prompt_tokens: usage
                .and_then(|u| u.get("promptTokenCount"))
                .and_then(Value::as_u64)
                .unwrap_or(0),
            completion_tokens: usage
                .and_then(|u| u.get("candidatesTokenCount"))
                .and_then(Value::as_u64)
                .unwrap_or(0),
            reasoning_tokens: usage
                .and_then(|u| u.get("thoughtsTokenCount"))
                .and_then(Value::as_u64)
                .unwrap_or(0),
            session_id: None,
        })
    }

    async fn call_anthropic_messages(
        &self,
        provider: &ProviderSpec,
        api_key: Option<&str>,
        req: &TurnRequest,
    ) -> Result<TurnResponse, DriverError> {
        let base = provider.base_url.trim_end_matches('/');
        let url = format!("{base}/messages");
        let key = api_key.ok_or_else(|| {
            DriverError::Config(format!(
                "missing Anthropic API key for `{}`",
                req.provider_id
            ))
        })?;

        let wire_messages: Vec<Value> = req
            .messages
            .iter()
            .map(|msg| {
                let mut content = Vec::new();
                for attachment in &msg.media {
                    if let DriverMediaInput::Inline { mime_type, bytes } = attachment {
                        content.push(json!({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": BASE64_STD.encode(bytes),
                            }
                        }));
                    }
                }
                if !msg.text.is_empty() {
                    content.push(json!({ "type": "text", "text": msg.text }));
                }
                json!({ "role": msg.role, "content": content })
            })
            .collect();

        let mut body = json!({
            "model": req.model,
            "max_tokens": req.max_tokens.unwrap_or(2048),
            "messages": wire_messages,
        });
        if !req.system_prompt.trim().is_empty()
            && let Some(obj) = body.as_object_mut()
        {
            obj.insert("system".to_string(), json!(req.system_prompt));
        }
        if let Some(temp) = req.temperature
            && let Some(obj) = body.as_object_mut()
        {
            obj.insert("temperature".to_string(), json!(temp));
        }

        let resp = self
            .http
            .post(&url)
            .header("x-api-key", key)
            .header("anthropic-version", "2023-06-01")
            .json(&body)
            .send()
            .await?;
        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let err_body = resp.text().await.unwrap_or_default();
            return Err(DriverError::RemoteStatus {
                status,
                endpoint: url,
                body: err_body,
            });
        }

        let parsed: Value = resp.json().await?;
        let mut text = String::new();
        if let Some(blocks) = parsed.get("content").and_then(Value::as_array) {
            for b in blocks {
                if b.get("type").and_then(Value::as_str) == Some("text")
                    && let Some(t) = b.get("text").and_then(Value::as_str)
                {
                    text.push_str(t);
                }
            }
        }
        let usage = parsed.get("usage");
        Ok(TurnResponse {
            provider_id: req.provider_id.clone(),
            model: req.model.clone(),
            text,
            reasoning_text: None,
            prompt_tokens: usage
                .and_then(|u| u.get("input_tokens"))
                .and_then(Value::as_u64)
                .unwrap_or(0),
            completion_tokens: usage
                .and_then(|u| u.get("output_tokens"))
                .and_then(Value::as_u64)
                .unwrap_or(0),
            reasoning_tokens: 0,
            session_id: None,
        })
    }
}

async fn parse_openai_sse_stream(
    resp: reqwest::Response,
    req: &TurnRequest,
    session_id: Option<String>,
) -> Result<TurnResponse, DriverError> {
    let mut stream = resp.bytes_stream();
    let mut line_buf = String::new();
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut prompt_tokens = 0_u64;
    let mut completion_tokens = 0_u64;
    let mut reasoning_tokens = 0_u64;

    while let Some(chunk_res) = stream.next().await {
        let bytes = chunk_res?;
        line_buf.push_str(&String::from_utf8_lossy(&bytes));
        while let Some(nl_pos) = line_buf.find('\n') {
            let raw_line = line_buf[..nl_pos].trim_end_matches('\r').trim().to_string();
            line_buf.drain(..=nl_pos);
            let Some(data) = raw_line.strip_prefix("data:").map(str::trim) else {
                continue;
            };
            if data == "[DONE]" {
                break;
            }
            if let Ok(val) = serde_json::from_str::<Value>(data) {
                if let Some(delta) = val
                    .get("choices")
                    .and_then(|c| c.get(0))
                    .and_then(|c| c.get("delta"))
                {
                    text.push_str(&extract_openai_message_text(delta.get("content")));
                    if let Some(r) = delta.get("reasoning_content").and_then(Value::as_str) {
                        reasoning.push_str(r);
                    }
                }
                if let Some(usage) = val.get("usage") {
                    prompt_tokens = usage
                        .get("prompt_tokens")
                        .and_then(Value::as_u64)
                        .unwrap_or(prompt_tokens);
                    completion_tokens = usage
                        .get("completion_tokens")
                        .and_then(Value::as_u64)
                        .unwrap_or(completion_tokens);
                    reasoning_tokens = usage
                        .get("completion_tokens_details")
                        .and_then(|d| d.get("reasoning_tokens"))
                        .and_then(Value::as_u64)
                        .unwrap_or(reasoning_tokens);
                }
            }
        }
    }

    Ok(TurnResponse {
        provider_id: req.provider_id.clone(),
        model: req.model.clone(),
        text,
        reasoning_text: if reasoning.is_empty() {
            None
        } else {
            Some(reasoning)
        },
        prompt_tokens,
        completion_tokens,
        reasoning_tokens,
        session_id,
    })
}

fn extract_openai_message_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => {
            let mut joined = String::new();
            for p in parts {
                if let Some(t) = p.get("text").and_then(Value::as_str) {
                    joined.push_str(t);
                }
            }
            joined
        }
        _ => String::new(),
    }
}

/// Strips optional markdown ```json ... ``` wrappers around JSON text returned
/// by chat models.
#[must_use]
pub fn strip_markdown_json_fences(raw: &str) -> &str {
    let trimmed = raw.trim();
    if let Some(after_open) = trimmed.strip_prefix("```") {
        let after_lang = if let Some(nl) = after_open.find('\n') {
            &after_open[nl + 1..]
        } else {
            after_open
        };
        if let Some(before_close) = after_lang.strip_suffix("```") {
            return before_close.trim();
        }
        return after_lang.trim();
    }
    trimmed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opencode_session_uuid_matches_v4_format_and_determinism() {
        let a = derive_opencode_session_uuid("sys\0hello world");
        let b = derive_opencode_session_uuid("sys\0hello world");
        let c = derive_opencode_session_uuid("sys\0different");

        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(a.len(), 36);
        let bytes = a.as_bytes();
        assert_eq!(bytes[8], b'-');
        assert_eq!(bytes[13], b'-');
        assert_eq!(bytes[14], b'4');
        assert_eq!(bytes[18], b'-');
        assert_eq!(bytes[19], b'8');
        assert_eq!(bytes[23], b'-');
    }

    #[test]
    fn strip_markdown_json_fences_handles_code_blocks() {
        let wrapped = "```json\n{\"answers\":{\"q\":1}}\n```";
        assert_eq!(
            strip_markdown_json_fences(wrapped),
            "{\"answers\":{\"q\":1}}"
        );
        assert_eq!(
            strip_markdown_json_fences("  {\"ok\":true}  "),
            "{\"ok\":true}"
        );
    }
}
