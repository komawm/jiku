//! Concrete `DecisionsDriver` client for `POST /v1beta/decisions` (`gemini-decision-1`)
//! paired with automatic [`DecisionExecutionMode::EmulatedStructuredTextModel`] fallback.

use std::{collections::BTreeMap, time::Duration};

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64_STD};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{
    DecisionBatchSpec, DecisionExecutionMode, DecisionMediaSource, DecisionQuestionKind,
    DecisionRouterPolicy, GEMINI_DECISION_1_MODEL_ID,
    config::{JikuConfig, ProviderKind},
    drivers::{
        DriverError, DriverMediaInput, build_http_client,
        turn::{TurnMessage, TurnModelClient, TurnRequest, strip_markdown_json_fences},
    },
    synthesize_emulated_decision_prompt,
};

/// Single evaluated answer in a [`DecisionBatchResponse`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DecisionAnswer {
    /// Binary `noul` probability clamped in `[0.01, 0.99]`.
    Noul {
        /// Calibrated `Yes` probability.
        noul: f32,
        /// Model confidence scalar.
        #[serde(default = "default_confidence")]
        confidence: f32,
    },
    /// Mutually exclusive categorical choice distribution (`sum == 1.0`).
    Choice {
        /// Selected winning choice key.
        choice: String,
        /// Top choice confidence scalar.
        #[serde(default = "default_confidence")]
        confidence: f32,
        /// Per-candidate probability map.
        #[serde(default)]
        probabilities: BTreeMap<String, f32>,
    },
    /// Ordered rubric score evaluation.
    Score {
        /// Expected continuous rubric score.
        score: f32,
        /// Confidence scalar.
        #[serde(default = "default_confidence")]
        confidence: f32,
        /// Per-level probability distribution.
        #[serde(default)]
        probabilities: BTreeMap<String, f32>,
        /// Rubric level description mapping.
        #[serde(default)]
        legend: BTreeMap<String, String>,
    },
}

const fn default_confidence() -> f32 {
    0.0
}

impl DecisionAnswer {
    /// Convenience accessor returning the `noul` `Yes` probability if `Self::Noul`.
    #[must_use]
    pub const fn noul_prob(&self) -> Option<f32> {
        match self {
            Self::Noul { noul, .. } => Some(*noul),
            _ => None,
        }
    }

    /// Returns `(choice_key, winning_prob)` when `Self::Choice`.
    #[must_use]
    pub fn top_choice(&self) -> Option<(&str, f32)> {
        match self {
            Self::Choice {
                choice,
                confidence,
                probabilities,
            } => {
                let prob = probabilities.get(choice).copied().unwrap_or(*confidence);
                Some((choice.as_str(), prob))
            }
            _ => None,
        }
    }
}

/// Token billing metrics reported by `/v1beta/decisions` or emulated fallback.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecisionUsage {
    /// Input/prompt tokens consumed.
    #[serde(alias = "input_tokens")]
    pub total_input_tokens: u64,
    /// Output tokens generated (`0` for native zero-decode `gemini-decision-1`).
    #[serde(alias = "output_tokens")]
    pub total_output_tokens: u64,
    /// Cached prefix tokens reused, if reported.
    #[serde(default)]
    pub cached_tokens: u64,
}

/// Unified structured output from either `NativeDecisionsApi` or
/// `EmulatedStructuredTextModel`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionBatchResponse {
    /// `"native_decisions_api"` or `"emulated_structured_text_model"`.
    pub execution_mode: String,
    /// Provider key used.
    pub provider_id: String,
    /// Model ID executed.
    pub model: String,
    /// Answers keyed by question ID.
    #[serde(alias = "results")]
    pub answers: BTreeMap<String, DecisionAnswer>,
    /// Token accounting.
    #[serde(default)]
    pub usage: DecisionUsage,
}

/// Low-level HTTP client targeting Google AI Studio's `POST /v1beta/decisions`.
#[derive(Debug, Clone)]
pub struct DecisionsClient {
    http: reqwest::Client,
}

impl DecisionsClient {
    /// Creates a native `DecisionsDriver` HTTP client.
    pub fn new(timeout: Duration) -> Result<Self, DriverError> {
        Ok(Self {
            http: build_http_client(timeout)?,
        })
    }

    /// Serializes the verified `POST /v1beta/decisions` request wire body
    /// (`model`, `input`, `questions`).
    pub fn build_native_wire_payload(
        spec: &DecisionBatchSpec,
        context_text: &str,
        media: Option<&DriverMediaInput>,
    ) -> Result<Value, DriverError> {
        spec.validate()
            .map_err(|err| DriverError::Validation(format!("{err:?}")))?;

        let input_value = build_decisions_input_value(context_text, &spec.media_source, media)?;
        let mut questions_map = serde_json::Map::with_capacity(spec.questions.len());

        for (qid, kind) in &spec.questions {
            let q_val = match kind {
                DecisionQuestionKind::Noul { instructions } => json!({
                    "type": "noul",
                    "instructions": instructions,
                }),
                DecisionQuestionKind::Choice {
                    instructions,
                    choices,
                    ..
                } => {
                    let mut criteria = serde_json::Map::with_capacity(choices.len());
                    for choice in choices {
                        criteria.insert(choice.clone(), Value::String(choice.clone()));
                    }
                    json!({
                        "type": "choice",
                        "instructions": instructions,
                        "criteria": criteria,
                    })
                }
                DecisionQuestionKind::Score {
                    instructions,
                    rubric,
                } => json!({
                    "type": "score",
                    "instructions": instructions,
                    "criteria": rubric,
                }),
            };
            questions_map.insert(qid.clone(), q_val);
        }

        Ok(json!({
            "model": spec.model,
            "input": input_value,
            "questions": questions_map,
        }))
    }

    /// Dispatches a validated `DecisionBatchSpec` to Google AI Studio `POST {base_url}/decisions`.
    pub async fn evaluate_native(
        &self,
        base_url: &str,
        api_key: &str,
        provider_id: &str,
        spec: &DecisionBatchSpec,
        context_text: &str,
        media: Option<&DriverMediaInput>,
    ) -> Result<DecisionBatchResponse, DriverError> {
        let payload = Self::build_native_wire_payload(spec, context_text, media)?;
        let url = format!("{}/decisions", base_url.trim_end_matches('/'));

        let resp = self
            .http
            .post(&url)
            .header("x-goog-api-key", api_key)
            .json(&payload)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let body = resp.text().await.unwrap_or_default();
            return Err(DriverError::RemoteStatus {
                status,
                endpoint: url,
                body,
            });
        }

        let wire: RawDecisionsWireResponse = resp.json().await?;
        let mut answers = wire.answers;
        if answers.is_empty() && !wire.results.is_empty() {
            answers = wire.results;
        }
        if answers.is_empty() {
            return Err(DriverError::InvalidPayload(
                "/v1beta/decisions returned zero answers".to_string(),
            ));
        }

        Ok(DecisionBatchResponse {
            execution_mode: "native_decisions_api".to_string(),
            provider_id: provider_id.to_string(),
            model: GEMINI_DECISION_1_MODEL_ID.to_string(),
            answers,
            usage: wire.usage,
        })
    }
}

/// Unified executor supporting `NativeDecisionsApi` (`gemini-decision-1`) and
/// automatic `EmulatedStructuredTextModel` fallback via [`TurnModelClient`].
#[derive(Debug, Clone)]
pub struct DecisionExecutor {
    decisions: DecisionsClient,
    turn: TurnModelClient,
}

impl DecisionExecutor {
    /// Constructs a unified decision executor.
    pub fn new(timeout: Duration) -> Result<Self, DriverError> {
        Ok(Self {
            decisions: DecisionsClient::new(timeout)?,
            turn: TurnModelClient::new(timeout)?,
        })
    }

    /// Evaluates a validated decision batch using either the native `/v1beta/decisions`
    /// endpoint or structured-output emulation on `policy.primary_text_model_id`.
    ///
    /// When `force_emulate` is `false` and `DecisionExecutionMode::NativeDecisionsApi`
    /// cannot run (for example `GEMINI_API_KEY` is unset and `fallback_to_primary`
    /// is `true`), this degrades seamlessly onto [`TurnModelClient`] without error.
    #[allow(clippy::too_many_arguments)]
    pub async fn evaluate(
        &self,
        config: &JikuConfig,
        policy: &DecisionRouterPolicy,
        spec: &DecisionBatchSpec,
        context_text: &str,
        media: Option<&DriverMediaInput>,
        force_emulate: bool,
        fallback_to_primary: bool,
    ) -> Result<DecisionBatchResponse, DriverError> {
        spec.validate()
            .map_err(|err| DriverError::Validation(format!("{err:?}")))?;

        let resolved_mode = policy
            .resolve_mode()
            .map_err(|msg| DriverError::Validation(msg.to_string()))?;

        let target_mode = if force_emulate {
            DecisionExecutionMode::EmulatedStructuredTextModel {
                provider: policy.primary_provider.clone(),
                model: policy.primary_text_model_id.clone(),
                enforce_json_schema: true,
            }
        } else {
            resolved_mode
        };

        match target_mode {
            DecisionExecutionMode::NativeDecisionsApi { provider, .. } => {
                let spec_prov = config.providers.get(&provider).ok_or_else(|| {
                    DriverError::Config(format!("unknown decision provider `{provider}`"))
                })?;
                if spec_prov.kind != ProviderKind::GoogleGemini {
                    return Err(DriverError::Config(format!(
                        "native /v1beta/decisions requires google_gemini provider, got `{provider}`"
                    )));
                }
                match config.read_provider_api_key(&provider) {
                    Ok(Some(api_key)) => {
                        match self
                            .decisions
                            .evaluate_native(
                                &spec_prov.base_url,
                                &api_key,
                                &provider,
                                spec,
                                context_text,
                                media,
                            )
                            .await
                        {
                            Ok(res) => Ok(res),
                            Err(err) if fallback_to_primary => {
                                tracing::warn!(
                                    provider = %provider,
                                    "native decisions API failed ({err}); falling back to structured text model"
                                );
                                self.evaluate_emulated(
                                    config,
                                    &policy.primary_provider,
                                    &policy.primary_text_model_id,
                                    spec,
                                    context_text,
                                    media,
                                )
                                .await
                            }
                            Err(err) => Err(err),
                        }
                    }
                    Ok(None) | Err(_) if fallback_to_primary => {
                        self.evaluate_emulated(
                            config,
                            &policy.primary_provider,
                            &policy.primary_text_model_id,
                            spec,
                            context_text,
                            media,
                        )
                        .await
                    }
                    Ok(None) => Err(DriverError::Config(format!(
                        "missing API key for native decision provider `{provider}`"
                    ))),
                    Err(e) => Err(DriverError::Config(e.to_string())),
                }
            }
            DecisionExecutionMode::EmulatedStructuredTextModel {
                provider, model, ..
            } => {
                self.evaluate_emulated(config, &provider, &model, spec, context_text, media)
                    .await
            }
        }
    }

    /// Evaluates `spec` on any general text/VLM model via [`synthesize_emulated_decision_prompt`].
    pub async fn evaluate_emulated(
        &self,
        config: &JikuConfig,
        provider_id: &str,
        model: &str,
        spec: &DecisionBatchSpec,
        context_text: &str,
        media: Option<&DriverMediaInput>,
    ) -> Result<DecisionBatchResponse, DriverError> {
        spec.validate()
            .map_err(|err| DriverError::Validation(format!("{err:?}")))?;

        let provider = config
            .providers
            .get(provider_id)
            .ok_or_else(|| DriverError::Config(format!("unknown provider `{provider_id}`")))?;
        let planner_role = config.roles.get("planner");
        let thinking_enabled = config.resolve_thinking_enabled(planner_role, provider);

        let schema_prompt = synthesize_emulated_decision_prompt(&spec.questions);
        let full_prompt = if context_text.trim().is_empty() {
            schema_prompt
        } else {
            format!("Context:\n{context_text}\n\n{schema_prompt}")
        };

        let mut media_vec = Vec::new();
        if let Some(m) = media {
            media_vec.push(m.clone());
        }

        let turn_resp = self
            .turn
            .generate(
                config,
                &TurnRequest {
                    provider_id: provider_id.to_string(),
                    model: model.to_string(),
                    system_prompt: "You are Jiku's deterministic decision evaluator. Output strictly valid JSON with no commentary.".to_string(),
                    messages: vec![TurnMessage {
                        role: "user".to_string(),
                        text: full_prompt,
                        media: media_vec,
                    }],
                    temperature: Some(0.0),
                    max_tokens: Some(1024),
                    json_object: true,
                    stream: false,
                    thinking_enabled,
                },
            )
            .await?;

        let json_slice = strip_markdown_json_fences(&turn_resp.text);
        let parsed: Value = serde_json::from_str(json_slice).map_err(|err| {
            DriverError::InvalidPayload(format!(
                "emulated decision response was not valid JSON ({err}): {}",
                turn_resp.text
            ))
        })?;

        let answers_obj = parsed
            .get("answers")
            .or_else(|| parsed.get("results"))
            .and_then(Value::as_object)
            .ok_or_else(|| {
                DriverError::InvalidPayload(
                    "emulated decision JSON missing top-level `answers` object".to_string(),
                )
            })?;

        let mut answers = BTreeMap::new();
        for (qid, kind) in &spec.questions {
            let val = answers_obj.get(qid).ok_or_else(|| {
                DriverError::InvalidPayload(format!(
                    "emulated decision JSON omitted required question `{qid}`"
                ))
            })?;
            let normalized = normalize_emulated_answer(kind, val)?;
            answers.insert(qid.clone(), normalized);
        }

        Ok(DecisionBatchResponse {
            execution_mode: "emulated_structured_text_model".to_string(),
            provider_id: provider_id.to_string(),
            model: model.to_string(),
            answers,
            usage: DecisionUsage {
                total_input_tokens: turn_resp.prompt_tokens,
                total_output_tokens: turn_resp.completion_tokens,
                cached_tokens: 0,
            },
        })
    }
}

#[derive(Debug, Deserialize)]
struct RawDecisionsWireResponse {
    #[serde(default)]
    answers: BTreeMap<String, DecisionAnswer>,
    #[serde(default)]
    results: BTreeMap<String, DecisionAnswer>,
    #[serde(default)]
    usage: DecisionUsage,
}

fn build_decisions_input_value(
    context_text: &str,
    media_source: &DecisionMediaSource,
    media: Option<&DriverMediaInput>,
) -> Result<Value, DriverError> {
    let mut blocks = Vec::new();
    match media {
        Some(DriverMediaInput::Inline { mime_type, bytes }) => {
            if bytes.is_empty() {
                if !context_text.trim().is_empty() {
                    blocks.push(json!({ "type": "text", "text": context_text }));
                }
            } else {
                let block_type = decision_block_type_for_mime(mime_type);
                blocks.push(json!({
                    "type": block_type,
                    "mime_type": mime_type,
                    "data": BASE64_STD.encode(bytes),
                }));
                if !context_text.trim().is_empty() {
                    blocks.push(json!({ "type": "text", "text": context_text }));
                }
            }
        }
        Some(DriverMediaInput::FilesUri { uri }) => {
            blocks.push(json!({ "type": "file", "file_uri": uri }));
            if !context_text.trim().is_empty() {
                blocks.push(json!({ "type": "text", "text": context_text }));
            }
        }
        None => {
            if let DecisionMediaSource::FilesApi { file_uri } = media_source {
                blocks.push(json!({ "type": "file", "file_uri": file_uri }));
            }
            if !context_text.trim().is_empty() {
                blocks.push(json!({ "type": "text", "text": context_text }));
            }
        }
    }

    if blocks.is_empty() {
        return Err(DriverError::Validation(
            "decisions batch requires non-empty context_text or media payload".to_string(),
        ));
    }
    Ok(Value::Array(blocks))
}

fn decision_block_type_for_mime(mime: &str) -> &'static str {
    let m = mime.trim().to_ascii_lowercase();
    if m.starts_with("image/") {
        "image"
    } else if m.starts_with("audio/") {
        "audio"
    } else if m.starts_with("video/") {
        "video"
    } else {
        "document"
    }
}

fn normalize_emulated_answer(
    expected: &DecisionQuestionKind,
    val: &Value,
) -> Result<DecisionAnswer, DriverError> {
    if let Ok(exact) = serde_json::from_value::<DecisionAnswer>(val.clone()) {
        return Ok(exact);
    }
    match expected {
        DecisionQuestionKind::Noul { .. } => {
            let p = val
                .get("noul")
                .or_else(|| val.get("probability"))
                .and_then(Value::as_f64)
                .unwrap_or(0.5) as f32;
            let conf = val
                .get("confidence")
                .and_then(Value::as_f64)
                .map(|c| c as f32)
                .unwrap_or_else(|| p.max(1.0 - p));
            Ok(DecisionAnswer::Noul {
                noul: p.clamp(0.01, 0.99),
                confidence: conf.clamp(0.0, 1.0),
            })
        }
        DecisionQuestionKind::Choice { choices, .. } => {
            let chosen = val
                .get("choice")
                .and_then(Value::as_str)
                .unwrap_or_else(|| {
                    choices
                        .last()
                        .map(String::as_str)
                        .unwrap_or("NONE_OR_OTHER")
                })
                .to_string();
            let mut probs = BTreeMap::new();
            if let Some(map) = val.get("probabilities").and_then(Value::as_object) {
                for (k, v) in map {
                    if let Some(f) = v.as_f64() {
                        probs.insert(k.clone(), f as f32);
                    }
                }
            }
            let conf = val
                .get("confidence")
                .and_then(Value::as_f64)
                .map(|v| v as f32)
                .or_else(|| probs.get(&chosen).copied())
                .unwrap_or(0.5);
            if probs.is_empty() {
                probs.insert(chosen.clone(), conf);
            }
            Ok(DecisionAnswer::Choice {
                choice: chosen,
                confidence: conf,
                probabilities: probs,
            })
        }
        DecisionQuestionKind::Score { rubric, .. } => {
            let score = val.get("score").and_then(Value::as_f64).unwrap_or(1.0) as f32;
            let conf = val
                .get("confidence")
                .and_then(Value::as_f64)
                .map(|v| v as f32)
                .unwrap_or(0.5);
            let mut legend = BTreeMap::new();
            for (idx, desc) in rubric.iter().enumerate() {
                legend.insert(idx.to_string(), desc.clone());
            }
            Ok(DecisionAnswer::Score {
                score,
                confidence: conf,
                probabilities: BTreeMap::new(),
                legend,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MANDATORY_CHOICE_FALLBACK;

    #[test]
    fn native_wire_payload_uses_verified_input_and_questions_contract() {
        let spec = DecisionBatchSpec {
            model: GEMINI_DECISION_1_MODEL_ID.to_string(),
            media_source: DecisionMediaSource::InlineBytes {
                mime_type: "image/png".to_string(),
            },
            questions: vec![
                (
                    "q_in_scope".to_string(),
                    DecisionQuestionKind::Noul {
                        instructions: "Is the terminal window visible?".to_string(),
                    },
                ),
                (
                    "q_target".to_string(),
                    DecisionQuestionKind::Choice {
                        instructions: "Which cell contains the cursor?".to_string(),
                        choices: vec!["G8".to_string(), MANDATORY_CHOICE_FALLBACK.to_string()],
                        scope_gate_question_id: "q_in_scope".to_string(),
                    },
                ),
            ],
        };
        let media = DriverMediaInput::Inline {
            mime_type: "image/png".to_string(),
            bytes: vec![137, 80, 78, 71],
        };
        let wire =
            DecisionsClient::build_native_wire_payload(&spec, "Check focus", Some(&media)).unwrap();
        assert_eq!(wire["model"], "gemini-decision-1");
        assert!(wire["input"].is_array());
        assert_eq!(wire["input"][0]["type"], "image");
        assert_eq!(wire["input"][1]["type"], "text");
        assert!(wire.get("contents").is_none());
        assert_eq!(wire["questions"]["q_target"]["criteria"]["G8"], "G8");
    }

    #[test]
    fn parses_live_gemini_decision_1_answer_shape() {
        let sample = r#"{
            "answers": {
                "q_in_scope": { "type": "noul", "noul": 0.91, "confidence": 0.91 },
                "q_target": {
                    "type": "choice",
                    "choice": "G8",
                    "confidence": 0.86,
                    "probabilities": { "G8": 0.86, "NONE_OR_OTHER": 0.14 }
                },
                "q_ready": {
                    "type": "score",
                    "score": 0.68,
                    "confidence": 0.68,
                    "probabilities": { "0": 0.32, "1": 0.68 },
                    "legend": { "0": "Not ready", "1": "Ready" }
                }
            },
            "usage": { "total_input_tokens": 512, "total_output_tokens": 0 }
        }"#;
        let parsed: RawDecisionsWireResponse = serde_json::from_str(sample).unwrap();
        assert_eq!(parsed.usage.total_output_tokens, 0);
        assert_eq!(parsed.answers["q_target"].top_choice(), Some(("G8", 0.86)));
        assert_eq!(parsed.answers["q_in_scope"].noul_prob(), Some(0.91));
    }
}
