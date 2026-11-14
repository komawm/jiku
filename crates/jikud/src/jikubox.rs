//! `~/JikuBox/` multimodal batch classification and two-stage checklist triage (`JIK-0007`).

use std::{
    fs,
    path::{Path, PathBuf},
};

use jiku_memory::expand_tilde_path;
use serde::{Deserialize, Serialize};

use crate::{
    DecisionBatchSpec, DecisionMediaSource, DecisionQuestionKind, GEMINI_DECISION_1_MODEL_ID,
    MANDATORY_CHOICE_FALLBACK,
    config::JikuConfig,
    decompose_rubric_to_noul_checklist,
    drivers::{
        DriverError, DriverMediaInput,
        decisions::{DecisionAnswer, DecisionExecutor},
    },
};

const TRIAGE_CATEGORIES: &[&str] = &[
    "screenshot_ui",
    "document_spec",
    "code_or_config",
    "receipt_or_invoice",
    "media_asset",
    MANDATORY_CHOICE_FALLBACK,
];

/// Classification summary produced for a single file inside `~/JikuBox/`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JikuBoxItemReport {
    /// Artifact filename relative to the target directory.
    pub file_name: String,
    /// Detected MIME type.
    pub mime_type: String,
    /// Byte length on disk.
    pub size_bytes: u64,
    /// Primary category selected by `DecisionsDriver` (or emulated planner).
    pub category: String,
    /// Confidence score of the selected category.
    pub category_confidence: f32,
    /// Whether `q_relevant` deemed the artifact relevant (`noul` probability).
    pub relevant_prob: f32,
    /// Whether `q_sensitive` detected credentials/PII risk (`noul` probability).
    pub sensitive_prob: f32,
    /// Decomposed `Noul` checklist probabilities (`criterion_id -> yes_prob`).
    pub checklist: Vec<(String, f32)>,
    /// Driver execution mode used (`"native_decisions_api"` or `"emulated_structured_text_model"`).
    pub execution_mode: String,
    /// Input/output token accounting for this artifact.
    pub input_tokens: u64,
    /// Output tokens generated (`0` on native `gemini-decision-1`).
    pub output_tokens: u64,
}

/// Aggregate triage report emitted by [`classify_dropbox_dir`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JikuBoxReport {
    /// Resolved directory scanned.
    pub directory: String,
    /// Total files examined.
    pub total_files: usize,
    /// Successfully classified artifacts.
    pub items: Vec<JikuBoxItemReport>,
    /// Files skipped due to unsupported extensions or size ceilings.
    pub skipped: Vec<(String, String)>,
}

impl JikuBoxReport {
    /// Renders a concise Markdown triage report suitable for CLI output or archiving.
    #[must_use]
    pub fn to_markdown(&self) -> String {
        let mut out = format!(
            "# JikuBox Triage Report\n\n- Directory: `{}`\n- Total entries scanned: {}\n- Classified: {} | Skipped: {}\n\n| File | Category | Conf | Relevant | Sensitive | Mode | Out Tokens |\n| --- | --- | ---: | ---: | ---: | --- | ---: |\n",
            self.directory,
            self.total_files,
            self.items.len(),
            self.skipped.len()
        );
        for item in &self.items {
            out.push_str(&format!(
                "| `{}` | `{}` | {:.2} | {:.2} | {:.2} | `{}` | {} |\n",
                item.file_name,
                item.category,
                item.category_confidence,
                item.relevant_prob,
                item.sensitive_prob,
                item.execution_mode,
                item.output_tokens,
            ));
        }
        out
    }
}

/// Scans `dir` deterministically and evaluates each supported artifact with
/// two-stage `Noul` scope/safety gates, categorical tagging, and decomposed checklist rubrics.
pub async fn classify_dropbox_dir(
    config: &JikuConfig,
    executor: &DecisionExecutor,
    dir: &Path,
    force_emulate: bool,
) -> Result<JikuBoxReport, DriverError> {
    let resolved_dir = expand_tilde_path(dir);
    if !resolved_dir.is_dir() {
        return Err(DriverError::Validation(format!(
            "target dropbox directory does not exist: {}",
            resolved_dir.display()
        )));
    }

    let mut entries: Vec<PathBuf> = fs::read_dir(&resolved_dir)
        .map_err(|e| DriverError::Validation(format!("failed listing dropbox: {e}")))?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.is_file())
        .collect();
    entries.sort();

    let policy = config.to_decision_router_policy();
    let mut items = Vec::new();
    let mut skipped = Vec::new();

    for path in &entries {
        let file_name = path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "unknown".to_string());
        let meta = match fs::metadata(path) {
            Ok(m) => m,
            Err(e) => {
                skipped.push((file_name, format!("metadata error: {e}")));
                continue;
            }
        };
        if meta.len() == 0 || meta.len() > 8 * 1024 * 1024 {
            skipped.push((file_name, format!("unsupported size {} bytes", meta.len())));
            continue;
        }

        let Some(kind) = detect_artifact_kind(path) else {
            skipped.push((file_name, "unsupported file extension".to_string()));
            continue;
        };

        let raw_bytes = match fs::read(path) {
            Ok(b) => b,
            Err(e) => {
                skipped.push((file_name, format!("read error: {e}")));
                continue;
            }
        };

        let (context_text, media_source, media_input) = match kind {
            ArtifactKind::BinaryMedia { mime_type } => (
                format!("Classify dropped artifact `{file_name}` (`{mime_type}`)."),
                DecisionMediaSource::InlineBytes {
                    mime_type: mime_type.to_string(),
                },
                Some(DriverMediaInput::Inline {
                    mime_type: mime_type.to_string(),
                    bytes: raw_bytes,
                }),
            ),
            ArtifactKind::TextSnippet { mime_type } => {
                let text_preview: String = String::from_utf8_lossy(&raw_bytes)
                    .chars()
                    .take(12_000)
                    .collect();
                (
                    format!(
                        "Artifact filename: `{file_name}` (type `{mime_type}`).\n\nContent:\n{text_preview}"
                    ),
                    // Decisions API inline media validator applies when media is attached;
                    // use valid image/png mime marker for spec contract while omitting empty bytes.
                    DecisionMediaSource::InlineBytes {
                        mime_type: "image/png".to_string(),
                    },
                    None,
                )
            }
        };

        let mut questions = vec![
            (
                "q_relevant".to_string(),
                DecisionQuestionKind::Noul {
                    instructions: "Is this artifact a meaningful desktop, engineering, financial, or media item worth organizing?".to_string(),
                },
            ),
            (
                "q_sensitive".to_string(),
                DecisionQuestionKind::Noul {
                    instructions: "Does this artifact appear to contain private passwords, API keys, OTP codes, or sensitive personal identifiers?".to_string(),
                },
            ),
            (
                "q_category".to_string(),
                DecisionQuestionKind::Choice {
                    instructions: "Which primary category best describes this artifact?".to_string(),
                    choices: TRIAGE_CATEGORIES.iter().map(|s| (*s).to_string()).collect(),
                    scope_gate_question_id: "q_relevant".to_string(),
                },
            ),
        ];
        if config.jikubox.triage.prefer_noul_checklists {
            questions.extend(decompose_rubric_to_noul_checklist(
                "triage",
                &[
                    "Is the text or visual content clearly legible?",
                    "Does this item require follow-up user review or action?",
                ],
            ));
        }

        let spec = DecisionBatchSpec {
            model: GEMINI_DECISION_1_MODEL_ID.to_string(),
            media_source,
            questions,
        };

        let resp = executor
            .evaluate(
                config,
                &policy,
                &spec,
                &context_text,
                media_input.as_ref(),
                force_emulate,
                true,
            )
            .await?;

        let relevant_prob = resp
            .answers
            .get("q_relevant")
            .and_then(DecisionAnswer::noul_prob)
            .unwrap_or(0.5);
        let sensitive_prob = resp
            .answers
            .get("q_sensitive")
            .and_then(DecisionAnswer::noul_prob)
            .unwrap_or(0.0);
        let (category, category_confidence) = resp
            .answers
            .get("q_category")
            .and_then(DecisionAnswer::top_choice)
            .map(|(c, p)| (c.to_string(), p))
            .unwrap_or_else(|| (MANDATORY_CHOICE_FALLBACK.to_string(), 0.0));

        let mut checklist = Vec::new();
        for (qid, ans) in &resp.answers {
            if qid.starts_with("triage_check_")
                && let Some(p) = ans.noul_prob()
            {
                checklist.push((qid.clone(), p));
            }
        }

        items.push(JikuBoxItemReport {
            file_name,
            mime_type: kind.mime().to_string(),
            size_bytes: meta.len(),
            category,
            category_confidence,
            relevant_prob,
            sensitive_prob,
            checklist,
            execution_mode: resp.execution_mode,
            input_tokens: resp.usage.total_input_tokens,
            output_tokens: resp.usage.total_output_tokens,
        });
    }

    Ok(JikuBoxReport {
        directory: resolved_dir.display().to_string(),
        total_files: entries.len(),
        items,
        skipped,
    })
}

enum ArtifactKind {
    BinaryMedia { mime_type: &'static str },
    TextSnippet { mime_type: &'static str },
}

impl ArtifactKind {
    const fn mime(&self) -> &'static str {
        match self {
            Self::BinaryMedia { mime_type } | Self::TextSnippet { mime_type } => mime_type,
        }
    }
}

fn detect_artifact_kind(path: &Path) -> Option<ArtifactKind> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)?;
    match ext.as_str() {
        "png" => Some(ArtifactKind::BinaryMedia {
            mime_type: "image/png",
        }),
        "jpg" | "jpeg" => Some(ArtifactKind::BinaryMedia {
            mime_type: "image/jpeg",
        }),
        "webp" => Some(ArtifactKind::BinaryMedia {
            mime_type: "image/webp",
        }),
        "pdf" => Some(ArtifactKind::BinaryMedia {
            mime_type: "application/pdf",
        }),
        "txt" => Some(ArtifactKind::TextSnippet {
            mime_type: "text/plain",
        }),
        "md" => Some(ArtifactKind::TextSnippet {
            mime_type: "text/markdown",
        }),
        "json" => Some(ArtifactKind::TextSnippet {
            mime_type: "application/json",
        }),
        "toml" => Some(ArtifactKind::TextSnippet {
            mime_type: "application/toml",
        }),
        "rs" | "lua" | "py" | "sh" | "yaml" | "yml" => Some(ArtifactKind::TextSnippet {
            mime_type: "text/plain",
        }),
        _ => None,
    }
}
