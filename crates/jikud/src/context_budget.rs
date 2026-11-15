//! Context window token budgeting, estimation, and 4-stage progressive
//! compaction waterfall for small-context VLM/LLM backends (`4K–8K` contexts).
//!
//! Designed so local `llama-server` / OpenAI-compatible 4096-token models never
//! receive uncompacted payloads or hardcoded `max_tokens: 8192` parameters
//! that trigger HTTP `400 Bad Request` context-length overflows.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::{
    a11y::{A11yEntry, A11ySnapshot, REDACTED_VALUE},
    drivers::turn::TurnMessage,
    grounding::SystemGroundingPack,
};

/// Safety headroom reserved inside `max_context_tokens` for JSON schema
/// punctuation, chat-template tokens, and tokenizer drift.
pub const CONTEXT_SAFETY_HEADROOM_TOKENS: u32 = 64;

/// Maximum token budget enforced for [`DEFAULT_MICRO_SYSTEM_PROMPT`] and
/// custom micro system prompts in Stage 4.
pub const MICRO_SYSTEM_PROMPT_MAX_TOKENS: u32 = 220;

/// Ultra-compact desktop planner system prompt (`<= 220` tokens) preserving
/// all step variants and epoch/surface safety guardrails for Stage 4 fallback.
pub const DEFAULT_MICRO_SYSTEM_PROMPT: &str = "\
You are jikud desktop planner. Reply ONLY with valid JSON: {\"summary\":str,\"abort_reason\"?:str,\"steps\":[...]}.\n\
Step variants:\n\
- {\"kind\":\"path\",\"waypoints\":[{\"x\":f,\"y\":f,\"t_ms\"?:u64}],\"dwell_ms\"?:u64,\"pressure\"?:f}\n\
- {\"kind\":\"wheel\",\"x\"?:f,\"y\"?:f,\"dx\":f,\"dy\":f,\"discrete_x\":i32,\"discrete_y\":i32,\"modifiers\":[str]}\n\
- {\"kind\":\"relative_move\",\"dx\":f,\"dy\":f,\"modifiers\":[str]}\n\
- {\"kind\":\"click\",\"button\":str,\"x\"?:f,\"y\"?:f,\"modifiers\":[str]}\n\
- {\"kind\":\"text\",\"text\":str}|{\"kind\":\"key_combo\",\"keys\":[str]}|{\"kind\":\"wait_ms\",\"ms\":u64}\n\
Safety: never forge secrets; halt on protected surface (4032) or human Mod+G preempt (4091); re-observe on stale epoch (4090); set abort_reason if unsafe/uncertain.";

/// `[context]` budgeting and progressive compaction settings in `JikuConfig`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ContextConfig {
    /// Default model context window in tokens (overridden per role when set).
    pub max_context_tokens: u32,
    /// Target completion token reservation before clamping.
    pub reserved_output_tokens: u32,
    /// Minimum acceptable completion token budget before erroring.
    pub min_output_tokens: u32,
    /// Trigger compaction when `raw_prompt_tokens / max_context_tokens` exceeds this ratio.
    pub compaction_trigger_ratio: f32,
    /// Estimated prompt token cost per attached image frame/ROI crop.
    pub image_token_estimate: u32,
    /// Maximum recent visual attachments retained in Stage 1 on standard contexts.
    pub keep_recent_images: usize,
    /// Maximum AT-SPI entries retained when Stage 2 skeleton folding triggers.
    pub a11y_max_entries_small_ctx: usize,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            max_context_tokens: 8192,
            reserved_output_tokens: 768,
            min_output_tokens: 256,
            compaction_trigger_ratio: 0.72,
            image_token_estimate: 512,
            keep_recent_images: 1,
            a11y_max_entries_small_ctx: 28,
        }
    }
}

/// Structured prompt assembly and context limits passed into [`compact_context_waterfall`].
#[derive(Debug, Clone)]
pub struct CompactionInput {
    /// Primary verbose planner/executor system prompt.
    pub system_prompt: String,
    /// Compact fallback system prompt (`<= 220` tokens) activated in Stage 4.
    pub micro_system_prompt: Option<String>,
    /// Chronological turn messages containing conversation history and the current goal/media.
    pub messages: Vec<TurnMessage>,
    /// Optional structured visual/UI grounding bundle (`SystemGroundingPack`).
    pub grounding_pack: Option<SystemGroundingPack>,
    /// Optional accessibility snapshot tree (`A11ySnapshot`).
    pub a11y_snapshot: Option<A11ySnapshot>,
    /// Chronological historical step/episode log lines (oldest first).
    pub episodic_history: Vec<String>,
    /// Retrieved L1 FTS5 skill snippets.
    pub l1_skill_snippets: Vec<String>,
    /// Caller's requested `max_tokens` (clamped safely against the budget).
    pub requested_output_tokens: Option<u32>,
    /// Effective model context ceiling (defaults to `ContextConfig::max_context_tokens`).
    pub max_context_tokens: Option<u32>,
}

/// Compacted prompt payload ready to populate a [`crate::drivers::turn::TurnRequest`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactedTurnPayload {
    /// Active system prompt (full or micro-mode).
    pub system_prompt: String,
    /// Compacted message sequence with context blocks injected onto the latest user turn.
    pub messages: Vec<TurnMessage>,
    /// Clamped `max_tokens` value guaranteed to fit inside `max_context_tokens - 64`.
    pub max_output_tokens: u32,
}

/// Telemetry emitted alongside every compacted turn and exposed via `context.inspect`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextBudgetReport {
    /// Effective context window enforced for this turn.
    pub max_context_tokens: u32,
    /// Token estimate prior to running the compaction waterfall.
    pub raw_prompt_tokens: u32,
    /// Final prompt token estimate after all applied compaction stages.
    pub compacted_prompt_tokens: u32,
    /// Final `max_tokens` budget after clamping to `max_context_tokens - 64`.
    pub clamped_max_output_tokens: u32,
    /// Ordered list of compaction stages applied (`stage0_passthrough`, `stage1_*`, ...).
    pub stages_applied: Vec<String>,
    /// Total image attachments evicted across messages in Stage 1.
    pub dropped_images: usize,
    /// Total AT-SPI / grounding UI entries pruned in Stage 2+.
    pub truncated_a11y_nodes: usize,
}

/// Estimates tokens in a single UTF-8 string using ASCII (~4 chars/tok) and
/// non-ASCII/CJK (~1.5 chars/tok) heuristics.
///
/// Time complexity: $O(C)$ over UTF-8 characters $C$ in `text`.
/// Space complexity: $O(1)$.
#[must_use]
pub fn estimate_text_tokens(text: &str) -> u32 {
    let mut ascii_chars: u64 = 0;
    let mut non_ascii_chars: u64 = 0;
    for ch in text.chars() {
        if ch.is_ascii() {
            ascii_chars += 1;
        } else {
            non_ascii_chars += 1;
        }
    }
    let ascii_tokens = ascii_chars.div_ceil(4);
    let non_ascii_tokens = (non_ascii_chars.saturating_mul(2)).div_ceil(3);
    u32::try_from(ascii_tokens.saturating_add(non_ascii_tokens)).unwrap_or(u32::MAX)
}

/// Estimates total prompt tokens across `system_prompt`, `messages`, and media frames.
///
/// Time complexity: $O(C + M)$ over UTF-8 characters $C$ across `system_prompt` and
/// `messages` plus media attachment count $M$.
/// Space complexity: $O(1)$.
#[must_use]
pub fn estimate_turn_tokens(
    system_prompt: &str,
    messages: &[TurnMessage],
    image_token_estimate: u32,
) -> u32 {
    let mut total = estimate_text_tokens(system_prompt);
    for msg in messages {
        total = total.saturating_add(estimate_text_tokens(&msg.text));
        let img_tokens = u32::try_from(msg.media.len())
            .unwrap_or(u32::MAX)
            .saturating_mul(image_token_estimate);
        total = total.saturating_add(img_tokens);
    }
    total
}

/// Clamps requested completion output tokens so `prompt_tokens + out <= max_ctx.saturating_sub(64)`.
///
/// Returns `Err` only when `prompt_tokens + min_out > max_ctx`.
///
/// Time complexity: $O(1)$.
/// Space complexity: $O(1)$.
pub fn clamp_max_output_tokens(
    max_ctx: u32,
    prompt_tokens: u32,
    requested_out: u32,
    min_out: u32,
) -> Result<u32, String> {
    if prompt_tokens.saturating_add(min_out) > max_ctx {
        return Err(format!(
            "prompt tokens ({prompt_tokens}) plus min_output_tokens ({min_out}) exceed max_context_tokens ({max_ctx})"
        ));
    }
    let safe_ceiling = max_ctx.saturating_sub(CONTEXT_SAFETY_HEADROOM_TOKENS);
    let available = safe_ceiling.saturating_sub(prompt_tokens);
    let target = requested_out.max(min_out.min(available));
    Ok(target.min(available))
}

/// Runs the 4-stage progressive compaction waterfall so multimodal + AT-SPI +
/// episodic turns fit cleanly within `max_context_tokens`.
///
/// - **Stage 0**: Pass-through when raw tokens `<= max_ctx * compaction_trigger_ratio`.
/// - **Stage 1 (Visual ROI Pruning)**: Retain at most `keep_recent_images` (or 1 ROI/last image
///   when `max_ctx <= 4096` and multiple frames are attached on the current turn), replacing
///   evicted images with `[evicted_frame: ...]`.
/// - **Stage 2 (AT-SPI Skeleton Folding)**: Prune verbose AT-SPI / grounding entries down to
///   top `a11y_max_entries_small_ctx` interactive skeleton entries (`@snap:eN role "name" [actions]`).
/// - **Stage 3 (Episodic L1 Folding)**: Fold history into single-line `[step k: action -> outcome]`
///   records, cap L1 snippets (`<= 2`, `<= 160` chars each), and trim oldest history lines first
///   while preserving the current goal/query.
/// - **Stage 4 (System Prompt Micro-Mode)**: Switch planner instructions to `micro_system_prompt`
///   (`<= 220` tokens) and apply final `clamp_max_output_tokens`.
///
/// Time complexity: $O(T + N + E)$ over text chars $T$, AT-SPI nodes $N$, and episodic lines $E$.
/// Space complexity: $O(T + N + E)$ for the compacted payload buffers.
pub fn compact_context_waterfall(
    cfg: &ContextConfig,
    input: &CompactionInput,
) -> Result<(CompactedTurnPayload, ContextBudgetReport), String> {
    let max_ctx = input.max_context_tokens.unwrap_or(cfg.max_context_tokens);
    let requested_out = input
        .requested_output_tokens
        .unwrap_or(cfg.reserved_output_tokens);
    let trigger_ratio = cfg.compaction_trigger_ratio.clamp(0.1, 0.98);
    let trigger_limit = ((max_ctx as f32) * trigger_ratio).floor() as u32;
    let safe_prompt_ceiling = max_ctx
        .saturating_sub(CONTEXT_SAFETY_HEADROOM_TOKENS)
        .saturating_sub(
            cfg.min_output_tokens
                .min(max_ctx.saturating_sub(CONTEXT_SAFETY_HEADROOM_TOKENS)),
        );
    let target_prompt_limit = trigger_limit.min(safe_prompt_ceiling);

    // Build initial uncompacted view to measure raw token footprint accurately.
    let mut work_messages = if input.messages.is_empty() {
        vec![TurnMessage {
            role: "user".to_string(),
            text: String::new(),
            media: Vec::new(),
        }]
    } else {
        input.messages.clone()
    };
    let mut work_system = input.system_prompt.clone();

    let mut grounding_block = input
        .grounding_pack
        .as_ref()
        .map(SystemGroundingPack::format_prompt_block);
    let mut a11y_block = input
        .a11y_snapshot
        .as_ref()
        .map(A11ySnapshot::format_text_tree);
    let mut episodic_lines: VecDeque<String> = input.episodic_history.iter().cloned().collect();
    let mut l1_snippets: Vec<String> = input.l1_skill_snippets.clone();

    let raw_messages = attach_context_blocks(
        &work_messages,
        grounding_block.as_deref(),
        a11y_block.as_deref(),
        &episodic_lines,
        &l1_snippets,
    );
    let raw_prompt_tokens =
        estimate_turn_tokens(&work_system, &raw_messages, cfg.image_token_estimate);

    if raw_prompt_tokens <= target_prompt_limit {
        let clamped = clamp_max_output_tokens(
            max_ctx,
            raw_prompt_tokens,
            requested_out,
            cfg.min_output_tokens,
        )?;
        let report = ContextBudgetReport {
            max_context_tokens: max_ctx,
            raw_prompt_tokens,
            compacted_prompt_tokens: raw_prompt_tokens,
            clamped_max_output_tokens: clamped,
            stages_applied: vec!["stage0_passthrough".to_string()],
            dropped_images: 0,
            truncated_a11y_nodes: 0,
        };
        return Ok((
            CompactedTurnPayload {
                system_prompt: work_system,
                messages: raw_messages,
                max_output_tokens: clamped,
            },
            report,
        ));
    }

    let mut stages_applied = Vec::new();
    let mut dropped_images = 0usize;
    let mut truncated_a11y_nodes = 0usize;

    // Stage 1: Visual ROI pruning.
    let keep_limit = if max_ctx <= 4096 {
        cfg.keep_recent_images.min(1)
    } else {
        cfg.keep_recent_images
    };
    let total_imgs: usize = work_messages.iter().map(|m| m.media.len()).sum();
    let stage1_dropped = if total_imgs > keep_limit {
        prune_images_keep_recent(&mut work_messages, keep_limit)
    } else {
        0
    };
    if stage1_dropped > 0 {
        dropped_images += stage1_dropped;
        stages_applied.push("stage1_visual_roi_pruning".to_string());
    }

    // Track incremental token budget in O(1) per block update after Stage 1.
    let mut current_tokens = estimate_assembled_tokens(
        cfg.image_token_estimate,
        &work_system,
        &work_messages,
        grounding_block.as_deref(),
        a11y_block.as_deref(),
        &episodic_lines,
        &l1_snippets,
    );

    // Stage 2: AT-SPI skeleton folding & grounding UI entry cap.
    if current_tokens > target_prompt_limit {
        let mut stage2_truncated = 0usize;
        if let Some(snap) = &input.a11y_snapshot {
            let (folded_a11y, dropped) = format_a11y_skeleton(snap, cfg.a11y_max_entries_small_ctx);
            a11y_block = Some(folded_a11y);
            stage2_truncated += dropped;
        }
        if let Some(pack) = &input.grounding_pack {
            let (folded_pack, dropped) =
                format_grounding_compact(pack, cfg.a11y_max_entries_small_ctx);
            grounding_block = Some(folded_pack);
            stage2_truncated += dropped;
        }
        if stage2_truncated > 0 || input.a11y_snapshot.is_some() || input.grounding_pack.is_some() {
            truncated_a11y_nodes += stage2_truncated;
            stages_applied.push("stage2_a11y_skeleton_folding".to_string());
            current_tokens = estimate_assembled_tokens(
                cfg.image_token_estimate,
                &work_system,
                &work_messages,
                grounding_block.as_deref(),
                a11y_block.as_deref(),
                &episodic_lines,
                &l1_snippets,
            );
        }
    }

    // Stage 3: Episodic & L1 skill folding + oldest-first O(E) history trimming.
    if current_tokens > target_prompt_limit {
        let mut did_stage3 =
            !episodic_lines.is_empty() || !l1_snippets.is_empty() || work_messages.len() > 1;

        episodic_lines = episodic_lines
            .iter()
            .enumerate()
            .map(|(idx, line)| fold_episodic_line(line, idx + 1))
            .collect();
        l1_snippets = cap_l1_snippets(&l1_snippets, 2, 160);

        current_tokens = estimate_assembled_tokens(
            cfg.image_token_estimate,
            &work_system,
            &work_messages,
            grounding_block.as_deref(),
            a11y_block.as_deref(),
            &episodic_lines,
            &l1_snippets,
        );

        // Trim oldest episodic lines in O(E) without rescanning the full prompt.
        while current_tokens > target_prompt_limit && !episodic_lines.is_empty() {
            if let Some(removed) = episodic_lines.pop_front() {
                current_tokens = current_tokens.saturating_sub(estimate_text_tokens(&removed) + 1);
                did_stage3 = true;
            }
        }

        // Trim older conversation turns if still over target, keeping final goal message intact.
        if current_tokens > target_prompt_limit && work_messages.len() > 1 {
            let last_idx = work_messages.len() - 1;
            for (idx, msg) in work_messages[..last_idx].iter_mut().enumerate() {
                if current_tokens <= target_prompt_limit {
                    break;
                }
                let old_tokens = estimate_text_tokens(&msg.text);
                if old_tokens > 16 {
                    let marker = format!(
                        "[folded_turn_{}: {}]",
                        idx + 1,
                        truncate_chars_clean(&msg.text, 64)
                    );
                    let new_tokens = estimate_text_tokens(&marker);
                    msg.text = marker;
                    current_tokens =
                        current_tokens.saturating_sub(old_tokens.saturating_sub(new_tokens));
                    did_stage3 = true;
                }
            }
        }

        // Drop L1 snippets if still over target before triggering micro-prompt.
        if current_tokens > target_prompt_limit && !l1_snippets.is_empty() {
            l1_snippets.clear();
            did_stage3 = true;
            current_tokens = estimate_assembled_tokens(
                cfg.image_token_estimate,
                &work_system,
                &work_messages,
                grounding_block.as_deref(),
                a11y_block.as_deref(),
                &episodic_lines,
                &l1_snippets,
            );
        }

        if did_stage3 {
            stages_applied.push("stage3_episodic_l1_folding".to_string());
        }
    }

    // Stage 4: Switch planner system prompt to concise <=220-token micro-mode.
    if current_tokens > target_prompt_limit {
        let candidate_micro = input
            .micro_system_prompt
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or(DEFAULT_MICRO_SYSTEM_PROMPT);
        work_system = cap_prompt_tokens(candidate_micro, MICRO_SYSTEM_PROMPT_MAX_TOKENS);
        stages_applied.push("stage4_system_micro_mode".to_string());

        current_tokens = estimate_assembled_tokens(
            cfg.image_token_estimate,
            &work_system,
            &work_messages,
            grounding_block.as_deref(),
            a11y_block.as_deref(),
            &episodic_lines,
            &l1_snippets,
        );

        // If a massive AT-SPI/grounding/history tail still threatens hard_prompt_limit,
        // tighten non-goal blocks progressively so min_output_tokens fits inside max_ctx - 64.
        if current_tokens > safe_prompt_ceiling {
            if let Some(snap) = &input.a11y_snapshot {
                let cap = cfg.a11y_max_entries_small_ctx.min(8);
                let prev_kept = snap.entries.len().saturating_sub(
                    input
                        .a11y_snapshot
                        .as_ref()
                        .map_or(0, |s| s.entries.len().min(cfg.a11y_max_entries_small_ctx)),
                );
                let (folded_a11y, total_dropped) = format_a11y_skeleton(snap, cap);
                a11y_block = Some(folded_a11y);
                truncated_a11y_nodes += total_dropped.saturating_sub(prev_kept);
            }
            if estimate_assembled_tokens(
                cfg.image_token_estimate,
                &work_system,
                &work_messages,
                grounding_block.as_deref(),
                a11y_block.as_deref(),
                &episodic_lines,
                &l1_snippets,
            ) > safe_prompt_ceiling
                && work_messages.len() > 1
            {
                let latest = work_messages.pop().expect("len > 1 checked");
                work_messages = vec![latest];
            }
            if estimate_assembled_tokens(
                cfg.image_token_estimate,
                &work_system,
                &work_messages,
                grounding_block.as_deref(),
                a11y_block.as_deref(),
                &episodic_lines,
                &l1_snippets,
            ) > safe_prompt_ceiling
            {
                if let Some(snap) = &input.a11y_snapshot {
                    truncated_a11y_nodes = snap.entries.len();
                }
                a11y_block = None;
            }
        }
    }

    let final_messages = attach_context_blocks(
        &work_messages,
        grounding_block.as_deref(),
        a11y_block.as_deref(),
        &episodic_lines,
        &l1_snippets,
    );
    let compacted_prompt_tokens =
        estimate_turn_tokens(&work_system, &final_messages, cfg.image_token_estimate);
    let clamped_max_output_tokens = clamp_max_output_tokens(
        max_ctx,
        compacted_prompt_tokens,
        requested_out,
        cfg.min_output_tokens,
    )?;

    let report = ContextBudgetReport {
        max_context_tokens: max_ctx,
        raw_prompt_tokens,
        compacted_prompt_tokens,
        clamped_max_output_tokens,
        stages_applied,
        dropped_images,
        truncated_a11y_nodes,
    };

    Ok((
        CompactedTurnPayload {
            system_prompt: work_system,
            messages: final_messages,
            max_output_tokens: clamped_max_output_tokens,
        },
        report,
    ))
}

fn prune_images_keep_recent(messages: &mut [TurnMessage], keep_recent: usize) -> usize {
    let total_imgs: usize = messages.iter().map(|m| m.media.len()).sum();
    let to_drop = total_imgs.saturating_sub(keep_recent);
    if to_drop == 0 {
        return 0;
    }
    let mut dropped = 0usize;
    for (msg_idx, msg) in messages.iter_mut().enumerate() {
        if dropped >= to_drop {
            break;
        }
        let drop_here = (to_drop - dropped).min(msg.media.len());
        if drop_here > 0 {
            msg.media.drain(0..drop_here);
            let note = if drop_here == 1 {
                format!("[evicted_frame: msg={} idx=0]", msg_idx + 1)
            } else {
                format!("[evicted_frame: msg={} count={drop_here}]", msg_idx + 1)
            };
            if msg.text.trim().is_empty() {
                msg.text = note;
            } else {
                msg.text = format!("{note}\n{}", msg.text);
            }
            dropped += drop_here;
        }
    }
    dropped
}

fn is_interactive_entry(entry: &A11yEntry) -> bool {
    !entry.actions.is_empty()
        || matches!(
            entry.role.as_str(),
            "button"
                | "push-button"
                | "toggle-button"
                | "text"
                | "entry"
                | "password-text"
                | "menu-item"
                | "check-box"
                | "radio-button"
                | "combo-box"
                | "dialog"
                | "canvas"
                | "drawing-area"
                | "link"
        )
}

fn format_a11y_skeleton(snap: &A11ySnapshot, max_entries: usize) -> (String, usize) {
    let interactive: Vec<&A11yEntry> = snap
        .entries
        .iter()
        .filter(|e| is_interactive_entry(e))
        .collect();
    let non_interactive: Vec<&A11yEntry> = snap
        .entries
        .iter()
        .filter(|e| !is_interactive_entry(e) && e.depth <= 2)
        .collect();

    let mut chosen: Vec<&A11yEntry> = Vec::with_capacity(max_entries.min(snap.entries.len()));
    for entry in interactive {
        if chosen.len() >= max_entries {
            break;
        }
        chosen.push(entry);
    }
    for entry in non_interactive {
        if chosen.len() >= max_entries {
            break;
        }
        chosen.push(entry);
    }

    let dropped = snap.entries.len().saturating_sub(chosen.len());
    let mut out = format!(
        "### AT-SPI SKELETON\nsnapshot=@{} app={:?} kept={}/{} [skeleton]\n",
        snap.snapshot_id,
        snap.app_id,
        chosen.len(),
        snap.entries.len()
    );
    for entry in chosen {
        let actions = if entry.actions.is_empty() {
            String::new()
        } else {
            format!(" [{}]", entry.actions.join(","))
        };
        let rect_part = entry
            .rect
            .map(|[x, y, w, h]| format!(" [{x},{y},{w},{h}]"))
            .unwrap_or_default();
        let val_part = match &entry.value {
            Some(_) if entry.is_secure => format!(" val={REDACTED_VALUE:?}"),
            Some(v) if !v.is_empty() => format!(" val={:?}", truncate_chars_clean(v, 36)),
            _ => String::new(),
        };
        out.push_str(&format!(
            "{} {} {:?}{}{}{}\n",
            entry.ref_id,
            entry.role,
            truncate_chars_clean(&entry.name, 48),
            actions,
            rect_part,
            val_part
        ));
    }
    (out, dropped)
}

fn format_grounding_compact(pack: &SystemGroundingPack, max_entries: usize) -> (String, usize) {
    let mut compact_pack = pack.clone();
    let original_len = compact_pack.top_a11y_entries.len();
    let cap = max_entries.min(12);
    if compact_pack.top_a11y_entries.len() > cap {
        compact_pack.top_a11y_entries.truncate(cap);
    }
    let dropped = original_len.saturating_sub(compact_pack.top_a11y_entries.len());
    (compact_pack.format_prompt_block(), dropped)
}

fn fold_episodic_line(raw: &str, step_idx: usize) -> String {
    let single_line: String = raw.split_whitespace().collect::<Vec<_>>().join(" ");
    if single_line.starts_with("[step ") {
        return truncate_chars_clean(&single_line, 96);
    }
    let compact = truncate_chars_clean(&single_line, 76);
    if let Some((action, outcome)) = compact.split_once("->") {
        format!("[step {step_idx}: {} -> {}]", action.trim(), outcome.trim())
    } else {
        format!("[step {step_idx}: {compact}]")
    }
}

fn cap_l1_snippets(snippets: &[String], max_count: usize, max_chars: usize) -> Vec<String> {
    snippets
        .iter()
        .take(max_count)
        .map(|s| {
            let single_line: String = s.split_whitespace().collect::<Vec<_>>().join(" ");
            truncate_chars_clean(&single_line, max_chars)
        })
        .collect()
}

fn cap_prompt_tokens(prompt: &str, max_tokens: u32) -> String {
    if estimate_text_tokens(prompt) <= max_tokens {
        return prompt.to_string();
    }
    let approx_chars = usize::try_from(max_tokens.saturating_mul(3)).unwrap_or(usize::MAX);
    truncate_chars_clean(prompt, approx_chars)
}

fn truncate_chars_clean(s: &str, max_chars: usize) -> String {
    let mut chars = s.chars();
    let collected: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        if max_chars > 3 {
            let trimmed: String = collected.chars().take(max_chars - 3).collect();
            format!("{trimmed}...")
        } else {
            collected
        }
    } else {
        collected
    }
}

fn attach_context_blocks(
    messages: &[TurnMessage],
    grounding_block: Option<&str>,
    a11y_block: Option<&str>,
    episodic_lines: &VecDeque<String>,
    l1_snippets: &[String],
) -> Vec<TurnMessage> {
    let mut out = messages.to_vec();
    let mut sections: Vec<String> = Vec::new();
    if let Some(g) = grounding_block.map(str::trim).filter(|s| !s.is_empty()) {
        sections.push(g.to_string());
    }
    if let Some(a) = a11y_block.map(str::trim).filter(|s| !s.is_empty()) {
        sections.push(a.to_string());
    }
    if !episodic_lines.is_empty() {
        let joined: Vec<&str> = episodic_lines.iter().map(String::as_str).collect();
        sections.push(format!("### EPISODIC HISTORY\n{}", joined.join("\n")));
    }
    if !l1_snippets.is_empty() {
        sections.push(format!("### L1 SKILLS\n{}", l1_snippets.join("\n")));
    }
    if sections.is_empty() {
        return out;
    }

    let bundle = sections.join("\n\n");
    if let Some(last_user) = out.iter_mut().rev().find(|m| m.role == "user") {
        if last_user.text.trim().is_empty() {
            last_user.text = bundle;
        } else {
            last_user.text = format!("{}\n\n{bundle}", last_user.text);
        }
    } else if let Some(last) = out.last_mut() {
        if last.text.trim().is_empty() {
            last.text = bundle;
        } else {
            last.text = format!("{}\n\n{bundle}", last.text);
        }
    } else {
        out.push(TurnMessage {
            role: "user".to_string(),
            text: bundle,
            media: Vec::new(),
        });
    }
    out
}

fn estimate_assembled_tokens(
    image_token_estimate: u32,
    system_prompt: &str,
    messages: &[TurnMessage],
    grounding_block: Option<&str>,
    a11y_block: Option<&str>,
    episodic_lines: &VecDeque<String>,
    l1_snippets: &[String],
) -> u32 {
    let mut total = estimate_turn_tokens(system_prompt, messages, image_token_estimate);
    if let Some(g) = grounding_block {
        total = total.saturating_add(estimate_text_tokens(g));
    }
    if let Some(a) = a11y_block {
        total = total.saturating_add(estimate_text_tokens(a));
    }
    if !episodic_lines.is_empty() {
        total = total.saturating_add(6);
        for line in episodic_lines {
            total = total.saturating_add(estimate_text_tokens(line));
        }
    }
    if !l1_snippets.is_empty() {
        total = total.saturating_add(4);
        for s in l1_snippets {
            total = total.saturating_add(estimate_text_tokens(s));
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        a11y::{InteractionRoute, RouteDecision},
        drivers::DriverMediaInput,
    };

    fn fake_png_media(tag: u8) -> DriverMediaInput {
        DriverMediaInput::Inline {
            mime_type: "image/png".to_string(),
            bytes: vec![tag; 64],
        }
    }

    fn make_large_a11y_snapshot(count: usize) -> A11ySnapshot {
        let mut entries = Vec::with_capacity(count);
        for i in 0..count {
            let interactive = i % 3 == 0;
            entries.push(A11yEntry {
                ref_id: format!("@s9:e{}", i + 1),
                raw_ref: format!("e{}", i + 1),
                role: if interactive { "button" } else { "panel" }.to_string(),
                name: format!(
                    "Control Item {} with long descriptive accessible label padding for benchmark",
                    i + 1
                ),
                value: Some(format!("detail payload string number {}", i + 1)),
                rect: Some([10 + (i as i32) * 2, 20 + (i as i32), 120, 32]),
                actions: if interactive {
                    vec!["click".to_string(), "focus".to_string()]
                } else {
                    Vec::new()
                },
                depth: if interactive { 1 } else { 4 },
                is_secure: false,
                child_count: 0,
            });
        }
        A11ySnapshot {
            snapshot_id: "s9".to_string(),
            app_id: "virtual-paint".to_string(),
            window_title: "Untitled Canvas".to_string(),
            entries,
            total_raw_nodes: count,
            collapsed_wrappers_count: 12,
            truncated: false,
            skeleton: false,
            warning: None,
            addresses: Default::default(),
        }
    }

    #[test]
    fn token_estimation_weights_ascii_cjk_and_images() {
        let msgs = vec![TurnMessage {
            role: "user".to_string(),
            text: "hello world!".to_string(), // 12 ASCII -> 3 tokens
            media: vec![fake_png_media(1), fake_png_media(2)],
        }];
        let total = estimate_turn_tokens("测试系统", &msgs, 512);
        // 4 non-ASCII chars -> (4*2).div_ceil(3) = 3; 12 ASCII -> 3; 2 * 512 = 1024
        assert_eq!(total, 3 + 3 + 1024);
    }

    #[test]
    fn clamp_max_output_tokens_enforces_headroom_and_rejects_overflow() {
        // Hardcoded 8192 request against 4096 ctx + 3000 prompt must clamp cleanly <= 4032 total.
        let clamped = clamp_max_output_tokens(4096, 3000, 8192, 256).expect("should fit");
        assert_eq!(clamped, 1032);
        assert!(3000 + clamped <= 4032);

        // When prompt + min_out exceeds max_ctx, returns Err.
        let err = clamp_max_output_tokens(4096, 3900, 8192, 256).unwrap_err();
        assert!(err.contains("exceed max_context_tokens"));
    }

    #[test]
    fn waterfall_compacts_7500_plus_payload_to_fit_4096_llama_server() {
        let cfg = ContextConfig {
            keep_recent_images: 1,
            a11y_max_entries_small_ctx: 28,
            ..ContextConfig::default()
        };
        let verbose_sys = "You are Jiku desktop planner. ".repeat(110); // ~825 tokens
        let snap = make_large_a11y_snapshot(120);
        let episodic: Vec<String> = (1..=65)
            .map(|i| {
                format!(
                    "Detailed multi-sentence episode execution log #{i}: moved cursor towards palette toolbar -> stroke verified on pixel cluster region ({i}, {})",
                    i * 3
                )
            })
            .collect();
        let l1: Vec<String> = (1..=6)
            .map(|i| {
                format!(
                    "Skill snippet #{i}: long explanation about drawing strokes accurately with smooth mouse paths, avoiding dialog overlays, verifying RGB diff changes across canvas bounds."
                )
            })
            .collect();
        let grounding = SystemGroundingPack {
            route: RouteDecision {
                route: InteractionRoute::PrecisionCanvas,
                prefer_roi_crop: true,
                recommended_profile: "canvas",
                recommended_hz: 60,
                reason: "canvas goal".to_string(),
            },
            cursor_xy: Some((200.0, 200.0)),
            active_window_size: (640, 480),
            hit_test_summary: Some("clear_unoccluded".to_string()),
            top_a11y_entries: snap.entries.iter().take(40).cloned().collect(),
            last_diff: None,
            canvas_bounds: Some([120, 60, 400, 360]),
        };

        let messages = vec![
            TurnMessage {
                role: "user".to_string(),
                text: "Initial screenshot inspection turn.".to_string(),
                media: vec![fake_png_media(1), fake_png_media(2)],
            },
            TurnMessage {
                role: "user".to_string(),
                text: "GOAL: Draw a green diagonal stroke on the canvas center.".to_string(),
                media: vec![fake_png_media(3), fake_png_media(4)], // ruler + ROI crop
            },
        ];

        let input = CompactionInput {
            system_prompt: verbose_sys,
            micro_system_prompt: None,
            messages,
            grounding_pack: Some(grounding),
            a11y_snapshot: Some(snap),
            episodic_history: episodic,
            l1_skill_snippets: l1,
            requested_output_tokens: Some(8192), // intentionally oversized
            max_context_tokens: Some(4096),
        };

        let (payload, report) =
            compact_context_waterfall(&cfg, &input).expect("7500+ payload should compact cleanly");

        assert!(
            report.raw_prompt_tokens >= 7500,
            "expected raw tokens >= 7500, got {}",
            report.raw_prompt_tokens
        );
        assert!(
            report.compacted_prompt_tokens + report.clamped_max_output_tokens <= 4032,
            "overflow risk for 4096 context: prompt={} + out={}",
            report.compacted_prompt_tokens,
            report.clamped_max_output_tokens
        );
        assert!(report.clamped_max_output_tokens >= cfg.min_output_tokens);
        assert!(report.dropped_images >= 3);
        assert!(report.truncated_a11y_nodes > 0);
        assert!(payload.max_output_tokens == report.clamped_max_output_tokens);
        assert!(
            payload
                .messages
                .last()
                .expect("last user turn")
                .text
                .contains("GOAL: Draw a green diagonal stroke")
        );
        assert!(
            estimate_text_tokens(DEFAULT_MICRO_SYSTEM_PROMPT) <= MICRO_SYSTEM_PROMPT_MAX_TOKENS
        );
    }

    #[test]
    fn stage0_passthrough_preserves_small_payload_unchanged() {
        let cfg = ContextConfig::default();
        let input = CompactionInput {
            system_prompt: "Compact system prompt.".to_string(),
            micro_system_prompt: None,
            messages: vec![TurnMessage {
                role: "user".to_string(),
                text: "Click OK button.".to_string(),
                media: vec![fake_png_media(1)],
            }],
            grounding_pack: None,
            a11y_snapshot: None,
            episodic_history: vec!["[step 1: observe -> ok]".to_string()],
            l1_skill_snippets: Vec::new(),
            requested_output_tokens: Some(512),
            max_context_tokens: Some(8192),
        };

        let (payload, report) = compact_context_waterfall(&cfg, &input).expect("should pass");
        assert_eq!(
            report.stages_applied,
            vec!["stage0_passthrough".to_string()]
        );
        assert_eq!(report.dropped_images, 0);
        assert_eq!(report.truncated_a11y_nodes, 0);
        assert_eq!(payload.max_output_tokens, 512);
    }
}
