---
name: jiku-desktop-orchestration
description: Standard operating rules for driving desktop automation safely with Jiku over Koma Wayland IPC.
---

# Jiku Desktop Orchestration Skill

Use this bundled SOP whenever dispatching tasks through `jiku` / `jikud` into [Koma](https://github.com/komawm/koma).

## Safety & Concurrency Invariants

1. **Observe Before Mutating (`capture.frame` -> `Epoch`)**: Always read the active workspace inventory and capture a fresh `grid_225` frame over Koma's `SCM_RIGHTS` sealed `memfd` channel before synthesizing clicks or text injection.
2. **Respect `Epoch` Fences (`4090`) & `Mod+G` Preemption (`4091`)**: Never blind-retry `vseat.*` mutations on stale epochs without re-observing; stop immediately when the human preempts (`4091`) or hits a protected blackout surface (`4032`).
3. **Human Verification Gate**: Halt autonomous execution whenever the calibrated safety `Noul` probability triggers `RequireHumanVerification`, emitting the `security.verification_needed` Lua hook and desktop notification.
4. **Two-Phase Skill Promotion**: Candidate Lua or Markdown SOP skills emitted during tasks enter `staged_skills` in `~/.local/state/jiku/state.db`, sanitized to block `io.*`, `os.*`, `debug.*`, `package.*`, `loadfile`, `dofile`, `load(`, and `require(`. Review diffs with `jiku skills diff <id>` before running `jiku skills approve <id>`.
5. **JikuBox Triage**: Use `jiku box classify [dir]` to batch-tag multimodal dropzone artifacts with binary `Noul` scope/safety gates and categorical `Choice` + `NONE_OR_OTHER` fallback.
