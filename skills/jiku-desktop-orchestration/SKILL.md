---
name: jiku-desktop-orchestration
description: >
    Drive real Linux desktop applications through Jiku and the Koma Wayland
    compositor. Load when a task needs genuine screen interaction — clicking,
    typing into native windows, scrolling, or verifying rendered UI — that shell
    commands or headless browser automation cannot cover.
license: MIT
metadata:
    author: Xuepoo
    version: "0.0.1"
    requires: "koma >= 0.0.1, jiku >= 0.0.1"
---

# Jiku Desktop Orchestration

Jiku is a same-UID desktop agent that drives [Koma](https://github.com/komawm/koma)
workspaces over `koma-agent.sock`. It plans a goal, observes a workspace, and
mutates it through capability-scoped, epoch-fenced RPCs.

## When to Apply

Use Jiku when the task needs the actual desktop:

- Verifying a native GUI application's layout or behaviour after a code change.
- Operating an application that exposes no CLI, HTTP API, or automation harness.
- Reproducing a user-reported visual bug that only appears on a real compositor.
- Driving a browser only when the workflow genuinely depends on the user's
  session, profile, or window management — otherwise prefer headless
  Playwright/Chromium.

Do **not** use Jiku for file edits, builds, tests, or anything a shell can do.
Jiku has no shell tool and should not gain one.

## Prerequisites

Check these before planning work, because every failure below looks like a
mysterious tool error later:

```bash
komactl inventory          # a Koma compositor session must be reachable
jiku status                # daemon, socket, model drivers, memory counts
jikud --headless-check     # validate configuration without a graphical session
```

- Koma compositor running, with `koma-agent.sock` under
  `$XDG_RUNTIME_DIR/koma/$KOMA_INSTANCE_SIGNATURE/`.
- `KOMA_CAPABILITY_TOKEN` exported (mint with `komactl token mint <label>`), or
  `KOMA_INSTANCE_SIGNATURE` + a token in the MCP host environment.
- Never export raw model keys. Configuration carries `api_key_env` **variable
  names**; the sandboxed Lua configuration VM cannot read the environment.

## Three Ways In

| Entry point                          | Use when                                                         |
| ------------------------------------ | ---------------------------------------------------------------- |
| `jiku run --workspace <ws> "<goal>"` | You want goal-level delegation and a structured report.          |
| `jiku mcp`                           | Your host is an MCP client and you want step-level control.      |
| `jiku <subcommand>`                  | You need one deterministic operation (status, a11y, TTS, bench). |

`jiku run` accepts `--dry-run` (plan only) and `--emulate` (force the structured
planner instead of the native decision model). Start with `--dry-run` on an
unfamiliar goal; it costs one planning call and no input events.

## The Core Loop

Every mutating action follows the same four steps. Skipping step 1 is the single
most common cause of wrong-window clicks.

1. **Observe.** `koma_scene_inventory`, then `koma_capture_frame` with
   `overlay: "grid_225"`. This returns a PNG and records a fresh epoch.
2. **Decide.** Name the target as a `grid_225` cell (`A1`..`O15`) or an exact
   `(x, y)`. Prefer cells: they survive resolution changes and are what the
   model actually sees.
3. **Act.** Pass `expected_epoch` from step 1 on every mutating call.
4. **Verify.** Re-capture and confirm the intended change happened. Do not
   assume success from a non-error response.

## Error Triage

| Signal                       | Meaning                                                                           | Response                                                               |
| ---------------------------- | --------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| `4090 ERR_STALE_EPOCH`       | Workspace changed since observation. The action was aborted **before** execution. | Nothing was applied. Re-capture, then retry once with the new epoch.   |
| `4091 ERR_HUMAN_PREEMPTED`   | The human took the workspace (`Mod+G`).                                           | Stop. Do not retry. Ask the user.                                      |
| `4032 ERR_PROTECTED_SURFACE` | Sensitive surface blackout.                                                       | Stop. Report what you were trying to do. Never try to route around it. |
| `4030 ERR_CAPABILITY_DENIED` | Token lacks the privilege level or the grant expired.                             | Ask the user for a new token; do not attempt to mint one.              |
| `RequireHumanVerification`   | Calibrated safety gate fired.                                                     | Halt and surface `security.verification_needed`.                       |

A stale epoch is not an error condition to work around — it is the fencing
mechanism doing its job. See `references/epoch-and-preemption.md`.

## Isolation Rules

- Create a `Headless` workspace (`koma_ws_create`) for agent work. Never drive a
  `Headed` workspace that holds the human's windows unless they asked you to.
- Capability tokens cannot hold `Control`/`Owner` over a physical monitor
  workspace in `HumanPreempted` mode; plan around this instead of fighting it.
- Text injection goes through `koma_inject_text` / `koma_paste_mime`, which
  bypass the IME. Do not synthesize individual keystrokes for free text.

## Where the Details Live

- `references/epoch-and-preemption.md` — epoch semantics, recovery procedure,
  dry-run and emulation.
- `references/mcp-setup.md` — `jiku mcp` and `koma-mcp` configuration for
  Claude Code, Codex, and OpenCode; which server to attach.
- `references/a11y-targeting.md` — AT-SPI `@snap:eN` handles, the five action
  verbs, and when accessibility targeting beats pixels.
- `references/skill-promotion.md` — how Jiku stages and promotes Lua/Markdown
  skills, and the review gate you must not bypass.

## Reporting

Return evidence, not assertions: the workspace id, the epoch you acted on, the
capture path or cell you targeted, and the verification frame. "Clicked the
button" is not a result; "clicked `G8` at epoch 12, settings panel now shows
Dark Mode enabled in the verification capture" is.
