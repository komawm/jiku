# Jiku

Jiku is the proposed out-of-process desktop GUI orchestration companion for [Koma](https://github.com/komawm/koma). This independent repository establishes its ownership boundary and preserves the existing configuration and service templates. It contains four library-only scaffolding crates (`jikud`, `jiku`, `jiku-memory`, `jiku-tts`), all non-runnable and unpublished. `crates/jikud` and `crates/jiku` define pure-data contract types and unit-tested invariants for:

- **Three-Driver Model Architecture**: `DecisionsDriver` (`POST /v1beta/decisions` for `gemini-decision-1`), `TurnModelDriver` (`HTTP/2 SSE` for primary text/VLM models), and optional `LiveSessionDriver` (`BidiGenerateContent` full-duplex WebSocket).
- **Primary Text/VLM Model + Zero-Error Decision Fallback (`DecisionExecutionMode`)**: Uses the primary text/VLM model (`[roles.planner]`) as the foundational model, accelerates closed-set routing via `NativeDecisionsApi` (`0` output tokens) when `gemini-decision-1` is configured, and automatically falls back to `EmulatedStructuredTextModel` (`synthesize_emulated_decision_prompt` + JSON Schema constraints) without error when no dedicated decision model is configured.
- **`DecisionCompiler` & Configurable Defaults**: Compiles single-request decision distributions into multi-step macros (`TimeSlotMacro`, `BernoulliExpectedStep`, `ThresholdProbe`) and sub-cell `GridSpec` centroid coordinates (`interpolate_grid_225_centroid`), with all grid dimensions, confidence thresholds, fallback sentinels, and `LiveSessionConfig` sample rates/FPS backed by `Default` implementations configurable via `examples/config.toml`.
- **Standalone & Sub-Agent Surface (`crates/jiku`)**: Supports standalone interactive/CLI operation as well as stateless MCP (`jiku mcp --stateless`) and CLI sub-agent invocation from external AI coding agents (`Claude Code`, `Codex`, `GitHub Copilot`, `Cursor`, `OpenCode`), while remaining library-only non-runnable scaffolds prior to `v0.1.0` — there is no `jiku`/`jikud` executable, network provider integration, or functioning daemon yet.

- [Canonical architecture](https://github.com/komawm/docs/blob/main/docs/JIKU_ARCHITECTURE.md) and [configuration proposal](https://github.com/komawm/docs/blob/main/docs/JIKU_CONFIG_SPEC.md) live in koma-docs.
- [Crate scaffolding](https://github.com/komawm/jiku/tree/main/crates) reserves the daemon, controller, memory and speech boundaries.
- [Examples](https://github.com/komawm/jiku/tree/main/examples) contain the imported provider and Lua-policy templates.
- [Resources](https://github.com/komawm/jiku/tree/main/resources) contain the proposed `jikud.service`; it cannot run without an implementation and validated installation.
- [Skills](https://github.com/komawm/jiku/tree/main/skills) is an internal Jiku directory. There is no separate jiku-skills repository or implemented skill runtime.

Provider names, routes, memory/voice settings, sandbox restrictions and CLI commands in the templates are design inputs. Keep real credentials out of examples and do not treat a service file as security or runtime acceptance. Avatar integration is deferred and does not have a repository.

The incubation baseline is 0.0.1 on the 0.0.x line before a first formal 0.1.0 release line. The Rust scaffolding targets Rust 1.100.0 and Rust 2024; its local checks and pure-data contract unit tests pass under that target compiler, with no runtime daemon tests. A scaffold build does not establish an agent runtime. All Jiku crates use [GPL-3.0-or-later](https://github.com/komawm/jiku/blob/main/LICENSE). Preserve third-party attribution when implementation is added.

Read [AGENTS.md](https://github.com/komawm/jiku/blob/main/AGENTS.md), use the repository justfile for its actual documentation/template gates, and track work in its local CarryCtx project. Local initialization does not publish or release the component.
