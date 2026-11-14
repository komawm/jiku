# Changelog

All notable changes to Jiku are recorded here. Versions follow the `0.0.x`
incubation line before a formal `0.1.0` release.

## [Unreleased]

### Added

- Runnable `jikud` desktop orchestration daemon with `$XDG_RUNTIME_DIR/jiku.sock`
  (`0600`, `SO_PEERCRED` UID verification, 4-byte LE JSON framing),
  `--validate-config`, `--headless-check`, and clean `SIGINT`/`SIGTERM`
  shutdown.
- Runnable `jiku` CLI controller (`status`, `validate-config`, `decide`, `plan`,
  `run`, `box classify`, `memory search`, `skills list|diff|stage|approve`)
  and JSON-RPC 2.0 stdio MCP bridge (`jiku mcp --stateless | --print-tools`).
- Concrete Three-Driver HTTP/WebSocket clients:
  - `DecisionsClient` targeting `POST /v1beta/decisions` (`gemini-decision-1`
    with `$GEMINI_API_KEY`, `"input"` + `"answers"` wire contract) and automatic
    `EmulatedStructuredTextModel` fallback.
  - `TurnModelClient` with OpenCode Go provider defaults targeting
    `deepseek-v4.1-flash` (`$OPENCODE_GO_API_KEY`, `"thinking": {"type": "disabled"}`,
    and deterministic `x-opencode-session` UUID header), plus Gemini, Anthropic,
    and OpenAI-compatible adapters.
  - `LiveSessionConnection` WebSocket client (`BidiGenerateContent` setup,
    rate-limited screen/PCM frames, barge-in interruption flush, session
    resumption, and `goAway` handling).
- Native Koma Unix socket IPC bridge (`KomaIpcClient`) implementing 4-byte LE
  framing, `SCM_RIGHTS` sealed `memfd` reception (`mmap PROT_READ`), PNG
  encoding, `64-bit dHash`, and `Epoch`-fenced `vseat.*` dispatch with `4090`
  re-observation retry and `4091`/`4032` halts.
- Embedded [Phodopus](https://github.com/Xuepoo/phodopus) Lua 5.4 sandbox
  (`jiku.memory.on_prefetch`, `jiku.on`, `jiku.notify`, `jiku.tts.say`,
  `jiku.avatar.set_emotion`), SQLite WAL + FTS5 trajectory/skill store in
  `jiku-memory`, TTS/notifier providers in `jiku-tts`, `~/JikuBox` classifier
  in `jikud::jikubox`, and [`skills/SKILL.md`](skills/SKILL.md).

### Known limits (deferred to `v0.1.0`)

- Live PipeWire hardware microphone capture and speaker playout for interactive
  `jiku live` sessions.
- Production Linux Landlock / systemd sandbox hardening verified on host targets.
- Live2D / VRM companion rendering UI consuming `AvatarCue` streams.

## [0.0.1] - 2026-11-14

Source-only development prerelease of the library-only scaffold. There is no
`jiku` or `jikud` executable, functioning daemon, or network provider
integration in this release.

### Added

- Three-Driver model contracts in `jikud`: `DecisionsDriver`,
  `TurnModelDriver`, and optional `LiveSessionDriver` as pure-data types with
  unit-tested invariants.
- `DecisionExecutionMode` selecting `NativeDecisionsApi` when a dedicated
  decision model is configured and falling back to
  `EmulatedStructuredTextModel` (structured prompt plus JSON Schema
  constraints on the primary text/VLM model) otherwise.
- `DecisionCompiler` primitives compiling decision distributions into
  `TimeSlotMacro`, `BernoulliExpectedStep`, and `ThresholdProbe` steps, plus
  sub-cell grid centroid interpolation.
- Configurable defaults via `Default` implementations for `GridSpec`,
  `DecisionCompilerConfig`, and `LiveSessionConfig`, documented in
  `examples/config.toml`.
- Source release tooling: `scripts/package-source.sh`, `just package-source`,
  and a tag-triggered prerelease workflow.

### Known limits

- Library-only scaffold: no daemon, no CLI binary, no installable service.
- No network provider integration; provider settings in `examples/` are
  design inputs only.
- Memory and speech crates are documentation-only boundaries.

[Unreleased]: https://github.com/komawm/jiku/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/komawm/jiku/releases/tag/v0.0.1
