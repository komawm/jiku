# Changelog

All notable changes to Jiku are recorded here. Versions follow the `0.0.x`
incubation line before a formal `0.1.0` release.

## [Unreleased]

### Planned for `v0.1.0`

- Live PipeWire full-duplex hardware microphone I/O for interactive
  `jiku live` voice sessions (TTS playout via `AudioPlaybackSink` is included
  in `v0.0.1`).
- Production Linux `Landlock` + `seccomp` / systemd sandbox enforcement
  verified on host targets.
- Native Wayland layer-shell companion renderer (`MMD` / `Inochi2D` overlays)
  complementing the browser-hosted `jiku avatar serve` WebSocket VRM stage.

## [0.0.1] - 2026-11-16

### Added (trajectories, grounding, small-context compaction, TTS, avatar & MCP)

- Multi-waypoint cursor trajectories (`vseat.path` / `vseat.stroke`),
  `vseat.cursor_settle`, configurable `hover_glide_ms` cursor glides, and
  automatic workspace lease release after `run_task`.
- Situational AT-SPI2 UI accessibility tree capture and semantic action
  dispatch (`jiku a11y snapshot` / `jiku a11y act` with `@snap:eN` refs) and
  `SystemGroundingPack` context enrichment.
- 4-stage `context_budget` compaction waterfall (`stage1_visual_roi_pruning`,
  `stage2_a11y_skeleton_folding`, `stage3_episodic_l1_folding`, plus output
  token reservation clamping via `jiku context inspect`) and tolerant balanced
  JSON slice extraction for 2B VLMs in `decisions.rs`.
- Reproducible `2x2` `VirtualPaint` decision vs. grounding ablation harness
  (`jiku bench run --offline` and `--live`).
- Modular speech synthesis and 20 ms RMS lip-sync cue extractor (`jiku-tts`:
  `espeak_ng`, `irodori`, `voicevox`, `openai_speech`, and `noop` providers,
  barge-in `AudioPlaybackSink`, `jiku tts <say|stop>`).
- Companion avatar stage server (`jiku avatar serve [--bind <ADDR>] [--vrm <PATH>]`)
  serving HTTP + WebSocket `Snapshot` / `AvatarCue` streams with `vseat`
  idle gaze tracking (`idle_gaze_follow_vseat`).
- Dual-framing (`Content-Length` + NDJSON) `jiku mcp` stdio server exposing 12
  tools (`jiku_status`, `jiku_run_task`, `jiku_decide`, `jiku_plan`,
  `jiku_box_classify`, `jiku_memory_search`, `jiku_skills_list`,
  `jiku_a11y_snapshot`, `jiku_a11y_act`, `jiku_tts_say`,
  `jiku_context_inspect`, `jiku_bench_run`).

Initial runnable `0.0.x` development prerelease shipping multi-architecture
(`x86_64-unknown-linux-gnu` and 16 KiB page-aligned `aarch64-unknown-linux-gnu`)
binary archives (`jikud` and `jiku`) alongside the source archive. Not an
accepted `v0.1.0` product release.

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
- Three-Driver decision contracts (`DecisionsDriver`, `TurnModelDriver`,
  `LiveSessionDriver`), `DecisionCompiler`, and configurable defaults in
  `examples/config.toml`.
- Multi-architecture release tooling (`scripts/package-release.sh`,
  `just package-release`, `scripts/package-source.sh`) producing
  `jiku-<version>-x86_64-unknown-linux-gnu.tar.gz`,
  `jiku-<version>-aarch64-unknown-linux-gnu.tar.gz`, `jiku-<version>-src.tar.gz`,
  and `SHA256SUMS`.

### Known limits (deferred to `v0.1.0`)

- Live PipeWire full-duplex hardware microphone I/O for interactive
  `jiku live` sessions.
- Production Linux `Landlock` + `seccomp` / systemd sandbox enforcement
  verified on host targets.
- Native Wayland layer-shell companion renderer (`MMD` / `Inochi2D` overlays)
  complementing the WebSocket `jiku avatar serve` stage.

[Unreleased]: https://github.com/komawm/jiku/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/komawm/jiku/releases/tag/v0.0.1
