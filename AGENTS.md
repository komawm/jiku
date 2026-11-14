# Jiku Agent Guidance

Follow [Koma workspace governance](https://github.com/komawm/.github/blob/main/AGENTS.md) and the canonical specifications in [komawm/docs](https://github.com/komawm/docs/blob/main/docs/JIKU_ARCHITECTURE.md). Keep one root GPL-3.0-or-later [LICENSE](LICENSE) and use canonical HTTPS links to `https://github.com/komawm/...`.

## Repository Topology

- `crates/jikud`: `jikud` orchestration daemon binary + library (`config`, `daemon`, `drivers/{decisions,turn,live}`, `koma_ipc`, `lua_policy`, `jikubox`).
- `crates/jiku`: `jiku` CLI controller & JSON-RPC 2.0 MCP server (`client`, `mcp`).
- `crates/jiku-memory`: L0 perceptual `dHash` frame ring, L1 app quirks, L2 SQLite WAL + FTS5 episode search, and sanitized `staged_skills` management.
- `crates/jiku-tts`: `DesktopNotifier`, `NoopTtsProvider`, `VoicevoxProvider`, `OpenAiSpeechProvider`, and `AvatarCue` lip-sync envelope extractors.
- `skills/`: Bundled agent SOPs ([SKILL.md](skills/SKILL.md)). No separate `jiku-skills` repository exists.

## Development & Quality Gates

- Toolchain: Rust `1.100.0`, Edition `2024` ([`rust-toolchain.toml`](rust-toolchain.toml)).
- Always run gates cleanly without requiring API keys:

```bash
just fmt
env -u GEMINI_API_KEY -u OPENCODE_GO_API_KEY just check
```

- `just check` runs `scripts/check-infrastructure.py`, `actionlint`, `shellcheck scripts/*.sh`, `cargo fmt --all -- --check`, `cargo check --workspace --all-targets --locked`, `cargo clippy --workspace --all-targets --locked -- -D warnings`, and `cargo test --workspace --all-targets --locked`.
- Keep `.github/workflows/ci.yml` self-contained without host `apt-get` steps; crates must remain pure-Rust / bundled (`rusqlite` uses `features = ["bundled"]`).

## Coding & Safety Invariants

- Never log, echo, write to disk, or commit `$GEMINI_API_KEY`, `$OPENCODE_GO_API_KEY`, or Koma capability tokens.
- Keep all Koma IPC mutations fenced by observed `Epoch` (`4090` triggers re-capture; `4091` `Mod+G` preemption and `4032` protected surfaces halt immediately).
- Do not activate `resources/jikud.service` on the host desktop during automated tests.
- Real-time PipeWire audio streaming, production Landlock service sandboxing, and Live2D/VRM UI rendering remain targeted for `v0.1.0`; do not claim them as verified host hardware behavior until implemented and tested.
