# Jiku

Desktop orchestration daemon (`jikud`), CLI/MCP controller (`jiku`), three-tier SQLite WAL+FTS5 memory (`jiku-memory`), and companion TTS/notification hooks (`jiku-tts`) built to pair with the [Koma](https://github.com/komawm/koma) Wayland compositor.

Canonical ecosystem documentation lives in [komawm/docs](https://github.com/komawm/docs), specifically [JIKU_ARCHITECTURE.md](https://github.com/komawm/docs/blob/main/docs/JIKU_ARCHITECTURE.md).

## Workspace Crates

| Crate                                      | Purpose                                                                                                                                                                                                                                                                               |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`crates/jikud`](crates/jikud)             | Daemon (`jikud`), TOML config, Three-Driver clients (`DecisionsDriver`, `TurnModelDriver`, `LiveSessionDriver`), `KomaIpcClient` (`SCM_RIGHTS` sealed `memfd` + `Epoch` fencing), embedded [Phodopus](https://github.com/Xuepoo/phodopus) Lua 5.4 sandbox, and `~/JikuBox` classifier |
| [`crates/jiku`](crates/jiku)               | Controller binary (`jiku`), `jiku.sock` RPC / in-process fallback client, and JSON-RPC 2.0 stdio MCP server (`jiku mcp`)                                                                                                                                                              |
| [`crates/jiku-memory`](crates/jiku-memory) | L0 perceptual `dHash` frame ring, L1 per-`app_id` quirk notes, L2 SQLite WAL + FTS5 BM25 episode index, and sanitized `staged_skills` diff/approval store                                                                                                                             |
| [`crates/jiku-tts`](crates/jiku-tts)       | D-Bus/`notify-send` alerts (`DesktopNotifier`), `NoopTtsProvider`, VOICEVOX mora-viseme extractor, OpenAI-compatible `/audio/speech`, and 24 kHz PCM `AvatarCue` envelope extraction                                                                                                  |

## Model Configuration

See [`examples/config.toml`](examples/config.toml), [`examples/config.lua`](examples/config.lua), and [`examples/init.lua`](examples/init.lua). All three are generic templates: provider endpoints, model IDs, and the `api_key_env` variable names are placeholders you replace with your own bindings. No vendor or model is hard-coded.

Configuration is accepted as TOML or Lua, selected by file extension (`--config config.toml` or `--config config.lua`). In the default search `config.lua` is merged field by field over `config.toml`, so a Lua overlay only states the keys it changes. The Lua VM shares the `init.lua` sandbox profile (8 MiB heap, deterministic fuel, no `io`/`os`/`dofile`/`loadfile`), so it cannot read the environment: `api_key_env` stays a variable _name_ and secrets never enter the Lua heap.

- **Decision Router & JikuBox Triage**: an optional dedicated decision model reached through a native decisions endpoint (`POST /v1beta/decisions`) with zero output tokens, falling back seamlessly to `EmulatedStructuredTextModel` on the primary planner when the role is unset or `--emulate` is requested.
- **Planner (`[roles.planner]`)**: any text/VLM provider you configure (OpenAI-compatible, Anthropic, Google AI Studio, or a local `vLLM` / `llama.cpp` / `Ollama` endpoint), with optional `thinking = false` (`"thinking": {"type": "disabled"}`), deterministic session headers, and the API key read from the variable named by `api_key_env`.

## Quick Start

```bash
just check
cargo build --workspace --release --locked

# Offline validation & daemon self-check (no API keys required)
./target/release/jikud --headless-check
./target/release/jikud --validate-config --config examples/config.toml --policy examples/init.lua
./target/release/jikud --validate-config --config examples/config.lua --policy examples/init.lua

# Run daemon ($XDG_RUNTIME_DIR/jiku.sock) or invoke CLI directly
./target/release/jikud --daemon --config examples/config.toml --policy examples/init.lua
./target/release/jiku --config examples/config.toml status --json
./target/release/jiku --config examples/config.toml decide --image shot.png "Is a terminal visible?"
./target/release/jiku --config examples/config.toml plan "Summarize the active workspace"
./target/release/jiku --config examples/config.toml run --workspace ws_01 "Open a file and verify output"
./target/release/jiku --config examples/config.toml box classify ~/JikuBox
./target/release/jiku --config examples/config.toml mcp --print-tools
```

## Pre-`v0.1.0` Incubation Scope

The following items remain deferred to `v0.1.0` and are not yet active runtime features:

- Live PipeWire hardware microphone capture (`16 kHz` uplink) and `24 kHz` speaker playback for interactive `jiku live` voice sessions (protocol WebSocket transport and `BidiGenerateContent` framing are implemented in `drivers::live`).
- Production Linux Landlock / Bubblewrap hardening tuned per distribution beyond unit-tested Phodopus Lua memory/fuel isolation (`resources/jikud.service` is provided as a systemd user unit template).
- Live2D Cubism / VRM companion overlay rendering UI consuming `AvatarCue` streams.

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
