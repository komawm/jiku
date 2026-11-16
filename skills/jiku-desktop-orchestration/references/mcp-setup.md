# MCP Setup

Two MCP servers are involved. They are not alternatives to the same thing —
they sit at different layers.

|           | `jiku mcp`                                        | `koma-mcp`                                |
| --------- | ------------------------------------------------- | ----------------------------------------- |
| Layer     | Orchestration, planning, memory, a11y, TTS        | Compositor input and capture              |
| Transport | `jiku.sock`, falling back to an in-process engine | `koma-agent.sock` with a capability token |
| Returns   | JSON only                                         | JSON plus `image/png` content blocks      |
| Fencing   | None at the MCP layer                             | Required `expected_epoch` on mutations    |
| Tools     | 12 `jiku_*`                                       | 17 `koma_*` plus runtime macro tools      |

Attach **one** unless you have a specific reason. A coding agent that only needs
eyes and hands should attach `koma-mcp`; an agent that needs goal-level
delegation should attach `jiku mcp`.

## jiku mcp

```bash
jiku mcp --print-tools     # dump the 12 tool schemas
jiku mcp --stateless       # one-shot pipe use
jiku mcp                   # long-lived stdio server
```

- Protocol `2024-11-05`, `Content-Length` and NDJSON framing auto-detected per
  message.
- `resources/list` and `prompts/list` are advertised and return empty arrays.
- Tools: `jiku_status`, `jiku_run_task`, `jiku_decide`, `jiku_plan`,
  `jiku_box_classify`, `jiku_memory_search`, `jiku_skills_list`,
  `jiku_a11y_snapshot`, `jiku_a11y_act`, `jiku_tts_say`,
  `jiku_context_inspect`, `jiku_bench_run`.
- Most tools take `compact` (default `true`) and return a reduced JSON payload.
  Keep it on unless you are debugging the full report.

## koma-mcp

```bash
koma-mcp --print-tools
koma-mcp --auto-mint-admin    # local development only — see below
```

Resolution order for the session:

1. `XDG_RUNTIME_DIR` must be set.
2. `KOMA_INSTANCE_SIGNATURE` selects `$XDG_RUNTIME_DIR/koma/<signature>/`.
3. `KOMA_CAPABILITY_TOKEN` authenticates the handshake.

A missing token is `401`-equivalent (`ERR_UNAUTHENTICATED`). `--auto-mint-admin`
mints an ephemeral admin-scoped token through `koma-admin.sock`; that is a
developer convenience for a human running the server by hand. Never enable it
for an agent host, because the agent would then hold the ability to mint its own
grants.

## Minting a Scoped Token

```bash
ls "$XDG_RUNTIME_DIR/koma"          # instance signature
komactl token mint coding-agent     # koma_cap_... (TTL 8h, 4 workspaces, headless allowed)
```

Tokens are ephemeral (256-bit, `koma_cap_<base64url>`). They expire; a long agent
session will eventually need a fresh one. Treat expiry like `4030`: ask the
human, do not retry.

## Host Configuration

Claude Code, Codex, OpenCode, Cursor, and other hosts all accept the same JSON
shape:

```json
{
    "mcpServers": {
        "koma": {
            "command": "koma-mcp",
            "args": [],
            "env": {
                "KOMA_INSTANCE_SIGNATURE": "<signature>",
                "KOMA_CAPABILITY_TOKEN": "koma_cap_..."
            }
        }
    }
}
```

For `jiku mcp` the same shape applies with `"command": "jiku", "args": ["mcp"]`
and no Koma token (the daemon owns that handshake).

## Practical Caveats

- `koma-mcp` connects lazily, so `initialize` and `tools/list` succeed even
  without a running compositor. A tool call is where a dead session surfaces.
- Screenshots dominate token cost. Pass `max_edge: 1024` on `koma_capture_frame`
  and keep the `grid_225` overlay; a labelled grid usually removes a retry.
- If the host supports tool allowlists, expose only `koma_ping`,
  `koma_scene_inventory`, `koma_ws_create`, `koma_capture_frame`, `koma_click`,
  `koma_key`, `koma_inject_text`. The kinematic path/drag parameters are noise
  for most coding tasks.
- Terminal-first agents often do better with `komactl` and a skill than with an
  MCP server: no schema cost, no image cost, and the shell tool is already
  present.
