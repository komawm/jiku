# Bundled Jiku Agent Skills

This directory contains the bundled skills and safety SOPs shipped alongside `jikud` and the `jiku` controller CLI/MCP bridge (there is no separate `jiku-skills` repository).

- [SKILL.md](SKILL.md): Canonical desktop orchestration rules (`Epoch` fencing, protected-surface blackout handling, `RequireHumanVerification` halting, JikuBox triage, and staged-skill approval).

## Managing Dynamic Skills

Dynamically synthesized skill candidates are stored in SQLite (`~/.local/state/jiku/state.db`) and validated before promotion into user configuration:

```bash
jiku skills list
jiku skills diff <skill_id>
jiku skills approve <skill_id> [--promote-dir ~/.config/jiku/skills]
```
