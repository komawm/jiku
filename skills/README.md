# Bundled Jiku Agent Skill

This directory ships the installable agent skill for Jiku. It is not a separate
Git repository; the skill lives with the code it teaches.

## Install

Requires Node.js 22.20 or later.

```bash
# Install for every detected agent (global)
npx skills add komawm/jiku --skill jiku-desktop-orchestration -g

# Install for one agent, in the current project
npx skills add komawm/jiku --skill jiku-desktop-orchestration -a claude-code

# List what this repository publishes
npx skills add komawm/jiku --list
```

Once `@koma-wm/jiku` is published on npm, an installed package also exposes the
skill through dependency sync:

```bash
npm install -D skills @koma-wm/jiku
npx skills experimental_sync -y
```

## Layout

```text
skills/
└── jiku-desktop-orchestration/
    ├── SKILL.md          # thin router: when to use, core loop, error triage
    ├── LICENSE           # MIT (the repository root LICENSE stays GPL)
    └── references/
        ├── epoch-and-preemption.md
        ├── mcp-setup.md
        ├── a11y-targeting.md
        └── skill-promotion.md
```

The catalog layout (`skills/<name>/SKILL.md`) is what the [Skills CLI](https://github.com/vercel-labs/skills)
discovers. `name` in the frontmatter must equal the directory name.

## Licensing

The skill ships under **MIT** so it can be copied into any agent workspace. The
per-skill [`LICENSE`](jiku-desktop-orchestration/LICENSE) travels with the
directory because the installer copies only that folder — the repository root
[`LICENSE`](../LICENSE) is `GPL-3.0-or-later` and does not apply to a detached
skill copy.

## Managing Dynamic Skills

Runtime-synthesized candidates are separate from this bundled skill. They are
stored in SQLite (`~/.local/state/jiku/state.db`) and require review before
promotion:

```bash
jiku skills list
jiku skills diff <skill_id>
jiku skills approve <skill_id> [--promote-dir ~/.config/jiku/skills]
```

See [`references/skill-promotion.md`](jiku-desktop-orchestration/references/skill-promotion.md)
for the review gate and sanitization rules.
