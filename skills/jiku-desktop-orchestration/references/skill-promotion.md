# Skill Promotion

Jiku can emit new Lua or Markdown SOP skills while it works. Those candidates are
**staged**, never executed. Promotion is a review gate, and it exists because a
model-written automation script is untrusted code.

## Lifecycle

```text
task run -> candidate emitted -> staged_skills (SQLite) -> diff review -> approve
                                                            \-> reject
```

State lives in `~/.local/state/jiku/state.db` (SQLite, WAL). Approved skills are
copied into the user configuration directory (`~/.config/jiku/skills` by
default).

## Commands

```bash
jiku skills list
jiku skills diff <skill_id>
jiku skills stage <path>
jiku skills approve <skill_id> [--promote-dir ~/.config/jiku/skills]
```

- `list` shows staged, approved, and rejected candidates with their origin task.
- `diff` is the review step. Read it. Do not approve a skill you have not read.
- `approve` promotes the staged content; it does not re-validate semantics, only
  the sanitization rules below.

## Sanitization Rules

Staged content is rejected when it contains any of:

```text
io.   os.   debug.   package.   loadfile   dofile   load(   require(
```

This blocks filesystem access, process spawning, debug-library reflection,
module loading, and dynamic code loading. It is a denylist, so it is necessary
but not sufficient — a skill that only uses `koma.*` and `jiku.*` is still worth
reading before approval.

## Approval Discipline

1. Run `diff` and read the whole candidate.
2. Confirm every host call is a `koma.*` or `jiku.*` operation you understand.
3. Confirm the skill declares which workspace it expects and how it fails.
4. Approve only after the diff is clean in this session — not because the task
   reported success.

Approving a skill is a durable, user-level action. It survives the session,
applies to future tasks, and can drive real input events. Treat it like merging
code into a config that runs on login.

## When Not to Promote

- The skill hardcodes workspace ids, window titles, or coordinates.
- The skill retries mutations without re-observing an epoch.
- The skill handles `4091`/`4032` by trying another path instead of halting.
- The body reads like a transcription of one successful run rather than a
  reusable procedure.

In those cases the right output is a report to the user, not a staged skill.
