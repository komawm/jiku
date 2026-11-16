# Epoch and Preemption

Koma fences every workspace mutation with a monotonically increasing `Epoch`.
The number identifies _the exact state you observed_. If the workspace changed,
your number is stale and the action is refused.

## Why It Exists

An agent is blind between two tool calls. A human may switch windows, a
notification may pop, an application may resize. Without fencing, a cached
coordinate silently lands on the wrong widget. The epoch turns that silent
mistake into a deterministic refusal.

## Required Sequence

```text
scene.inventory            -> workspace ids and current leases
capture.frame(ws, grid)    -> PNG + observed_epoch
mutate(ws, expected_epoch) -> applied, or refused
capture.frame(ws, grid)    -> verify
```

- `expected_epoch` is a required argument on mutating vSeat calls in `koma-mcp`.
- Omitting it is not "best effort": the compositor either rejects the request or
  applies it under the lease you already hold, which is exactly the failure mode
  the mechanism prevents. Always pass it.

## Error Semantics

| Code   | Symbol                  | Applied?                          | Retry?                                                    |
| ------ | ----------------------- | --------------------------------- | --------------------------------------------------------- |
| `4090` | `ERR_STALE_EPOCH`       | **No** — aborted before execution | Yes, once, with the epoch returned in the new observation |
| `4091` | `ERR_HUMAN_PREEMPTED`   | No                                | No. The human owns the workspace.                         |
| `4032` | `ERR_PROTECTED_SURFACE` | No                                | No. Report and stop.                                      |
| `4030` | `ERR_CAPABILITY_DENIED` | No                                | Only after the user grants a wider scope                  |

`4090` responses from `koma-mcp` carry an **already-fresh** PNG of the new state
and a `new_observed_epoch`. Do not issue another capture before inspecting it —
the image in hand is current.

## Recovery Procedure for `4090`

1. Read the attached frame. Do not discard it.
2. Recompute the target. The window you meant to click may no longer be there.
3. Retry with `expected_epoch = new_observed_epoch`.
4. If the retry also returns `4090`, stop and re-plan the goal. Two consecutive
   stale epochs mean the workspace is actively changing and pixel targeting is
   the wrong strategy — switch to accessibility targeting.

## Preemption

`Mod+G` lets the human seize the workspace. From that moment:

- Any agent mutation returns `4091`.
- Any agent token is denied `Control`/`Owner` on a physical monitor workspace.
- Your job is to halt and hand back. Report the workspace id and what you were
  doing. Do not wait in a retry loop.

Preemption is a safety feature with a UI affordance, not a bug to route around.

## Dry Runs and Emulation

```bash
jiku run --workspace agent:work --dry-run "toggle dark mode in settings"
jiku run --workspace agent:work --emulate "toggle dark mode in settings"
```

- `--dry-run` produces the plan and the decision routing without dispatching
  input events. Use it to validate that the goal is understood before spending
  screenshots.
- `--emulate` forces the structured-text planner instead of the native decision
  driver. Use it when the configured decision model is unavailable or when you
  need deterministic, comparable output.

Neither flag makes a run safe to point at the human's `Headed` workspace — it
still emits real input for anything that is not a dry run.
