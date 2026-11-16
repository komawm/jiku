# Accessibility Targeting

Pixels are a lossy interface. AT-SPI gives you a tree of roles, names, states,
and actions — a button you can name instead of a rectangle you have to guess.

Reach for accessibility targeting when:

- The window has normal GTK/Qt/Chromium widgets (the common case).
- You need to set a text field's value rather than type into it.
- A pixel target keeps drifting because the layout animates.
- You already burned an epoch retry on the same coordinate twice.

Stay on pixels when the surface is a canvas, game, video, or anything that
publishes no accessible tree.

## Snapshot

```bash
jiku a11y snapshot
jiku a11y snapshot --json
jiku a11y snapshot --skeleton --app-id org.gnome.Settings --title "Settings"
jiku a11y snapshot --root @s3:e17
```

- Output is a normalized tree. `--skeleton` drops non-interactive containers and
  keeps a bounded set of interactive entries as
  `@snap:eN role "name" [actions]` — this is the form to feed a model, because it
  collapses a 2000-node tree into a few dozen actionable lines.
- `--app-id` and `--title` narrow the search to one window.
- `--root @s3:eN` re-roots at a subtree; the snapshot id stays the same, so
  existing handles remain valid.
- MCP equivalent: `jiku_a11y_snapshot` with `skeleton`, `app_id`,
  `window_title`, `root_ref`.

## Handles

A handle is `@<snapshot_id>:e<N>`, for example `@s3:e17`.

- It is **valid only against the snapshot that produced it**. The compositor
  keeps a bounded ring of recent snapshots; an evicted snapshot makes every
  handle from it stale.
- A stale handle returns `stale_ref`. There is no silent fallback to a
  similarly-positioned node.
- Handles are not stable across snapshots. Never cache one in a plan that spans
  a re-observation.

## Actions

```bash
jiku a11y act click     @s3:e17
jiku a11y act focus     @s3:e17
jiku a11y act set-value @s3:e17 "dark"
jiku a11y act type      @s3:e17 "dark mode"
jiku a11y act read-value @s3:e17
```

| Verb         | Effect                                                                                 |
| ------------ | -------------------------------------------------------------------------------------- |
| `click`      | Invokes the element's default action (`click`, `press`, or `activate`).                |
| `focus`      | Moves accessibility/input focus.                                                       |
| `set-value`  | Sets the value directly, without synthesizing keystrokes. Prefer this for text fields. |
| `type`       | Types through the input path when the toolkit ignores `set-value`.                     |
| `read-value` | Reads the current value back, useful for verification.                                 |

MCP equivalent: `jiku_a11y_act` with `action` and `target_ref`, plus `text` for
`set-value`/`type`.

The compositor only advertises verbs the element actually implements. If a verb
is missing from the snapshot's `[actions]` list, it will fail — do not retry it
with a different spelling.

## Choosing an Action

1. `set-value` if the element lists it. Deterministic, no focus dependency.
2. `click` if the element is a button/toggle.
3. `type` only for widgets that reject `set-value` (some Electron apps).
4. Fall back to `koma_click` on a `grid_225` cell when no accessible tree exists.

## Verification

Always confirm with `read-value` or a fresh `koma_capture_frame`. An action
returning `ok` means the AT-SPI call was delivered, not that the application
honoured it — modal dialogs and disabled states both accept and ignore.

## Security Notes

- Elements with `secret` or `sensitive` state are for password fields. Reading
  their value is refused; do not try to work around it.
- A window in a protected surface (`4032`) produces no usable tree. That is
  intentional.
- Accessibility data can contain user text. Do not copy it into logs, commit
  messages, or the final answer beyond what the task requires.
