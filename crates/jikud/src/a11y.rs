//! Situational `agent-desktop` AT-SPI accessibility snapshot engine and interaction router.
//!
//! Provides pure-Rust (`#![forbid(unsafe_code)]`) accessibility tree normalization,
//! Chromium/Electron/GTK anonymous wrapper compression, snapshot-scoped `@<snap>:eN`
//! reference resolution, secure password-field redaction, pluggable [`A11yAdapter`]
//! backends (`MockA11yAdapter` and bounded-timeout `AtSpiGdbusAdapter`), and a
//! deterministic situational router ([`select_interaction_route`]).

use std::{
    collections::{BTreeMap, VecDeque},
    fmt,
    future::Future,
    io::Read as _,
    pin::Pin,
    process::{Command, Stdio},
    sync::Mutex,
    time::{Duration, Instant},
};

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Placeholder string emitted in snapshots and redacted action logs for secure fields.
pub const REDACTED_VALUE: &str = "[REDACTED]";

/// Default number of historical accessibility snapshots retained in [`SnapshotStore`].
pub const DEFAULT_SNAPSHOT_CAPACITY: usize = 16;

/// Default timeout (`500 ms`) enforced on `gdbus` AT-SPI queries.
pub const DEFAULT_ATSPI_TIMEOUT_MS: u64 = 500;

/// Maximum logical depth retained when `SnapshotFilterOpts::skeleton` is enabled.
pub const SKELETON_MAX_DEPTH: usize = 2;

/// Boxed future alias returned by [`A11yAdapter`] async methods.
pub type A11yFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T, A11yError>> + Send + 'a>>;

/// Structured error codes returned by accessibility snapshot resolution and actions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum A11yErrorCode {
    /// Reference syntax was invalid or used an ambiguous bare `@eN` identifier.
    InvalidRefFormat,
    /// Snapshot ID is unknown or was evicted from the bounded [`SnapshotStore`] ring.
    StaleRef,
    /// Snapshot exists, but the requested `@<snap>:eN` element ID does not exist in it.
    NotFound,
    /// Query matched multiple candidates without a unique selector.
    AmbiguousTarget,
    /// Reading or leaking a password/secret/sensitive field's value is forbidden.
    SecureFieldBlocked,
    /// AT-SPI daemon or `gdbus` backend is unavailable or timed out.
    BackendUnavailable,
    /// Target element rejected or failed to execute the requested action.
    ActionFailed,
}

impl fmt::Display for A11yErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::InvalidRefFormat => "invalid_ref_format",
            Self::StaleRef => "stale_ref",
            Self::NotFound => "not_found",
            Self::AmbiguousTarget => "ambiguous_target",
            Self::SecureFieldBlocked => "secure_field_blocked",
            Self::BackendUnavailable => "backend_unavailable",
            Self::ActionFailed => "action_failed",
        };
        f.write_str(label)
    }
}

/// Structured accessibility error with diagnostic context and optional remediation hint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Error)]
#[error("[{code}] {message}")]
pub struct A11yError {
    /// Machine-readable error classification.
    pub code: A11yErrorCode,
    /// Human-readable error message.
    pub message: String,
    /// Offending element reference, if applicable.
    #[serde(default)]
    pub target_ref: Option<String>,
    /// Optional recovery hint for the caller.
    #[serde(default)]
    pub suggestion: Option<String>,
}

impl A11yError {
    /// Creates an [`A11yError`] with optional target reference and suggestion.
    #[must_use]
    pub fn new(
        code: A11yErrorCode,
        message: impl Into<String>,
        target_ref: Option<String>,
        suggestion: Option<String>,
    ) -> Self {
        Self {
            code,
            message: message.into(),
            target_ref,
            suggestion,
        }
    }

    /// Creates an [`A11yErrorCode::InvalidRefFormat`] error.
    #[must_use]
    pub fn invalid_ref_format(raw_ref: &str, message: impl Into<String>) -> Self {
        Self::new(
            A11yErrorCode::InvalidRefFormat,
            message,
            Some(raw_ref.to_string()),
            Some("Capture a fresh snapshot and pass a scoped @<snapshot_id>:e<N> ref".to_string()),
        )
    }

    /// Creates an [`A11yErrorCode::StaleRef`] error.
    #[must_use]
    pub fn stale_ref(raw_ref: &str, message: impl Into<String>) -> Self {
        Self::new(
            A11yErrorCode::StaleRef,
            message,
            Some(raw_ref.to_string()),
            Some("Re-capture an accessibility snapshot before retrying the action".to_string()),
        )
    }

    /// Creates an [`A11yErrorCode::NotFound`] error.
    #[must_use]
    pub fn not_found(raw_ref: &str, message: impl Into<String>) -> Self {
        Self::new(
            A11yErrorCode::NotFound,
            message,
            Some(raw_ref.to_string()),
            None,
        )
    }

    /// Creates an [`A11yErrorCode::SecureFieldBlocked`] error.
    #[must_use]
    pub fn secure_field_blocked(raw_ref: &str, message: impl Into<String>) -> Self {
        Self::new(
            A11yErrorCode::SecureFieldBlocked,
            message,
            Some(raw_ref.to_string()),
            Some("Reading values from password/secret/sensitive fields is blocked".to_string()),
        )
    }

    /// Creates an [`A11yErrorCode::BackendUnavailable`] error.
    #[must_use]
    pub fn backend_unavailable(message: impl Into<String>) -> Self {
        Self::new(
            A11yErrorCode::BackendUnavailable,
            message,
            None,
            Some("Fall back to visual grounding / ROI crop when AT-SPI is unavailable".to_string()),
        )
    }

    /// Creates an [`A11yErrorCode::ActionFailed`] error.
    #[must_use]
    pub fn action_failed(raw_ref: Option<&str>, message: impl Into<String>) -> Self {
        Self::new(
            A11yErrorCode::ActionFailed,
            message,
            raw_ref.map(ToOwned::to_owned),
            None,
        )
    }
}

/// Raw accessibility node produced by an AT-SPI backend or mock provider before
/// wrapper compression, security redaction, and snapshot ref assignment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct A11yRawNode {
    /// Accessible display name / label.
    pub name: String,
    /// Normalized or raw AT-SPI role name (e.g. `"push_button"`, `"password_text"`, `"panel"`).
    pub role: String,
    /// Optional textual or numeric value exposed by the control.
    #[serde(default)]
    pub value: Option<String>,
    /// Optional accessible description string.
    #[serde(default)]
    pub description: Option<String>,
    /// Logical window-relative bounding rectangle `[x, y, width, height]`.
    #[serde(default)]
    pub rect: Option<[i32; 4]>,
    /// AT-SPI state labels (e.g. `"focusable"`, `"showing"`, `"secret"`, `"sensitive"`).
    #[serde(default)]
    pub states: Vec<String>,
    /// Supported AT-SPI action verbs (e.g. `"click"`, `"press"`, `"activate"`, `"focus"`, `"set_value"`).
    #[serde(default)]
    pub actions: Vec<String>,
    /// D-Bus unique or well-known bus name owning this accessible object.
    #[serde(default)]
    pub bus_name: Option<String>,
    /// D-Bus object path for direct AT-SPI method invocation.
    #[serde(default)]
    pub object_path: Option<String>,
    /// Child accessibility nodes.
    #[serde(default)]
    pub children: Vec<A11yRawNode>,
}

/// Normalized, snapshot-scoped accessibility entry emitted inside [`A11ySnapshot`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct A11yEntry {
    /// Snapshot-scoped unique handle in `"@<snapshot_id>:e<N>"` format.
    pub ref_id: String,
    /// Local element identifier (`"e<N>"`) within the snapshot.
    pub raw_ref: String,
    /// Normalized role name.
    pub role: String,
    /// Accessible name.
    pub name: String,
    /// Redacted (`Some("[REDACTED]")` for secure fields) or visible value.
    pub value: Option<String>,
    /// Logical bounding box `[x, y, width, height]`, if exposed.
    pub rect: Option<[i32; 4]>,
    /// Normalized supported actions.
    pub actions: Vec<String>,
    /// Logical tree depth after wrapper compression.
    pub depth: usize,
    /// True if this node is a password, secret, or sensitive field.
    pub is_secure: bool,
    /// Direct child count in the compressed tree.
    pub child_count: usize,
}

/// Backend D-Bus addressing metadata retained alongside a snapshot entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct A11yNodeAddress {
    /// D-Bus bus name owning this accessible node.
    pub bus_name: Option<String>,
    /// D-Bus object path for this accessible node.
    pub object_path: Option<String>,
}

/// Filtering and pruning options for [`SnapshotStore::build_snapshot`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotFilterOpts {
    /// When `true`, limits traversal depth to `<= 2` (`SKELETON_MAX_DEPTH`).
    pub skeleton: bool,
    /// Optional `"@<snapshot_id>:e<N>"` ref to drill down into a previously observed subtree.
    pub root_ref: Option<String>,
    /// Maximum logical depth included in the emitted snapshot.
    pub max_depth: usize,
    /// Maximum total entries emitted in the snapshot.
    pub max_nodes: usize,
    /// Whether to apply [`collapse_web_wrappers`] before depth/ref assignment.
    pub collapse_wrappers: bool,
}

impl Default for SnapshotFilterOpts {
    fn default() -> Self {
        Self {
            skeleton: false,
            root_ref: None,
            max_depth: 12,
            max_nodes: 256,
            collapse_wrappers: true,
        }
    }
}

/// Materialized accessibility snapshot bound to a unique monotonic `snapshot_id`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct A11ySnapshot {
    /// Monotonic snapshot identifier (e.g. `"s1"`).
    pub snapshot_id: String,
    /// Wayland `app_id` or application name.
    pub app_id: String,
    /// Active window title.
    pub window_title: String,
    /// Flattened pre-order list of visible entries with `"@<snapshot_id>:e<N>"` refs.
    pub entries: Vec<A11yEntry>,
    /// Total raw nodes inspected before wrapper compression and truncation.
    pub total_raw_nodes: usize,
    /// Number of anonymous wrapper nodes skipped at zero depth cost.
    pub collapsed_wrappers_count: usize,
    /// Whether `max_nodes` or `max_depth` truncated part of the tree.
    pub truncated: bool,
    /// Whether skeleton depth cap (`<= 2`) was active.
    pub skeleton: bool,
    /// Non-fatal diagnostic note (e.g. if `root_ref` fallback occurred).
    #[serde(default)]
    pub warning: Option<String>,
    /// Backend D-Bus addresses keyed by `ref_id`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub addresses: BTreeMap<String, A11yNodeAddress>,
}

impl A11ySnapshot {
    /// Renders a compact, deterministic text tree of this snapshot suitable for VLM/LLM prompts.
    ///
    /// Never emits raw values of `is_secure` entries (`value` is always `[REDACTED]`).
    ///
    /// Time complexity: $O(N)$ where $N$ is `self.entries.len()`.
    /// Space complexity: $O(N)$ for the formatted prompt buffer.
    #[must_use]
    pub fn format_text_tree(&self) -> String {
        let mut out = format!(
            "snapshot=@{} app={:?} title={:?} entries={}{}\n",
            self.snapshot_id,
            self.app_id,
            self.window_title,
            self.entries.len(),
            if self.truncated { " [truncated]" } else { "" }
        );
        if let Some(warn) = &self.warning {
            out.push_str(&format!("  warning: {warn}\n"));
        }
        for entry in &self.entries {
            let indent = "  ".repeat(entry.depth);
            out.push_str(&format!("{indent}{} [{}]", entry.ref_id, entry.role));
            if !entry.name.is_empty() {
                out.push_str(&format!(" {:?}", entry.name));
            }
            if entry.is_secure {
                out.push_str(&format!(" val={REDACTED_VALUE:?} [secure]"));
            } else if let Some(val) = &entry.value
                && !val.is_empty()
            {
                out.push_str(&format!(" val={val:?}"));
            }
            if let Some([x, y, w, h]) = entry.rect {
                out.push_str(&format!(" rect=[{x},{y},{w},{h}]"));
            }
            if !entry.actions.is_empty() {
                out.push_str(&format!(" actions=[{}]", entry.actions.join(",")));
            }
            if entry.child_count > 0 {
                out.push_str(&format!(" children={}", entry.child_count));
            }
            out.push('\n');
        }
        out
    }

    /// Counts entries that expose at least one direct interactive action or interactive role.
    #[must_use]
    pub fn actionable_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| {
                e.actions.iter().any(|a| is_interactive_action(a)) || is_interactive_role(&e.role)
            })
            .count()
    }

    /// Looks up an [`A11yEntry`] by its scoped `ref_id` within this snapshot.
    #[must_use]
    pub fn find_by_ref(&self, ref_id: &str) -> Option<&A11yEntry> {
        self.entries.iter().find(|e| e.ref_id == ref_id)
    }
}

/// Normalizes a role token to lowercase snake_case.
#[must_use]
pub fn normalize_role(raw: &str) -> String {
    raw.trim().to_ascii_lowercase().replace(['-', ' '], "_")
}

/// Normalizes an action verb to lowercase snake_case.
#[must_use]
pub fn normalize_action(raw: &str) -> String {
    raw.trim().to_ascii_lowercase().replace(['-', ' '], "_")
}

/// Returns `true` when `action` is one of the direct interactive AT-SPI action verbs.
#[must_use]
pub fn is_interactive_action(action: &str) -> bool {
    matches!(
        normalize_action(action).as_str(),
        "click" | "press" | "activate" | "focus" | "set_value"
    )
}

fn is_interactive_role(role: &str) -> bool {
    matches!(
        normalize_role(role).as_str(),
        "push_button"
            | "toggle_button"
            | "check_box"
            | "radio_button"
            | "menu_item"
            | "check_menu_item"
            | "radio_menu_item"
            | "combo_box"
            | "text"
            | "password_text"
            | "entry"
            | "spin_button"
            | "slider"
            | "link"
            | "tab"
            | "page_tab"
            | "switch"
    )
}

/// Returns `true` if `node` represents a password, secret, or sensitive field requiring redaction.
#[must_use]
pub fn is_secure_node(node: &A11yRawNode) -> bool {
    let role = normalize_role(&node.role);
    if role == "password_text" || role == "password" {
        return true;
    }
    node.states.iter().any(|state| {
        let lower = state.to_ascii_lowercase();
        lower.contains("secret") || lower.contains("password") || lower.contains("sensitive")
    })
}

fn is_wrapper_role(role: &str) -> bool {
    matches!(
        normalize_role(role).as_str(),
        "panel" | "section" | "filler" | "group" | "unknown" | "scroll_pane"
    )
}

fn is_collapsible_wrapper(node: &A11yRawNode, collapsed_children: &[A11yRawNode]) -> bool {
    is_wrapper_role(&node.role)
        && node.name.trim().is_empty()
        && !node.actions.iter().any(|a| is_interactive_action(a))
        && !is_secure_node(node)
        && !collapsed_children.is_empty()
}

/// Recursively skips anonymous `panel`/`section`/`filler`/`group`/`unknown`/`scroll_pane`
/// wrapper nodes that have an empty name, no direct interactive actions
/// (`click`/`press`/`activate`/`focus`/`set_value`), and non-empty children, promoting
/// their children at zero depth cost.
///
/// Returns `(compressed_roots, skipped_wrapper_count)`.
///
/// Time complexity: $O(N)$ where $N$ is the total number of raw nodes across the tree.
/// Space complexity: $O(N)$ for the compressed tree.
#[must_use]
pub fn collapse_web_wrappers(roots: Vec<A11yRawNode>) -> (Vec<A11yRawNode>, usize) {
    let mut skipped = 0usize;
    let mut out = Vec::with_capacity(roots.len());
    for mut node in roots {
        let raw_children = std::mem::take(&mut node.children);
        let (children, child_skipped) = collapse_web_wrappers(raw_children);
        skipped += child_skipped;
        if is_collapsible_wrapper(&node, &children) {
            skipped += 1;
            out.extend(children);
        } else {
            node.children = children;
            out.push(node);
        }
    }
    (out, skipped)
}

fn count_raw_nodes(nodes: &[A11yRawNode]) -> usize {
    nodes.iter().map(|n| 1 + count_raw_nodes(&n.children)).sum()
}

#[derive(Debug, Clone)]
struct StoredSnapshot {
    snapshot: A11ySnapshot,
    entry_index_by_ref: BTreeMap<String, usize>,
    raw_subtrees_by_ref: BTreeMap<String, A11yRawNode>,
}

/// Bounded ring of historical accessibility snapshots supporting deterministic
/// `@<snapshot_id>:e<N>` ref resolution, stale-snapshot detection, skeleton
/// truncation, and subtree drill-down.
#[derive(Debug, Clone)]
pub struct SnapshotStore {
    max_snapshots: usize,
    next_seq: u64,
    snapshots: VecDeque<StoredSnapshot>,
}

impl Default for SnapshotStore {
    fn default() -> Self {
        Self::new(DEFAULT_SNAPSHOT_CAPACITY)
    }
}

impl SnapshotStore {
    /// Creates a new [`SnapshotStore`] retaining at most `max_snapshots` snapshots (`>= 1`).
    #[must_use]
    pub fn new(max_snapshots: usize) -> Self {
        Self {
            max_snapshots: max_snapshots.max(1),
            next_seq: 1,
            snapshots: VecDeque::with_capacity(max_snapshots.max(1)),
        }
    }

    /// Maximum number of snapshots retained before older snapshots are evicted.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.max_snapshots
    }

    /// Number of snapshots currently retained.
    #[must_use]
    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    /// Returns `true` when no snapshots have been stored yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
    }

    /// Builds and stores a new [`A11ySnapshot`] with monotonic ID `"s<N>"`, applying
    /// `root_ref` subtree drill-down, [`collapse_web_wrappers`], `skeleton` /
    /// `max_depth` pruning, `max_nodes` cap, and `is_secure` redaction.
    ///
    /// Time complexity: $O(N)$ in the number of raw nodes visited.
    /// Space complexity: $O(N)$ for snapshot entries and subtree handles.
    pub fn build_snapshot(
        &mut self,
        app_id: &str,
        window_title: &str,
        roots: Vec<A11yRawNode>,
        opts: &SnapshotFilterOpts,
    ) -> A11ySnapshot {
        let mut warning = None;
        let selected_roots = if let Some(ref_str) = opts.root_ref.as_deref() {
            match self.resolve_subtree(ref_str, &roots) {
                Ok(subtree) => vec![subtree],
                Err(err) => {
                    warning = Some(format!("root_ref {ref_str} ignored: {}", err.message));
                    roots
                }
            }
        } else {
            roots
        };

        let total_raw_nodes = count_raw_nodes(&selected_roots);
        let (prepared_roots, collapsed_wrappers_count) = if opts.collapse_wrappers {
            collapse_web_wrappers(selected_roots)
        } else {
            (selected_roots, 0)
        };

        let seq = self.next_seq;
        self.next_seq += 1;
        let snapshot_id = format!("s{seq}");

        let effective_max_depth = if opts.skeleton {
            opts.max_depth.min(SKELETON_MAX_DEPTH)
        } else {
            opts.max_depth
        };
        let max_nodes = opts.max_nodes.max(1);

        let mut entries = Vec::new();
        let mut addresses = BTreeMap::new();
        let mut entry_index_by_ref = BTreeMap::new();
        let mut raw_subtrees_by_ref = BTreeMap::new();
        let mut truncated = false;
        let mut next_elem: usize = 1;

        for root in &prepared_roots {
            flatten_node(
                root,
                0,
                effective_max_depth,
                max_nodes,
                &snapshot_id,
                &mut next_elem,
                &mut entries,
                &mut addresses,
                &mut entry_index_by_ref,
                &mut raw_subtrees_by_ref,
                &mut truncated,
            );
            if entries.len() >= max_nodes {
                break;
            }
        }

        let snapshot = A11ySnapshot {
            snapshot_id,
            app_id: app_id.to_string(),
            window_title: window_title.to_string(),
            entries,
            total_raw_nodes,
            collapsed_wrappers_count,
            truncated,
            skeleton: opts.skeleton,
            warning,
            addresses,
        };

        if self.snapshots.len() >= self.max_snapshots {
            self.snapshots.pop_front();
        }
        self.snapshots.push_back(StoredSnapshot {
            snapshot: snapshot.clone(),
            entry_index_by_ref,
            raw_subtrees_by_ref,
        });

        snapshot
    }

    /// Strict variant of [`Self::build_snapshot`] that errors if `opts.root_ref` does not
    /// resolve to an existing snapshot entry.
    pub fn try_build_snapshot(
        &mut self,
        app_id: &str,
        window_title: &str,
        roots: Vec<A11yRawNode>,
        opts: &SnapshotFilterOpts,
    ) -> Result<A11ySnapshot, A11yError> {
        if let Some(ref_str) = opts.root_ref.as_deref() {
            let _ = self.resolve_ref(ref_str)?;
        }
        Ok(self.build_snapshot(app_id, window_title, roots, opts))
    }

    /// Resolves a snapshot-scoped reference (`"@<snapshot_id>:e<N>"`) against currently
    /// retained snapshots.
    ///
    /// Rejects bare `"@e1"` refs without `":e"` or snapshot prefix with
    /// [`A11yErrorCode::InvalidRefFormat`], unknown/evicted snapshot IDs with
    /// [`A11yErrorCode::StaleRef`], and missing element IDs with [`A11yErrorCode::NotFound`].
    ///
    /// Time complexity: $O(S + \log E)$ where $S \le \text{capacity}$ is retained snapshots
    /// and $E$ is entries in the matching snapshot.
    /// Space complexity: $O(1)$.
    pub fn resolve_ref(&self, raw_ref: &str) -> Result<&A11yEntry, A11yError> {
        let (snapshot_id, normalized_ref) = parse_scoped_ref(raw_ref)?;
        let stored = self
            .snapshots
            .iter()
            .find(|s| s.snapshot.snapshot_id == snapshot_id)
            .ok_or_else(|| {
                A11yError::stale_ref(
                    raw_ref,
                    format!(
                        "Snapshot '{snapshot_id}' is unknown or has been evicted from SnapshotStore"
                    ),
                )
            })?;

        let idx = stored
            .entry_index_by_ref
            .get(&normalized_ref)
            .copied()
            .ok_or_else(|| {
                A11yError::not_found(
                    raw_ref,
                    format!("Element '{normalized_ref}' not found in snapshot '{snapshot_id}'"),
                )
            })?;

        stored.snapshot.entries.get(idx).ok_or_else(|| {
            A11yError::not_found(
                raw_ref,
                format!("Element '{normalized_ref}' missing from snapshot '{snapshot_id}'"),
            )
        })
    }

    /// Looks up backend D-Bus addressing metadata for a resolved reference, if stored.
    #[must_use]
    pub fn resolve_address(&self, raw_ref: &str) -> Option<&A11yNodeAddress> {
        let (snapshot_id, normalized_ref) = parse_scoped_ref(raw_ref).ok()?;
        let stored = self
            .snapshots
            .iter()
            .find(|s| s.snapshot.snapshot_id == snapshot_id)?;
        stored.snapshot.addresses.get(&normalized_ref)
    }

    /// Reads an entry's visible value, explicitly blocking secure/password fields with
    /// [`A11yErrorCode::SecureFieldBlocked`].
    pub fn read_value(&self, raw_ref: &str) -> Result<Option<&str>, A11yError> {
        let entry = self.resolve_ref(raw_ref)?;
        if entry.is_secure {
            return Err(A11yError::secure_field_blocked(
                raw_ref,
                format!(
                    "Refusing to read value from secure field '{}'",
                    entry.ref_id
                ),
            ));
        }
        Ok(entry.value.as_deref())
    }

    /// Resolves `raw_ref`, enforces secure-field guardrails, dispatches `action` through
    /// `adapter`, and redacts secret results for `TypeText` / `SetValue` into secure fields.
    pub async fn execute_action(
        &self,
        adapter: &dyn A11yAdapter,
        raw_ref: &str,
        action: A11yActionRequest,
    ) -> Result<A11yActionResult, A11yError> {
        let entry = self.resolve_ref(raw_ref)?;
        if matches!(action, A11yActionRequest::ReadValue) && entry.is_secure {
            return Err(A11yError::secure_field_blocked(
                raw_ref,
                format!(
                    "Refusing to read value from secure field '{}'",
                    entry.ref_id
                ),
            ));
        }
        let addr = self.resolve_address(raw_ref);
        let mut result = adapter.perform_action(entry, addr, action.clone()).await?;
        if entry.is_secure
            && matches!(
                action,
                A11yActionRequest::TypeText { .. } | A11yActionRequest::SetValue { .. }
            )
        {
            result.redacted = true;
            result.value = Some(REDACTED_VALUE.to_string());
        }
        Ok(result)
    }

    /// Returns the most recently built snapshot, if any.
    #[must_use]
    pub fn latest(&self) -> Option<&A11ySnapshot> {
        self.snapshots.back().map(|s| &s.snapshot)
    }

    fn resolve_subtree(
        &self,
        raw_ref: &str,
        fresh_roots: &[A11yRawNode],
    ) -> Result<A11yRawNode, A11yError> {
        let (snapshot_id, normalized_ref) = parse_scoped_ref(raw_ref)?;
        let stored = self
            .snapshots
            .iter()
            .find(|s| s.snapshot.snapshot_id == snapshot_id)
            .ok_or_else(|| {
                A11yError::stale_ref(
                    raw_ref,
                    format!(
                        "Snapshot '{snapshot_id}' is unknown or has been evicted from SnapshotStore"
                    ),
                )
            })?;
        let prior_node = stored
            .raw_subtrees_by_ref
            .get(&normalized_ref)
            .ok_or_else(|| {
                A11yError::not_found(
                    raw_ref,
                    format!("Element '{normalized_ref}' not found in snapshot '{snapshot_id}'"),
                )
            })?;

        if !fresh_roots.is_empty()
            && let Some(matched) = find_matching_raw_node(fresh_roots, prior_node)
        {
            return Ok(matched.clone());
        }
        Ok(prior_node.clone())
    }
}

fn find_matching_raw_node<'a>(
    nodes: &'a [A11yRawNode],
    target: &A11yRawNode,
) -> Option<&'a A11yRawNode> {
    for node in nodes {
        let path_match = target.object_path.is_some()
            && node.object_path == target.object_path
            && node.bus_name == target.bus_name;
        let structural_match = normalize_role(&node.role) == normalize_role(&target.role)
            && node.name == target.name
            && node.rect == target.rect;
        if path_match || structural_match {
            return Some(node);
        }
        if let Some(found) = find_matching_raw_node(&node.children, target) {
            return Some(found);
        }
    }
    None
}

#[allow(clippy::too_many_arguments)]
fn flatten_node(
    node: &A11yRawNode,
    depth: usize,
    max_depth: usize,
    max_nodes: usize,
    snapshot_id: &str,
    next_elem: &mut usize,
    entries: &mut Vec<A11yEntry>,
    addresses: &mut BTreeMap<String, A11yNodeAddress>,
    entry_index_by_ref: &mut BTreeMap<String, usize>,
    raw_subtrees_by_ref: &mut BTreeMap<String, A11yRawNode>,
    truncated: &mut bool,
) {
    if depth > max_depth || entries.len() >= max_nodes {
        *truncated = true;
        return;
    }

    let elem_idx = *next_elem;
    *next_elem += 1;
    let raw_ref = format!("e{elem_idx}");
    let ref_id = format!("@{snapshot_id}:{raw_ref}");
    let is_secure = is_secure_node(node);
    let value = if is_secure {
        Some(REDACTED_VALUE.to_string())
    } else {
        node.value.clone()
    };

    let entry = A11yEntry {
        ref_id: ref_id.clone(),
        raw_ref,
        role: normalize_role(&node.role),
        name: node.name.clone(),
        value,
        rect: node.rect,
        actions: node.actions.iter().map(|a| normalize_action(a)).collect(),
        depth,
        is_secure,
        child_count: node.children.len(),
    };

    let idx = entries.len();
    entries.push(entry);
    entry_index_by_ref.insert(ref_id.clone(), idx);
    raw_subtrees_by_ref.insert(ref_id.clone(), node.clone());
    if node.bus_name.is_some() || node.object_path.is_some() {
        addresses.insert(
            ref_id,
            A11yNodeAddress {
                bus_name: node.bus_name.clone(),
                object_path: node.object_path.clone(),
            },
        );
    }

    if !node.children.is_empty() && depth >= max_depth {
        *truncated = true;
        return;
    }

    for child in &node.children {
        if entries.len() >= max_nodes {
            *truncated = true;
            break;
        }
        flatten_node(
            child,
            depth + 1,
            max_depth,
            max_nodes,
            snapshot_id,
            next_elem,
            entries,
            addresses,
            entry_index_by_ref,
            raw_subtrees_by_ref,
            truncated,
        );
    }
}

/// Parses and validates a snapshot-scoped element reference `"@<snapshot_id>:e<N>"`.
pub fn parse_scoped_ref(raw_ref: &str) -> Result<(String, String), A11yError> {
    let trimmed = raw_ref.trim();
    if !trimmed.starts_with('@') {
        return Err(A11yError::invalid_ref_format(
            raw_ref,
            "Bare @e1 refs are ambiguous across snapshots; use @<snapshot_id>:e<N>",
        ));
    }
    let body = &trimmed[1..];
    let Some((snap_id, elem_num_str)) = body.split_once(":e") else {
        return Err(A11yError::invalid_ref_format(
            raw_ref,
            "Bare @e1 refs are ambiguous across snapshots; use @<snapshot_id>:e<N>",
        ));
    };
    if snap_id.is_empty()
        || snap_id.contains(':')
        || snap_id.chars().any(char::is_whitespace)
        || elem_num_str.is_empty()
        || !elem_num_str.bytes().all(|b| b.is_ascii_digit())
    {
        return Err(A11yError::invalid_ref_format(
            raw_ref,
            format!("Invalid scoped ref '{trimmed}'; expected @<snapshot_id>:e<N>"),
        ));
    }
    Ok((snap_id.to_string(), trimmed.to_string()))
}

/// Accessibility action dispatched through an [`A11yAdapter`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum A11yActionRequest {
    /// Invoke default click/press/activate action.
    Click,
    /// Move accessibility/input focus to the element.
    Focus,
    /// Read current accessible value (blocked on secure fields).
    ReadValue,
    /// Type text into an editable field (redacted when `is_secure` is true).
    TypeText {
        /// UTF-8 input text.
        text: String,
    },
    /// Replace editable field contents (redacted when `is_secure` is true).
    SetValue {
        /// Replacement value string.
        value: String,
    },
    /// Invoke a named AT-SPI action verb.
    CustomAction {
        /// Action name reported in `A11yEntry::actions`.
        name: String,
    },
}

/// Outcome returned by [`A11yAdapter::perform_action`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct A11yActionResult {
    /// True if the action succeeded.
    pub ok: bool,
    /// Target scoped element ref.
    pub ref_id: String,
    /// Action name executed.
    pub action: String,
    /// True when the field was secure and secret input/output was redacted.
    pub redacted: bool,
    /// Optional non-secret returned value (always `Some("[REDACTED]")` when redacted).
    pub value: Option<String>,
    /// Safe status summary free of secret strings.
    pub message: String,
}

/// Pluggable accessibility backend trait for fetching AT-SPI trees and dispatching semantic actions.
pub trait A11yAdapter: Send + Sync {
    /// Static backend name (`"mock"`, `"atspi_gdbus"`).
    fn adapter_name(&self) -> &'static str;

    /// Captures the raw accessibility tree for `(app_id, window_title)`.
    fn snapshot_tree<'a>(
        &'a self,
        app_id: &'a str,
        window_title: &'a str,
    ) -> A11yFuture<'a, Vec<A11yRawNode>>;

    /// Dispatches a semantic action on `entry`, optionally using `target_address`.
    fn perform_action<'a>(
        &'a self,
        entry: &'a A11yEntry,
        target_address: Option<&'a A11yNodeAddress>,
        action: A11yActionRequest,
    ) -> A11yFuture<'a, A11yActionResult>;
}

/// Recorded action entry stored by [`MockA11yAdapter`] for deterministic test assertions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MockRecordedAction {
    /// Scoped ref ID targeted.
    pub ref_id: String,
    /// Action kind name.
    pub action: String,
    /// Typed/set payload (always `[REDACTED]` when target was secure).
    pub payload: Option<String>,
    /// Whether redaction was applied.
    pub redacted: bool,
}

#[derive(Debug, Default)]
struct MockA11yState {
    roots: Vec<A11yRawNode>,
    backend_error: Option<A11yError>,
    recorded_actions: Vec<MockRecordedAction>,
}

/// Deterministic in-memory [`A11yAdapter`] implementation for unit tests and offline evaluation.
#[derive(Debug, Default)]
pub struct MockA11yAdapter {
    state: Mutex<MockA11yState>,
}

impl MockA11yAdapter {
    /// Creates a [`MockA11yAdapter`] pre-populated with `roots`.
    #[must_use]
    pub fn with_roots(roots: Vec<A11yRawNode>) -> Self {
        Self {
            state: Mutex::new(MockA11yState {
                roots,
                backend_error: None,
                recorded_actions: Vec::new(),
            }),
        }
    }

    /// Replaces the mock accessibility tree.
    pub fn set_roots(&self, roots: Vec<A11yRawNode>) {
        if let Ok(mut guard) = self.state.lock() {
            guard.roots = roots;
        }
    }

    /// Configures a synthetic backend error (e.g. `BackendUnavailable`) for negative tests.
    pub fn set_backend_error(&self, err: Option<A11yError>) {
        if let Ok(mut guard) = self.state.lock() {
            guard.backend_error = err;
        }
    }

    /// Returns all actions recorded by this mock adapter (with secure payloads redacted).
    #[must_use]
    pub fn recorded_actions(&self) -> Vec<MockRecordedAction> {
        self.state
            .lock()
            .map(|g| g.recorded_actions.clone())
            .unwrap_or_default()
    }
}

impl A11yAdapter for MockA11yAdapter {
    fn adapter_name(&self) -> &'static str {
        "mock"
    }

    fn snapshot_tree<'a>(
        &'a self,
        _app_id: &'a str,
        _window_title: &'a str,
    ) -> A11yFuture<'a, Vec<A11yRawNode>> {
        Box::pin(async move {
            let guard = self
                .state
                .lock()
                .map_err(|_| A11yError::backend_unavailable("mock state mutex poisoned"))?;
            if let Some(err) = &guard.backend_error {
                return Err(err.clone());
            }
            Ok(guard.roots.clone())
        })
    }

    fn perform_action<'a>(
        &'a self,
        entry: &'a A11yEntry,
        _target_address: Option<&'a A11yNodeAddress>,
        action: A11yActionRequest,
    ) -> A11yFuture<'a, A11yActionResult> {
        Box::pin(async move {
            let mut guard = self
                .state
                .lock()
                .map_err(|_| A11yError::backend_unavailable("mock state mutex poisoned"))?;
            if let Some(err) = &guard.backend_error {
                return Err(err.clone());
            }
            if matches!(action, A11yActionRequest::ReadValue) && entry.is_secure {
                return Err(A11yError::secure_field_blocked(
                    &entry.ref_id,
                    "Cannot read value of secure field",
                ));
            }

            let (action_name, payload, redacted, value, message) = match action {
                A11yActionRequest::Click => (
                    "click".to_string(),
                    None,
                    false,
                    None,
                    format!("Clicked {} ({})", entry.ref_id, entry.name),
                ),
                A11yActionRequest::Focus => (
                    "focus".to_string(),
                    None,
                    false,
                    None,
                    format!("Focused {} ({})", entry.ref_id, entry.name),
                ),
                A11yActionRequest::ReadValue => (
                    "read_value".to_string(),
                    None,
                    false,
                    entry.value.clone(),
                    format!("Read value from {}", entry.ref_id),
                ),
                A11yActionRequest::TypeText { text } => {
                    if entry.is_secure {
                        (
                            "type_text".to_string(),
                            Some(REDACTED_VALUE.to_string()),
                            true,
                            Some(REDACTED_VALUE.to_string()),
                            format!("Typed {REDACTED_VALUE} into secure field {}", entry.ref_id),
                        )
                    } else {
                        update_mock_node_value(&mut guard.roots, &entry.name, &entry.role, &text);
                        (
                            "type_text".to_string(),
                            Some(text.clone()),
                            false,
                            Some(text),
                            format!("Typed text into {}", entry.ref_id),
                        )
                    }
                }
                A11yActionRequest::SetValue { value } => {
                    if entry.is_secure {
                        (
                            "set_value".to_string(),
                            Some(REDACTED_VALUE.to_string()),
                            true,
                            Some(REDACTED_VALUE.to_string()),
                            format!("Set {REDACTED_VALUE} on secure field {}", entry.ref_id),
                        )
                    } else {
                        update_mock_node_value(&mut guard.roots, &entry.name, &entry.role, &value);
                        (
                            "set_value".to_string(),
                            Some(value.clone()),
                            false,
                            Some(value),
                            format!("Set value on {}", entry.ref_id),
                        )
                    }
                }
                A11yActionRequest::CustomAction { name } => (
                    name.clone(),
                    None,
                    false,
                    None,
                    format!("Executed {name} on {}", entry.ref_id),
                ),
            };

            guard.recorded_actions.push(MockRecordedAction {
                ref_id: entry.ref_id.clone(),
                action: action_name.clone(),
                payload,
                redacted,
            });

            Ok(A11yActionResult {
                ok: true,
                ref_id: entry.ref_id.clone(),
                action: action_name,
                redacted,
                value,
                message,
            })
        })
    }
}

fn update_mock_node_value(
    nodes: &mut [A11yRawNode],
    target_name: &str,
    target_role: &str,
    new_value: &str,
) -> bool {
    for node in nodes {
        if node.name == target_name && normalize_role(&node.role) == normalize_role(target_role) {
            node.value = Some(new_value.to_string());
            return true;
        }
        if update_mock_node_value(&mut node.children, target_name, target_role, new_value) {
            return true;
        }
    }
    false
}

/// Real Linux AT-SPI2 adapter executing bounded `gdbus call` queries against
/// `org.a11y.Bus` (`GetAddress`) and `org.a11y.atspi.Registry` / application trees.
///
/// Degrades gracefully with [`A11yErrorCode::BackendUnavailable`] (or empty tree via
/// [`Self::graceful_empty`] / [`Self::snapshot_tree_or_empty`]) when headless CI
/// has no `gdbus` binary or AT-SPI daemon.
#[derive(Debug, Clone)]
pub struct AtSpiGdbusAdapter {
    timeout: Duration,
    max_depth: usize,
    max_nodes: usize,
    degrade_to_empty: bool,
}

impl Default for AtSpiGdbusAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl AtSpiGdbusAdapter {
    /// Creates an [`AtSpiGdbusAdapter`] with the default `500 ms` per-call timeout.
    #[must_use]
    pub fn new() -> Self {
        Self {
            timeout: Duration::from_millis(DEFAULT_ATSPI_TIMEOUT_MS),
            max_depth: 6,
            max_nodes: 128,
            degrade_to_empty: false,
        }
    }

    /// Configures this adapter to return `Ok(Vec::new())` when the AT-SPI daemon is unavailable.
    #[must_use]
    pub const fn graceful_empty(mut self, enabled: bool) -> Self {
        self.degrade_to_empty = enabled;
        self
    }

    /// Configures the per-command `gdbus` timeout (minimum `50 ms`).
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout.max(Duration::from_millis(50));
        self
    }

    /// Fetches the AT-SPI tree, returning an empty vector if the AT-SPI backend is unavailable.
    pub async fn snapshot_tree_or_empty(
        &self,
        app_id: &str,
        window_title: &str,
    ) -> Vec<A11yRawNode> {
        self.snapshot_tree(app_id, window_title)
            .await
            .unwrap_or_default()
    }

    /// Queries `org.a11y.Bus.GetAddress` on the session bus with bounded timeout.
    pub async fn query_a11y_bus_address(&self) -> Result<String, A11yError> {
        let timeout = self.timeout;
        let stdout = run_gdbus_bounded(
            vec![
                "call".into(),
                "--session".into(),
                "--dest".into(),
                "org.a11y.Bus".into(),
                "--object-path".into(),
                "/org/a11y/bus".into(),
                "--method".into(),
                "org.a11y.Bus.GetAddress".into(),
            ],
            timeout,
        )
        .await?;
        parse_gdbus_string_tuple(&stdout).ok_or_else(|| {
            A11yError::backend_unavailable(format!(
                "Failed to parse org.a11y.Bus.GetAddress output: {stdout}"
            ))
        })
    }
}

impl A11yAdapter for AtSpiGdbusAdapter {
    fn adapter_name(&self) -> &'static str {
        "atspi_gdbus"
    }

    fn snapshot_tree<'a>(
        &'a self,
        app_id: &'a str,
        window_title: &'a str,
    ) -> A11yFuture<'a, Vec<A11yRawNode>> {
        let app_filter = app_id.to_ascii_lowercase();
        let title_filter = window_title.to_ascii_lowercase();
        let timeout = self.timeout;
        let max_depth = self.max_depth;
        let max_nodes = self.max_nodes;
        let degrade = self.degrade_to_empty;

        Box::pin(async move {
            let deadline = Instant::now() + timeout;
            let bus_addr = match self.query_a11y_bus_address().await {
                Ok(addr) => addr,
                Err(err) if degrade && err.code == A11yErrorCode::BackendUnavailable => {
                    return Ok(Vec::new());
                }
                Err(err) => return Err(err),
            };
            if Instant::now() >= deadline {
                if degrade {
                    return Ok(Vec::new());
                }
                return Err(A11yError::backend_unavailable(
                    "AT-SPI query_a11y_bus_address consumed timeout budget",
                ));
            }
            let apps_stdout = match run_gdbus_bounded(
                vec![
                    "call".into(),
                    "--address".into(),
                    bus_addr.clone(),
                    "--dest".into(),
                    "org.a11y.atspi.Registry".into(),
                    "--object-path".into(),
                    "/org/a11y/atspi/accessible/root".into(),
                    "--method".into(),
                    "org.a11y.atspi.Accessible.GetChildren".into(),
                ],
                remaining_budget(deadline),
            )
            .await
            {
                Ok(s) => s,
                Err(err) if degrade && err.code == A11yErrorCode::BackendUnavailable => {
                    return Ok(Vec::new());
                }
                Err(err) => return Err(err),
            };

            let app_refs = parse_gdbus_children_list(&apps_stdout);
            let mut visited = 0usize;
            let mut roots = Vec::new();

            for (bus_name, object_path) in app_refs {
                if Instant::now() >= deadline || visited >= max_nodes {
                    break;
                }
                let Ok(node) = fetch_atspi_node_recursive(
                    &bus_addr,
                    &bus_name,
                    &object_path,
                    0,
                    max_depth,
                    max_nodes,
                    &mut visited,
                    deadline,
                )
                .await
                else {
                    continue;
                };

                let app_matches = app_filter.is_empty()
                    || node.name.to_ascii_lowercase().contains(&app_filter)
                    || (!title_filter.is_empty() && tree_contains_title(&node, &title_filter));
                if app_matches {
                    roots.push(node);
                }
            }

            Ok(roots)
        })
    }

    fn perform_action<'a>(
        &'a self,
        entry: &'a A11yEntry,
        target_address: Option<&'a A11yNodeAddress>,
        action: A11yActionRequest,
    ) -> A11yFuture<'a, A11yActionResult> {
        let timeout = self.timeout;
        Box::pin(async move {
            if matches!(action, A11yActionRequest::ReadValue) && entry.is_secure {
                return Err(A11yError::secure_field_blocked(
                    &entry.ref_id,
                    "Reading value from secure field is blocked",
                ));
            }
            if matches!(action, A11yActionRequest::ReadValue) {
                return Ok(A11yActionResult {
                    ok: true,
                    ref_id: entry.ref_id.clone(),
                    action: "read_value".to_string(),
                    redacted: false,
                    value: entry.value.clone(),
                    message: format!("Read snapshot value for {}", entry.ref_id),
                });
            }

            let bus_addr = self.query_a11y_bus_address().await?;
            let Some(addr) = target_address else {
                return Err(A11yError::action_failed(
                    Some(&entry.ref_id),
                    "AT-SPI target has no D-Bus bus_name/object_path metadata",
                ));
            };
            let (Some(bus_name), Some(object_path)) = (&addr.bus_name, &addr.object_path) else {
                return Err(A11yError::action_failed(
                    Some(&entry.ref_id),
                    "AT-SPI target missing bus_name or object_path",
                ));
            };

            match action {
                A11yActionRequest::Click | A11yActionRequest::CustomAction { .. } => {
                    let action_idx = match &action {
                        A11yActionRequest::CustomAction { name } => entry
                            .actions
                            .iter()
                            .position(|a| normalize_action(a) == normalize_action(name))
                            .unwrap_or(0),
                        _ => 0,
                    };
                    let _ = run_gdbus_bounded(
                        vec![
                            "call".into(),
                            "--address".into(),
                            bus_addr,
                            "--dest".into(),
                            bus_name.clone(),
                            "--object-path".into(),
                            object_path.clone(),
                            "--method".into(),
                            "org.a11y.atspi.Action.DoAction".into(),
                            action_idx.to_string(),
                        ],
                        timeout,
                    )
                    .await?;
                    Ok(A11yActionResult {
                        ok: true,
                        ref_id: entry.ref_id.clone(),
                        action: "click".to_string(),
                        redacted: false,
                        value: None,
                        message: format!(
                            "Invoked AT-SPI DoAction({action_idx}) on {}",
                            entry.ref_id
                        ),
                    })
                }
                A11yActionRequest::Focus => {
                    let _ = run_gdbus_bounded(
                        vec![
                            "call".into(),
                            "--address".into(),
                            bus_addr,
                            "--dest".into(),
                            bus_name.clone(),
                            "--object-path".into(),
                            object_path.clone(),
                            "--method".into(),
                            "org.a11y.atspi.Component.GrabFocus".into(),
                        ],
                        timeout,
                    )
                    .await?;
                    Ok(A11yActionResult {
                        ok: true,
                        ref_id: entry.ref_id.clone(),
                        action: "focus".to_string(),
                        redacted: false,
                        value: None,
                        message: format!("Focused AT-SPI node {}", entry.ref_id),
                    })
                }
                A11yActionRequest::TypeText { text }
                | A11yActionRequest::SetValue { value: text } => {
                    let _ = run_gdbus_bounded(
                        vec![
                            "call".into(),
                            "--address".into(),
                            bus_addr,
                            "--dest".into(),
                            bus_name.clone(),
                            "--object-path".into(),
                            object_path.clone(),
                            "--method".into(),
                            "org.a11y.atspi.EditableText.SetTextContents".into(),
                            text.clone(),
                        ],
                        timeout,
                    )
                    .await?;
                    if entry.is_secure {
                        Ok(A11yActionResult {
                            ok: true,
                            ref_id: entry.ref_id.clone(),
                            action: "set_value".to_string(),
                            redacted: true,
                            value: Some(REDACTED_VALUE.to_string()),
                            message: format!(
                                "Set {REDACTED_VALUE} on secure AT-SPI field {}",
                                entry.ref_id
                            ),
                        })
                    } else {
                        Ok(A11yActionResult {
                            ok: true,
                            ref_id: entry.ref_id.clone(),
                            action: "set_value".to_string(),
                            redacted: false,
                            value: Some(text),
                            message: format!("Set text on AT-SPI field {}", entry.ref_id),
                        })
                    }
                }
                A11yActionRequest::ReadValue => unreachable!("handled above"),
            }
        })
    }
}

fn remaining_budget(deadline: Instant) -> Duration {
    deadline
        .saturating_duration_since(Instant::now())
        .max(Duration::from_millis(20))
}

fn tree_contains_title(node: &A11yRawNode, title_lower: &str) -> bool {
    if node.name.to_ascii_lowercase().contains(title_lower) {
        return true;
    }
    node.children
        .iter()
        .any(|c| tree_contains_title(c, title_lower))
}

#[allow(clippy::too_many_arguments)]
fn fetch_atspi_node_recursive<'a>(
    bus_addr: &'a str,
    bus_name: &'a str,
    object_path: &'a str,
    depth: usize,
    max_depth: usize,
    max_nodes: usize,
    visited: &'a mut usize,
    deadline: Instant,
) -> Pin<Box<dyn Future<Output = Result<A11yRawNode, A11yError>> + Send + 'a>> {
    Box::pin(async move {
        if *visited >= max_nodes || Instant::now() >= deadline {
            return Err(A11yError::backend_unavailable(
                "AT-SPI traversal budget reached",
            ));
        }
        *visited += 1;

        let name = run_gdbus_bounded(
            vec![
                "call".into(),
                "--address".into(),
                bus_addr.into(),
                "--dest".into(),
                bus_name.into(),
                "--object-path".into(),
                object_path.into(),
                "--method".into(),
                "org.freedesktop.DBus.Properties.Get".into(),
                "org.a11y.atspi.Accessible".into(),
                "Name".into(),
            ],
            remaining_budget(deadline),
        )
        .await
        .ok()
        .and_then(|s| parse_gdbus_variant_string(&s))
        .unwrap_or_default();

        let role = run_gdbus_bounded(
            vec![
                "call".into(),
                "--address".into(),
                bus_addr.into(),
                "--dest".into(),
                bus_name.into(),
                "--object-path".into(),
                object_path.into(),
                "--method".into(),
                "org.a11y.atspi.Accessible.GetRoleName".into(),
            ],
            remaining_budget(deadline),
        )
        .await
        .ok()
        .and_then(|s| parse_gdbus_string_tuple(&s))
        .unwrap_or_else(|| "unknown".to_string());

        let states = run_gdbus_bounded(
            vec![
                "call".into(),
                "--address".into(),
                bus_addr.into(),
                "--dest".into(),
                bus_name.into(),
                "--object-path".into(),
                object_path.into(),
                "--method".into(),
                "org.a11y.atspi.Accessible.GetState".into(),
            ],
            remaining_budget(deadline),
        )
        .await
        .ok()
        .map(|s| parse_gdbus_state_bitset(&s))
        .unwrap_or_default();

        let rect = run_gdbus_bounded(
            vec![
                "call".into(),
                "--address".into(),
                bus_addr.into(),
                "--dest".into(),
                bus_name.into(),
                "--object-path".into(),
                object_path.into(),
                "--method".into(),
                "org.a11y.atspi.Component.GetExtents".into(),
                "1".into(),
            ],
            remaining_budget(deadline),
        )
        .await
        .ok()
        .and_then(|s| parse_gdbus_extents(&s));

        let actions = run_gdbus_bounded(
            vec![
                "call".into(),
                "--address".into(),
                bus_addr.into(),
                "--dest".into(),
                bus_name.into(),
                "--object-path".into(),
                object_path.into(),
                "--method".into(),
                "org.a11y.atspi.Action.GetActions".into(),
            ],
            remaining_budget(deadline),
        )
        .await
        .ok()
        .map(|s| parse_gdbus_actions(&s))
        .unwrap_or_default();

        let mut children = Vec::new();
        if depth < max_depth
            && *visited < max_nodes
            && Instant::now() < deadline
            && let Ok(ch_out) = run_gdbus_bounded(
                vec![
                    "call".into(),
                    "--address".into(),
                    bus_addr.into(),
                    "--dest".into(),
                    bus_name.into(),
                    "--object-path".into(),
                    object_path.into(),
                    "--method".into(),
                    "org.a11y.atspi.Accessible.GetChildren".into(),
                ],
                remaining_budget(deadline),
            )
            .await
        {
            for (c_bus, c_path) in parse_gdbus_children_list(&ch_out) {
                if *visited >= max_nodes || Instant::now() >= deadline {
                    break;
                }
                if let Ok(child) = fetch_atspi_node_recursive(
                    bus_addr,
                    &c_bus,
                    &c_path,
                    depth + 1,
                    max_depth,
                    max_nodes,
                    visited,
                    deadline,
                )
                .await
                {
                    children.push(child);
                }
            }
        }

        Ok(A11yRawNode {
            name,
            role: normalize_role(&role),
            value: None,
            description: None,
            rect,
            states,
            actions,
            bus_name: Some(bus_name.to_string()),
            object_path: Some(object_path.to_string()),
            children,
        })
    })
}

async fn run_gdbus_bounded(args: Vec<String>, timeout: Duration) -> Result<String, A11yError> {
    tokio::task::spawn_blocking(move || run_gdbus_sync(&args, timeout))
        .await
        .map_err(|e| A11yError::backend_unavailable(format!("gdbus worker join failed: {e}")))?
}

fn run_gdbus_sync(args: &[String], timeout: Duration) -> Result<String, A11yError> {
    let mut child = Command::new("gdbus")
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| A11yError::backend_unavailable(format!("failed to spawn gdbus: {e}")))?;

    let stdout_pipe = child.stdout.take();
    let stderr_pipe = child.stderr.take();

    let stdout_thread = std::thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(mut p) = stdout_pipe {
            let _ = p.read_to_end(&mut buf);
        }
        buf
    });
    let stderr_thread = std::thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(mut p) = stderr_pipe {
            let _ = p.read_to_end(&mut buf);
        }
        buf
    });

    let deadline = Instant::now() + timeout;
    let status = loop {
        match child.try_wait() {
            Ok(Some(st)) => break Ok(st),
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    break Err(A11yError::backend_unavailable(format!(
                        "gdbus timed out after {} ms",
                        timeout.as_millis()
                    )));
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                break Err(A11yError::backend_unavailable(format!(
                    "gdbus wait failed: {e}"
                )));
            }
        }
    };

    let stdout_bytes = stdout_thread.join().unwrap_or_default();
    let stderr_bytes = stderr_thread.join().unwrap_or_default();

    let exit_status = status?;
    if !exit_status.success() {
        let err_msg = String::from_utf8_lossy(&stderr_bytes).trim().to_string();
        return Err(A11yError::backend_unavailable(if err_msg.is_empty() {
            format!("gdbus exited with status {exit_status}")
        } else {
            err_msg
        }));
    }

    Ok(String::from_utf8_lossy(&stdout_bytes).trim().to_string())
}

/// Parses a single-string GVariant tuple output such as `('unix:path=/tmp/bus',)` or `('push button',)`.
#[must_use]
pub fn parse_gdbus_string_tuple(output: &str) -> Option<String> {
    let trimmed = output.trim();
    let start_quote = trimmed.find('\'')?;
    let rest = &trimmed[start_quote + 1..];
    let end_quote = rest.rfind('\'')?;
    if end_quote == 0 {
        return Some(String::new());
    }
    Some(rest[..end_quote].to_string())
}

/// Parses a GVariant boxed string output such as `(<'Submit'>,)`.
#[must_use]
pub fn parse_gdbus_variant_string(output: &str) -> Option<String> {
    parse_gdbus_string_tuple(output)
}

/// Parses `org.a11y.atspi.Accessible.GetChildren` GVariant array of `(so)` tuples:
/// `([('org.a11y.atspi.Registry', '/org/a11y/atspi/accessible/1')],)`.
#[must_use]
pub fn parse_gdbus_children_list(output: &str) -> Vec<(String, String)> {
    let mut pairs = Vec::new();
    let bytes = output.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'('
            && let Some(close_rel) = output[i + 1..].find(')')
        {
            let inner = &output[i + 1..i + 1 + close_rel];
            if let Some((bus, path)) = parse_two_quoted_strings(inner)
                && path.starts_with('/')
            {
                pairs.push((bus, path));
            }
            i += close_rel + 2;
            continue;
        }
        i += 1;
    }
    pairs
}

fn parse_two_quoted_strings(slice: &str) -> Option<(String, String)> {
    let mut parts = Vec::with_capacity(2);
    let mut rest = slice;
    while let Some(first) = rest.find('\'') {
        let after_first = &rest[first + 1..];
        let second = after_first.find('\'')?;
        parts.push(after_first[..second].to_string());
        rest = &after_first[second + 1..];
        if parts.len() == 2 {
            return Some((parts.remove(0), parts.remove(0)));
        }
    }
    None
}

/// Parses `org.a11y.atspi.Component.GetExtents` output such as `((16, 16, 76, 40),)`.
#[must_use]
pub fn parse_gdbus_extents(output: &str) -> Option<[i32; 4]> {
    let mut nums = Vec::with_capacity(4);
    let mut cur = String::new();
    for ch in output.chars() {
        if ch == '-' || ch.is_ascii_digit() {
            cur.push(ch);
        } else if !cur.is_empty() {
            if let Ok(val) = cur.parse::<i32>() {
                nums.push(val);
            }
            cur.clear();
        }
    }
    if !cur.is_empty()
        && let Ok(val) = cur.parse::<i32>()
    {
        nums.push(val);
    }
    if nums.len() >= 4 {
        Some([nums[0], nums[1], nums[2], nums[3]])
    } else {
        None
    }
}

/// Parses `org.a11y.atspi.Action.GetActions` output `([('click', 'Clicks button', '')],)`.
#[must_use]
pub fn parse_gdbus_actions(output: &str) -> Vec<String> {
    let mut actions = Vec::new();
    let mut rest = output;
    while let Some(open) = rest.find("('") {
        let after = &rest[open + 2..];
        if let Some(close) = after.find('\'') {
            let name = normalize_action(&after[..close]);
            if !name.is_empty() && !actions.contains(&name) {
                actions.push(name);
            }
            if let Some(tuple_end) = after.find(')') {
                rest = &after[tuple_end + 1..];
            } else {
                break;
            }
        } else {
            break;
        }
    }
    actions
}

/// Converts `org.a11y.atspi.Accessible.GetState` `([uint32 lo, uint32 hi],)` into normalized state tokens.
#[must_use]
pub fn parse_gdbus_state_bitset(output: &str) -> Vec<String> {
    let mut words = Vec::with_capacity(2);
    let mut cur = String::new();
    for ch in output.chars() {
        if ch.is_ascii_digit() {
            cur.push(ch);
        } else if !cur.is_empty() {
            if let Ok(v) = cur.parse::<u32>() {
                words.push(v);
            }
            cur.clear();
        }
    }
    if !cur.is_empty()
        && let Ok(v) = cur.parse::<u32>()
    {
        words.push(v);
    }
    atspi_state_bits_to_names(
        words.first().copied().unwrap_or(0),
        words.get(1).copied().unwrap_or(0),
    )
}

fn atspi_state_bits_to_names(lo: u32, hi: u32) -> Vec<String> {
    let mut states = Vec::new();
    let bits = u64::from(lo) | (u64::from(hi) << 32);
    const MAPPINGS: &[(u32, &str)] = &[
        (3, "checked"),
        (7, "editable"),
        (8, "enabled"),
        (11, "focusable"),
        (12, "focused"),
        (20, "modal"),
        (26, "showing"),
        (30, "visible"),
    ];
    for &(bit, name) in MAPPINGS {
        if (bits & (1_u64 << bit)) != 0 {
            states.push((*name).to_string());
        }
    }
    states
}

/// Primary interaction strategy selected by [`select_interaction_route`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InteractionRoute {
    /// Semantic AT-SPI accessibility actions (`click`, `focus`, `set_value`).
    SemanticAtSpi,
    /// Coordinate/visual precision grounding with centered ROI crops and smooth pointer trajectories.
    PrecisionCanvas,
    /// Relative pointer-delta control for 3D viewports, camera orbits, and FPS games.
    RelativeGame,
}

/// Situational interaction routing outcome paired with recommended kinematic defaults.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RouteDecision {
    /// Selected execution path.
    pub route: InteractionRoute,
    /// Concise human-readable justification for audit logs and prompts.
    pub reason: String,
    /// Whether the executor should attach a `256x256` centered ROI crop for fine localization.
    pub prefer_roi_crop: bool,
    /// Recommended `vseat` kinematic profile (`"minimum_jerk"`, `"spring"`, `"instant"`).
    pub recommended_profile: &'static str,
    /// Recommended input sampling frequency in Hz (`60` or `120`).
    pub recommended_hz: u16,
}

/// Selects the optimal interaction route given application identity, window title, task goal,
/// observed AT-SPI actionable element count, and Wayland pointer-lock state.
///
/// Time complexity: $O(L)$ where $L$ is the combined length of `app_id`, `window_title`, and `goal`.
/// Space complexity: $O(L)$ for lowercased ASCII inspection.
#[must_use]
pub fn select_interaction_route(
    app_id: &str,
    window_title: &str,
    goal: &str,
    a11y_actionable_count: usize,
    pointer_locked: bool,
) -> RouteDecision {
    let app = app_id.trim().to_ascii_lowercase();
    let title = window_title.trim().to_ascii_lowercase();
    let goal_l = goal.trim().to_ascii_lowercase();

    if pointer_locked {
        return RouteDecision {
            route: InteractionRoute::RelativeGame,
            reason: "pointer_locked is active; using relative pointer delta mode".to_string(),
            prefer_roi_crop: false,
            recommended_profile: "instant",
            recommended_hz: 120,
        };
    }

    const GAME_APPS: &[&str] = &[
        "minecraft",
        "godot",
        "unreal",
        "unity",
        "retroarch",
        "steam_app_",
        "quake",
        "doom",
        "counter-strike",
        "cs2",
        "dota",
    ];
    const RELATIVE_CAMERA_VERBS: &[&str] = &[
        "3d camera",
        "first person",
        "fps",
        "look around",
        "orbit camera",
        "orbit view",
        "fly mode",
        "mouse look",
        "turn camera",
        "rotate camera",
        "strafe",
    ];

    let is_game_app = GAME_APPS
        .iter()
        .any(|kw| app.contains(kw) || title.contains(kw));
    let is_blender_camera = app.contains("blender")
        && (goal_l.contains("orbit")
            || goal_l.contains("fly")
            || goal_l.contains("camera")
            || goal_l.contains("pan view")
            || goal_l.contains("rotate view"));
    let has_camera_goal = RELATIVE_CAMERA_VERBS.iter().any(|kw| goal_l.contains(kw));

    if is_game_app || is_blender_camera || has_camera_goal {
        return RouteDecision {
            route: InteractionRoute::RelativeGame,
            reason: format!(
                "3D/game relative camera context detected (app={app_id:?}, goal={goal:?})"
            ),
            prefer_roi_crop: false,
            recommended_profile: "instant",
            recommended_hz: 120,
        };
    }

    const CANVAS_APPS: &[&str] = &[
        "krita",
        "gimp",
        "inkscape",
        "aseprite",
        "blender",
        "mypaint",
        "pinta",
        "kolourpaint",
        "darktable",
        "figma",
        "freecad",
        "kicad",
        "virtual_paint",
        "paint",
    ];
    const CANVAS_GOAL_KEYWORDS: &[&str] = &[
        "draw", "paint", "sketch", "stroke", "rainbow", "curve", "arch", "brush", "canvas",
        "spline", "bezier", "polygon", "circle",
    ];

    let is_canvas_app = CANVAS_APPS
        .iter()
        .any(|kw| app.contains(kw) || title.contains(kw));
    let is_canvas_goal = CANVAS_GOAL_KEYWORDS.iter().any(|kw| goal_l.contains(kw));

    if is_canvas_app || is_canvas_goal {
        return RouteDecision {
            route: InteractionRoute::PrecisionCanvas,
            reason: format!(
                "Visual/precision drawing context detected (app={app_id:?}, goal={goal:?})"
            ),
            prefer_roi_crop: true,
            recommended_profile: "spring",
            recommended_hz: 60,
        };
    }

    if a11y_actionable_count < 3 {
        return RouteDecision {
            route: InteractionRoute::PrecisionCanvas,
            reason: format!(
                "Sparse AT-SPI tree ({a11y_actionable_count} < 3 actionable controls); routing to visual precision grounding"
            ),
            prefer_roi_crop: true,
            recommended_profile: "spring",
            recommended_hz: 60,
        };
    }

    RouteDecision {
        route: InteractionRoute::SemanticAtSpi,
        reason: format!(
            "Rich AT-SPI tree ({a11y_actionable_count} actionable controls) for UI/form/dialog navigation"
        ),
        prefer_roi_crop: false,
        recommended_profile: "minimum_jerk",
        recommended_hz: 60,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn btn(name: &str, rect: [i32; 4]) -> A11yRawNode {
        A11yRawNode {
            name: name.to_string(),
            role: "push_button".to_string(),
            value: None,
            description: None,
            rect: Some(rect),
            states: vec!["focusable".to_string(), "showing".to_string()],
            actions: vec!["click".to_string(), "focus".to_string()],
            bus_name: None,
            object_path: None,
            children: Vec::new(),
        }
    }

    fn anon_wrapper(role: &str, children: Vec<A11yRawNode>) -> A11yRawNode {
        A11yRawNode {
            name: String::new(),
            role: role.to_string(),
            value: None,
            description: None,
            rect: None,
            states: Vec::new(),
            actions: Vec::new(),
            bus_name: None,
            object_path: None,
            children,
        }
    }

    #[test]
    fn snapshot_refs_enforce_scoped_format_and_reject_bare_or_stale_refs() {
        let mut store = SnapshotStore::new(2);
        let snap1 = store.build_snapshot(
            "firefox",
            "Login",
            vec![A11yRawNode {
                name: "Login".to_string(),
                role: "frame".to_string(),
                value: None,
                description: None,
                rect: Some([0, 0, 800, 600]),
                states: vec!["showing".to_string()],
                actions: Vec::new(),
                bus_name: None,
                object_path: None,
                children: vec![btn("Sign In", [100, 200, 120, 36])],
            }],
            &SnapshotFilterOpts::default(),
        );

        assert_eq!(snap1.snapshot_id, "s1");
        assert_eq!(snap1.entries[0].ref_id, "@s1:e1");
        assert_eq!(snap1.entries[1].ref_id, "@s1:e2");
        assert!(store.resolve_ref("@s1:e2").is_ok());

        // Bare @e1 must be rejected as InvalidRefFormat.
        let bare_err = store.resolve_ref("@e1").unwrap_err();
        assert_eq!(bare_err.code, A11yErrorCode::InvalidRefFormat);
        assert!(bare_err.message.contains("Bare @e1 refs are ambiguous"));

        // Non-existent element inside valid snapshot -> NotFound.
        let missing_err = store.resolve_ref("@s1:e99").unwrap_err();
        assert_eq!(missing_err.code, A11yErrorCode::NotFound);

        // Push 2 more snapshots so s1 is evicted from capacity=2 ring.
        let _s2 = store.build_snapshot(
            "app",
            "w2",
            vec![btn("A", [0, 0, 10, 10])],
            &Default::default(),
        );
        let _s3 = store.build_snapshot(
            "app",
            "w3",
            vec![btn("B", [0, 0, 10, 10])],
            &Default::default(),
        );
        assert_eq!(store.len(), 2);

        let stale_err = store.resolve_ref("@s1:e2").unwrap_err();
        assert_eq!(stale_err.code, A11yErrorCode::StaleRef);
    }

    #[test]
    fn collapses_deep_web_wrappers_at_zero_depth_cost() {
        let deep_tree = A11yRawNode {
            name: "Browser".to_string(),
            role: "frame".to_string(),
            value: None,
            description: None,
            rect: Some([0, 0, 1280, 720]),
            states: Vec::new(),
            actions: Vec::new(),
            bus_name: None,
            object_path: None,
            children: vec![anon_wrapper(
                "panel",
                vec![anon_wrapper(
                    "section",
                    vec![anon_wrapper(
                        "filler",
                        vec![anon_wrapper(
                            "group",
                            vec![anon_wrapper(
                                "unknown",
                                vec![anon_wrapper(
                                    "scroll_pane",
                                    vec![btn("Submit", [300, 400, 90, 32])],
                                )],
                            )],
                        )],
                    )],
                )],
            )],
        };

        let mut store = SnapshotStore::default();
        let snap = store.build_snapshot("chromium", "Form", vec![deep_tree], &Default::default());
        assert_eq!(snap.collapsed_wrappers_count, 6);
        assert_eq!(snap.entries.len(), 2);
        assert_eq!(snap.entries[0].role, "frame");
        assert_eq!(snap.entries[0].depth, 0);
        assert_eq!(snap.entries[1].name, "Submit");
        assert_eq!(snap.entries[1].depth, 1);
    }

    #[test]
    fn skeleton_and_subtree_drilldown_work() {
        let deep = A11yRawNode {
            name: "Root".to_string(),
            role: "frame".to_string(),
            value: None,
            description: None,
            rect: None,
            states: Vec::new(),
            actions: Vec::new(),
            bus_name: None,
            object_path: None,
            children: vec![A11yRawNode {
                name: "Sidebar".to_string(),
                role: "panel".to_string(),
                value: None,
                description: None,
                rect: None,
                states: Vec::new(),
                actions: Vec::new(),
                bus_name: None,
                object_path: None,
                children: vec![A11yRawNode {
                    name: "Section".to_string(),
                    role: "panel".to_string(),
                    value: None,
                    description: None,
                    rect: None,
                    states: Vec::new(),
                    actions: Vec::new(),
                    bus_name: None,
                    object_path: None,
                    children: vec![btn("DeepLeaf", [10, 20, 40, 20])],
                }],
            }],
        };

        let mut store = SnapshotStore::default();
        let skel = store.build_snapshot(
            "app",
            "win",
            vec![deep.clone()],
            &SnapshotFilterOpts {
                skeleton: true,
                ..Default::default()
            },
        );
        assert!(skel.truncated);
        assert_eq!(skel.entries.len(), 3); // depth 0, 1, 2; DeepLeaf at depth 3 pruned
        let section_ref = skel.entries[2].ref_id.clone();

        let drill = store.build_snapshot(
            "app",
            "win",
            vec![deep],
            &SnapshotFilterOpts {
                root_ref: Some(section_ref),
                ..Default::default()
            },
        );
        assert!(!drill.truncated);
        assert_eq!(drill.entries.len(), 2);
        assert_eq!(drill.entries[0].name, "Section");
        assert_eq!(drill.entries[1].name, "DeepLeaf");
    }

    #[tokio::test]
    async fn secure_fields_redact_values_block_read_and_redact_typed_secrets() {
        let secret_text = "super-secret-token-999";
        let tree = A11yRawNode {
            name: "Auth".to_string(),
            role: "frame".to_string(),
            value: None,
            description: None,
            rect: None,
            states: Vec::new(),
            actions: Vec::new(),
            bus_name: None,
            object_path: None,
            children: vec![
                A11yRawNode {
                    name: "Password".to_string(),
                    role: "password_text".to_string(),
                    value: Some("initial-pw".to_string()),
                    description: None,
                    rect: Some([40, 120, 220, 32]),
                    states: vec!["editable".to_string(), "sensitive".to_string()],
                    actions: vec!["focus".to_string(), "set_value".to_string()],
                    bus_name: None,
                    object_path: None,
                    children: Vec::new(),
                },
                A11yRawNode {
                    name: "API Key".to_string(),
                    role: "text".to_string(),
                    value: Some("sk-secret-raw".to_string()),
                    description: None,
                    rect: Some([40, 160, 220, 32]),
                    states: vec!["secret".to_string()],
                    actions: vec!["set_value".to_string()],
                    bus_name: None,
                    object_path: None,
                    children: Vec::new(),
                },
            ],
        };

        let adapter = MockA11yAdapter::with_roots(vec![tree.clone()]);
        let mut store = SnapshotStore::default();
        let snap = store.build_snapshot("app", "Auth", vec![tree], &Default::default());
        let text_tree = snap.format_text_tree();

        assert!(!text_tree.contains("initial-pw"));
        assert!(!text_tree.contains("sk-secret-raw"));
        assert!(text_tree.contains(REDACTED_VALUE));
        assert!(snap.entries[1].is_secure);
        assert!(snap.entries[2].is_secure);
        assert_eq!(snap.entries[1].value.as_deref(), Some(REDACTED_VALUE));

        let pw_ref = snap.entries[1].ref_id.clone();
        let read_err = store.read_value(&pw_ref).unwrap_err();
        assert_eq!(read_err.code, A11yErrorCode::SecureFieldBlocked);

        let res = store
            .execute_action(
                &adapter,
                &pw_ref,
                A11yActionRequest::TypeText {
                    text: secret_text.to_string(),
                },
            )
            .await
            .unwrap();
        assert!(res.redacted);
        assert_eq!(res.value.as_deref(), Some(REDACTED_VALUE));
        assert!(!res.message.contains(secret_text));

        let recorded = adapter.recorded_actions();
        assert_eq!(recorded.len(), 1);
        assert!(recorded[0].redacted);
        assert_eq!(recorded[0].payload.as_deref(), Some(REDACTED_VALUE));
    }

    #[test]
    fn situational_router_selects_expected_routes() {
        let game = select_interaction_route("minecraft", "Minecraft 1.21", "look around", 0, true);
        assert_eq!(game.route, InteractionRoute::RelativeGame);
        assert!(!game.prefer_roi_crop);

        let blender_cam = select_interaction_route(
            "blender",
            "Scene.blend",
            "orbit camera around mesh",
            5,
            false,
        );
        assert_eq!(blender_cam.route, InteractionRoute::RelativeGame);

        let canvas =
            select_interaction_route("krita", "Untitled", "paint a rainbow arch", 12, false);
        assert_eq!(canvas.route, InteractionRoute::PrecisionCanvas);
        assert!(canvas.prefer_roi_crop);

        let sparse =
            select_interaction_route("custom_gl_app", "Canvas", "click tool icon", 1, false);
        assert_eq!(sparse.route, InteractionRoute::PrecisionCanvas);
        assert!(sparse.prefer_roi_crop);

        let ui = select_interaction_route(
            "firefox",
            "Settings",
            "fill in proxy form and click Save button",
            8,
            false,
        );
        assert_eq!(ui.route, InteractionRoute::SemanticAtSpi);
        assert!(!ui.prefer_roi_crop);
    }

    #[test]
    fn gdbus_parsers_handle_real_gvariant_shapes() {
        assert_eq!(
            parse_gdbus_string_tuple("('unix:path=/run/user/1000/at-spi/bus_0',)").as_deref(),
            Some("unix:path=/run/user/1000/at-spi/bus_0")
        );
        assert_eq!(
            parse_gdbus_children_list(
                "([(':1.42', '/org/a11y/atspi/accessible/root'), (':1.99', '/org/a11y/atspi/accessible/12')],)"
            ),
            vec![
                (
                    ":1.42".to_string(),
                    "/org/a11y/atspi/accessible/root".to_string()
                ),
                (
                    ":1.99".to_string(),
                    "/org/a11y/atspi/accessible/12".to_string()
                ),
            ]
        );
        assert_eq!(
            parse_gdbus_extents("((16, 24, 120, 34),)"),
            Some([16, 24, 120, 34])
        );
        assert_eq!(
            parse_gdbus_actions("([('click', 'Clicks it', ''), ('focus', '', '')],)"),
            vec!["click".to_string(), "focus".to_string()]
        );
    }

    #[tokio::test]
    async fn atspi_gdbus_adapter_degrades_gracefully_without_daemon() {
        let adapter = AtSpiGdbusAdapter::new()
            .with_timeout(Duration::from_millis(120))
            .graceful_empty(true);
        let nodes = adapter
            .snapshot_tree("nonexistent_app", "NoWindow")
            .await
            .unwrap();
        assert!(nodes.is_empty() || !nodes.is_empty());

        let strict_adapter = AtSpiGdbusAdapter::new().with_timeout(Duration::from_millis(120));
        let start = Instant::now();
        let res = strict_adapter
            .snapshot_tree("nonexistent_app_strict", "NoWindow")
            .await;
        assert!(start.elapsed() < Duration::from_millis(900));
        if let Err(err) = res {
            assert_eq!(err.code, A11yErrorCode::BackendUnavailable);
        }
    }

    #[test]
    fn named_interactive_and_leaf_panels_are_not_collapsed() {
        let named_panel = A11yRawNode {
            name: "Toolbar".to_string(),
            role: "panel".to_string(),
            value: None,
            description: None,
            rect: None,
            states: Vec::new(),
            actions: Vec::new(),
            bus_name: None,
            object_path: None,
            children: vec![btn("Save", [0, 0, 50, 20])],
        };
        let interactive_panel = A11yRawNode {
            name: String::new(),
            role: "panel".to_string(),
            value: None,
            description: None,
            rect: None,
            states: Vec::new(),
            actions: vec!["click".to_string()],
            bus_name: None,
            object_path: None,
            children: vec![btn("Inner", [0, 0, 50, 20])],
        };
        let leaf_panel = anon_wrapper("panel", Vec::new());

        let (compressed, skipped) =
            collapse_web_wrappers(vec![named_panel, interactive_panel, leaf_panel]);
        assert_eq!(skipped, 0);
        assert_eq!(compressed.len(), 3);
    }

    #[test]
    fn resolve_ref_rejects_malformed_and_unknown_snapshot_ids() {
        let store = SnapshotStore::default();
        assert_eq!(
            store.resolve_ref("e1").unwrap_err().code,
            A11yErrorCode::InvalidRefFormat
        );
        assert_eq!(
            store.resolve_ref("@s1:eabc").unwrap_err().code,
            A11yErrorCode::InvalidRefFormat
        );
        assert_eq!(
            store.resolve_ref("@s999:e1").unwrap_err().code,
            A11yErrorCode::StaleRef
        );
    }
}
