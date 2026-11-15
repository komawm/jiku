#![forbid(unsafe_code)]

//! Pluggable three-tier memory & skill staging store for Jiku.
//!
//! - **L0**: Bounded sliding ring of recent workspace frames and 64-bit
//!   perceptual difference hashes (`dHash`) for zero-copy change detection.
//! - **L1**: Per-`app_id` UI quirk cards bounded by `app_quirk_max_chars`.
//! - **L2**: SQLite WAL + FTS5 trajectory index with BM25 search and sanitized
//!   `staged_skills` diff/approval promotion workflow.

use std::{
    collections::VecDeque,
    fmt::Write as _,
    fs,
    path::{Path, PathBuf},
    sync::Mutex,
    time::{SystemTime, UNIX_EPOCH},
};

use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Default maximum L0 recent frames kept per store instance.
pub const DEFAULT_L0_FRAME_RING_SIZE: usize = 4;

/// Default maximum character length of an L1 per-application quirk card.
pub const DEFAULT_APP_QUIRK_MAX_CHARS: usize = 600;

/// Maximum allowed size in bytes for a staged Lua/SOP skill artifact (64 KiB).
pub const MAX_STAGED_SKILL_BYTES: usize = 64 * 1024;

const FORBIDDEN_SKILL_TOKENS: &[&str] = &[
    "io.", "os.", "debug.", "package.", "loadfile", "dofile", "load(", "require(",
];

/// Errors returned by [`jiku-memory`](crate) storage and perceptual-hash APIs.
#[derive(Debug, Error)]
pub enum MemoryError {
    /// Invalid image buffer or frame dimensions passed to perceptual hashing.
    #[error("invalid RGBA8 frame buffer: {0}")]
    InvalidFrameBuffer(String),

    /// Candidate skill body failed safety sanitization or size checks.
    #[error("unsafe or invalid skill body: {0}")]
    UnsafeSkillBody(String),

    /// Requested skill ID was not found in `staged_skills`.
    #[error("staged skill not found: {0}")]
    SkillNotFound(String),

    /// Underlying filesystem error when creating database or promoted skill files.
    #[error("filesystem error: {0}")]
    Io(#[from] std::io::Error),

    /// Underlying SQLite / FTS5 storage error.
    #[error("sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
}

/// Entry stored in the L0 short-horizon frame perceptual-hash ring.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct L0FrameEntry {
    /// Compositor epoch observed for this frame.
    pub epoch: u64,
    /// Workspace identifier (e.g. `"ws_01"`).
    pub workspace: String,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// 64-bit difference hash (`dHash`) computed from luminance grid.
    pub dhash: u64,
    /// UNIX epoch timestamp in milliseconds.
    pub timestamp_ms: u64,
}

/// Computes a 64-bit horizontal luminance difference hash (`dHash`) over an
/// `RGBA8888` pixel buffer by strided sampling onto a `9 x 8` luminance grid.
///
/// Time complexity: $O(1)$ (fixed $9 \times 8 = 72$ point samples once buffer
/// length is validated).
/// Space complexity: $O(1)$ (`[u32; 72]` stack array).
pub fn compute_rgba_dhash64(rgba: &[u8], width: u32, height: u32) -> Result<u64, MemoryError> {
    if width == 0 || height == 0 {
        return Err(MemoryError::InvalidFrameBuffer(
            "width and height must be non-zero".to_string(),
        ));
    }
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|px| px.checked_mul(4))
        .ok_or_else(|| MemoryError::InvalidFrameBuffer("frame dimensions overflow".to_string()))?;
    if rgba.len() < expected {
        return Err(MemoryError::InvalidFrameBuffer(format!(
            "buffer length {} smaller than expected {} bytes",
            rgba.len(),
            expected
        )));
    }

    let mut grid = [0_u32; 9 * 8];
    for row in 0..8_u32 {
        let sy = ((u64::from(row) * 2 + 1) * u64::from(height)) / 16;
        let sy = (sy as usize).min(height as usize - 1);
        for col in 0..9_u32 {
            let sx = ((u64::from(col) * 2 + 1) * u64::from(width)) / 18;
            let sx = (sx as usize).min(width as usize - 1);
            let off = (sy * (width as usize) + sx) * 4;
            let r = u32::from(rgba[off]);
            let g = u32::from(rgba[off + 1]);
            let b = u32::from(rgba[off + 2]);
            // Standard BT.601 integer luma approximation scaled by 256.
            let luma = 77 * r + 150 * g + 29 * b;
            grid[(row as usize) * 9 + (col as usize)] = luma;
        }
    }

    let mut hash = 0_u64;
    let mut bit_idx = 0_u32;
    for row in 0..8_usize {
        for col in 0..8_usize {
            let left = grid[row * 9 + col];
            let right = grid[row * 9 + col + 1];
            if left < right {
                hash |= 1_u64 << bit_idx;
            }
            bit_idx += 1;
        }
    }
    Ok(hash)
}

/// Computes the Hamming distance between two 64-bit perceptual hashes.
///
/// Time complexity: $O(1)$.
/// Space complexity: $O(1)$.
#[must_use]
pub const fn hamming_distance64(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

/// Bounded L0 ring buffer tracking recent captured frames per workspace.
#[derive(Debug, Clone)]
pub struct L0FrameRing {
    capacity: usize,
    entries: VecDeque<L0FrameEntry>,
}

impl Default for L0FrameRing {
    fn default() -> Self {
        Self::new(DEFAULT_L0_FRAME_RING_SIZE)
    }
}

impl L0FrameRing {
    /// Creates an empty L0 ring bounded to `capacity` (minimum `1`).
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.max(1);
        Self {
            capacity: cap,
            entries: VecDeque::with_capacity(cap),
        }
    }

    /// Pushes a frame entry, evicting the oldest entry if full.
    pub fn push(&mut self, entry: L0FrameEntry) {
        if self.entries.len() >= self.capacity {
            self.entries.pop_front();
        }
        self.entries.push_back(entry);
    }

    /// Returns the most recent L0 entry for `workspace`, if present.
    #[must_use]
    pub fn latest_for_workspace(&self, workspace: &str) -> Option<&L0FrameEntry> {
        self.entries.iter().rev().find(|e| e.workspace == workspace)
    }

    /// Returns `true` when `workspace` already has a frame within
    /// `hamming_threshold` bits of `candidate_dhash`.
    #[must_use]
    pub fn is_unchanged(
        &self,
        workspace: &str,
        candidate_dhash: u64,
        hamming_threshold: u32,
    ) -> bool {
        self.latest_for_workspace(workspace).is_some_and(|prev| {
            hamming_distance64(prev.dhash, candidate_dhash) <= hamming_threshold
        })
    }

    /// Returns a snapshot of entries ordered oldest-to-newest.
    #[must_use]
    pub fn entries(&self) -> Vec<L0FrameEntry> {
        self.entries.iter().cloned().collect()
    }
}

/// L1 per-application UI quirk note record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AppQuirkRecord {
    /// Wayland `app_id` (e.g. `"org.PWMT.zathura"`).
    pub app_id: String,
    /// Concise guidance note truncated to `app_quirk_max_chars`.
    pub notes: String,
    /// Last update timestamp in milliseconds since UNIX epoch.
    pub updated_at_ms: u64,
}

/// Input payload for recording a completed execution episode in L2 memory.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NewEpisode {
    /// Target workspace identifier.
    pub workspace: String,
    /// User goal prompt or task title.
    pub goal: String,
    /// Route selected by `jikud` (`"fast_path_koma"`, `"turn_planner"`, `"human_verification"`).
    pub route: String,
    /// Human-readable episode summary.
    pub summary: String,
    /// Serialized JSON array of executed Koma RPC actions / steps.
    pub actions_json: String,
    /// Combined calibrated confidence (`0.0..=1.0`).
    pub confidence: f32,
    /// Wall-clock duration in milliseconds.
    pub duration_ms: u64,
}

/// Persisted L2 episode row returned by SQLite WAL + FTS5 queries.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EpisodeRecord {
    /// SQLite row identifier.
    pub id: i64,
    /// Target workspace identifier.
    pub workspace: String,
    /// User goal prompt or task title.
    pub goal: String,
    /// Route selected by `jikud`.
    pub route: String,
    /// Concise outcome summary.
    pub summary: String,
    /// Serialized JSON array of actions.
    pub actions_json: String,
    /// Combined confidence score.
    pub confidence: f32,
    /// Duration in milliseconds.
    pub duration_ms: u64,
    /// Creation timestamp in milliseconds since UNIX epoch.
    pub created_at_ms: u64,
}

/// Lifecycle state of a synthesized skill candidate in `staged_skills`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StagedSkillStatus {
    /// Awaiting human review via `jiku skills diff` / `jiku skills approve`.
    Staged,
    /// Reviewed and promoted into active skills.
    Approved,
    /// Explicitly rejected during review.
    Rejected,
}

impl StagedSkillStatus {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Staged => "staged",
            Self::Approved => "approved",
            Self::Rejected => "rejected",
        }
    }

    fn parse(raw: &str) -> Self {
        match raw {
            "approved" => Self::Approved,
            "rejected" => Self::Rejected,
            _ => Self::Staged,
        }
    }
}

/// Persisted skill candidate awaiting review or already promoted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StagedSkillRecord {
    /// Unique identifier / filename stem (e.g. `"export_pdf_macro"`).
    pub skill_id: String,
    /// Descriptive title.
    pub title: String,
    /// Candidate Lua or Markdown SOP body (sanitized).
    pub body: String,
    /// Prior active body if updating an existing skill (empty for new skills).
    pub baseline_body: String,
    /// Review status.
    pub status: StagedSkillStatus,
    /// Creation timestamp in milliseconds since UNIX epoch.
    pub created_at_ms: u64,
    /// Approval timestamp in milliseconds since UNIX epoch, if approved.
    pub approved_at_ms: Option<u64>,
}

/// Pluggable context and trajectory persistence interface for `jikud`.
pub trait ContextProvider: Send + Sync {
    /// Retrieves L1 UI quirk notes for `app_id`, if any.
    fn get_app_quirk(&self, app_id: &str) -> Result<Option<AppQuirkRecord>, MemoryError>;

    /// Stores or updates L1 UI quirk notes for `app_id`, truncating to the configured ceiling.
    fn upsert_app_quirk(&self, app_id: &str, notes: &str) -> Result<AppQuirkRecord, MemoryError>;

    /// Persists a completed episode in L2 SQLite + FTS5 index, returning its row ID.
    fn record_episode(&self, episode: &NewEpisode) -> Result<i64, MemoryError>;

    /// Searches historical episodes via FTS5 BM25 ranking.
    fn search_episodes(&self, query: &str, limit: usize)
    -> Result<Vec<EpisodeRecord>, MemoryError>;

    /// Lists recent L2 episodes ordered newest-first.
    fn recent_episodes(&self, limit: usize) -> Result<Vec<EpisodeRecord>, MemoryError>;

    /// Sanitizes and stages a synthesized Lua/SOP skill candidate for human review.
    fn stage_skill(
        &self,
        skill_id: &str,
        title: &str,
        body: &str,
        baseline_body: &str,
    ) -> Result<StagedSkillRecord, MemoryError>;

    /// Lists all skills recorded in `staged_skills`.
    fn list_skills(&self) -> Result<Vec<StagedSkillRecord>, MemoryError>;

    /// Produces a unified line diff between `baseline_body` and staged `body`.
    fn diff_skill(&self, skill_id: &str) -> Result<String, MemoryError>;

    /// Approves a staged skill and optionally writes the promoted file into `promote_dir`.
    fn approve_skill(
        &self,
        skill_id: &str,
        promote_dir: Option<&Path>,
    ) -> Result<(StagedSkillRecord, Option<PathBuf>), MemoryError>;
}

/// SQLite WAL + FTS5 implementation of [`ContextProvider`].
pub struct SqliteMemoryStore {
    conn: Mutex<Connection>,
    db_path: Option<PathBuf>,
    app_quirk_max_chars: usize,
}

impl SqliteMemoryStore {
    /// Opens or creates an on-disk SQLite WAL + FTS5 database at `path` (expanding `~/`).
    pub fn open(path: &Path, app_quirk_max_chars: usize) -> Result<Self, MemoryError> {
        let resolved = expand_tilde_path(path);
        if let Some(parent) = resolved.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(&resolved)?;
        init_sqlite_schema(&conn, true)?;
        Ok(Self {
            conn: Mutex::new(conn),
            db_path: Some(resolved),
            app_quirk_max_chars: app_quirk_max_chars.max(1),
        })
    }

    /// Opens an ephemeral in-memory SQLite + FTS5 database (for tests or `--no-db` runs).
    pub fn open_in_memory(app_quirk_max_chars: usize) -> Result<Self, MemoryError> {
        let conn = Connection::open_in_memory()?;
        init_sqlite_schema(&conn, false)?;
        Ok(Self {
            conn: Mutex::new(conn),
            db_path: None,
            app_quirk_max_chars: app_quirk_max_chars.max(1),
        })
    }

    /// Returns the resolved filesystem path when backed by an on-disk database.
    #[must_use]
    pub fn db_path(&self) -> Option<&Path> {
        self.db_path.as_deref()
    }
}

impl ContextProvider for SqliteMemoryStore {
    fn get_app_quirk(&self, app_id: &str) -> Result<Option<AppQuirkRecord>, MemoryError> {
        let guard = self.conn.lock().expect("memory mutex poisoned");
        let rec = guard
            .query_row(
                "SELECT app_id, notes, updated_at_ms FROM l1_app_quirks WHERE app_id = ?1",
                params![app_id],
                |row| {
                    Ok(AppQuirkRecord {
                        app_id: row.get(0)?,
                        notes: row.get(1)?,
                        updated_at_ms: row.get::<_, i64>(2)?.max(0) as u64,
                    })
                },
            )
            .optional()?;
        Ok(rec)
    }

    fn upsert_app_quirk(&self, app_id: &str, notes: &str) -> Result<AppQuirkRecord, MemoryError> {
        let clipped: String = notes
            .trim()
            .chars()
            .take(self.app_quirk_max_chars)
            .collect();
        let now = now_unix_ms();
        let guard = self.conn.lock().expect("memory mutex poisoned");
        guard.execute(
            "INSERT INTO l1_app_quirks (app_id, notes, updated_at_ms)
             VALUES (?1, ?2, ?3)
             ON CONFLICT(app_id) DO UPDATE SET notes = excluded.notes, updated_at_ms = excluded.updated_at_ms",
            params![app_id, clipped, now as i64],
        )?;
        Ok(AppQuirkRecord {
            app_id: app_id.to_string(),
            notes: clipped,
            updated_at_ms: now,
        })
    }

    fn record_episode(&self, episode: &NewEpisode) -> Result<i64, MemoryError> {
        let now = now_unix_ms();
        let guard = self.conn.lock().expect("memory mutex poisoned");
        guard.execute(
            "INSERT INTO l2_episodes (workspace, goal, route, summary, actions_json, confidence, duration_ms, created_at_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                episode.workspace,
                episode.goal,
                episode.route,
                episode.summary,
                episode.actions_json,
                f64::from(episode.confidence),
                episode.duration_ms as i64,
                now as i64,
            ],
        )?;
        Ok(guard.last_insert_rowid())
    }

    fn search_episodes(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<EpisodeRecord>, MemoryError> {
        let cap = limit.clamp(1, 200) as i64;
        let sanitized = sanitize_fts5_query(query);
        if sanitized.is_empty() {
            return self.recent_episodes(limit);
        }

        let guard = self.conn.lock().expect("memory mutex poisoned");
        let mut stmt = guard.prepare(
            "SELECT e.id, e.workspace, e.goal, e.route, e.summary, e.actions_json, e.confidence, e.duration_ms, e.created_at_ms
             FROM episodes_fts f
             JOIN l2_episodes e ON e.id = f.rowid
             WHERE episodes_fts MATCH ?1
             ORDER BY bm25(episodes_fts), e.created_at_ms DESC
             LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![sanitized, cap], map_episode_row)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    fn recent_episodes(&self, limit: usize) -> Result<Vec<EpisodeRecord>, MemoryError> {
        let cap = limit.clamp(1, 200) as i64;
        let guard = self.conn.lock().expect("memory mutex poisoned");
        let mut stmt = guard.prepare(
            "SELECT id, workspace, goal, route, summary, actions_json, confidence, duration_ms, created_at_ms
             FROM l2_episodes
             ORDER BY created_at_ms DESC, id DESC
             LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![cap], map_episode_row)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    fn stage_skill(
        &self,
        skill_id: &str,
        title: &str,
        body: &str,
        baseline_body: &str,
    ) -> Result<StagedSkillRecord, MemoryError> {
        let clean_id = sanitize_skill_id(skill_id)?;
        validate_skill_safety(body)?;
        let now = now_unix_ms();
        let guard = self.conn.lock().expect("memory mutex poisoned");
        guard.execute(
            "INSERT INTO staged_skills (skill_id, title, body, baseline_body, status, created_at_ms, approved_at_ms)
             VALUES (?1, ?2, ?3, ?4, 'staged', ?5, NULL)
             ON CONFLICT(skill_id) DO UPDATE SET
               title = excluded.title,
               body = excluded.body,
               baseline_body = excluded.baseline_body,
               status = 'staged',
               created_at_ms = excluded.created_at_ms,
               approved_at_ms = NULL",
            params![clean_id, title.trim(), body, baseline_body, now as i64],
        )?;
        Ok(StagedSkillRecord {
            skill_id: clean_id,
            title: title.trim().to_string(),
            body: body.to_string(),
            baseline_body: baseline_body.to_string(),
            status: StagedSkillStatus::Staged,
            created_at_ms: now,
            approved_at_ms: None,
        })
    }

    fn list_skills(&self) -> Result<Vec<StagedSkillRecord>, MemoryError> {
        let guard = self.conn.lock().expect("memory mutex poisoned");
        let mut stmt = guard.prepare(
            "SELECT skill_id, title, body, baseline_body, status, created_at_ms, approved_at_ms
             FROM staged_skills
             ORDER BY created_at_ms DESC, skill_id ASC",
        )?;
        let rows = stmt.query_map([], map_skill_row)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    fn diff_skill(&self, skill_id: &str) -> Result<String, MemoryError> {
        let rec = load_skill_by_id(&self.conn, skill_id)?;
        Ok(compute_unified_diff(
            &rec.skill_id,
            &rec.baseline_body,
            &rec.body,
        ))
    }

    fn approve_skill(
        &self,
        skill_id: &str,
        promote_dir: Option<&Path>,
    ) -> Result<(StagedSkillRecord, Option<PathBuf>), MemoryError> {
        let rec = load_skill_by_id(&self.conn, skill_id)?;
        validate_skill_safety(&rec.body)?;
        let now = now_unix_ms();

        let promoted_path = if let Some(dir) = promote_dir {
            let resolved_dir = expand_tilde_path(dir);
            fs::create_dir_all(&resolved_dir)?;
            let ext = if rec.body.trim_start().starts_with("--")
                || rec.body.contains("function")
                || rec.skill_id.ends_with("_lua")
            {
                "lua"
            } else {
                "md"
            };
            let dest = resolved_dir.join(format!("{}.{ext}", rec.skill_id));
            fs::write(&dest, rec.body.as_bytes())?;
            Some(dest)
        } else {
            None
        };

        let guard = self.conn.lock().expect("memory mutex poisoned");
        guard.execute(
            "UPDATE staged_skills SET status = 'approved', approved_at_ms = ?2 WHERE skill_id = ?1",
            params![rec.skill_id, now as i64],
        )?;

        Ok((
            StagedSkillRecord {
                status: StagedSkillStatus::Approved,
                approved_at_ms: Some(now),
                ..rec
            },
            promoted_path,
        ))
    }
}

/// Rejects candidate Lua/SOP bodies exceeding size limits or referencing unsafe
/// globals stripped from Jiku's Phodopus Lua sandbox.
///
/// Time complexity: $O(B \cdot K)$ where $B$ is `body.len()` and $K = 8$ forbidden tokens.
/// Space complexity: $O(B)$ for ASCII lowercasing.
pub fn validate_skill_safety(body: &str) -> Result<(), MemoryError> {
    if body.trim().is_empty() {
        return Err(MemoryError::UnsafeSkillBody(
            "skill body must not be empty".to_string(),
        ));
    }
    if body.len() > MAX_STAGED_SKILL_BYTES {
        return Err(MemoryError::UnsafeSkillBody(format!(
            "skill body exceeds {MAX_STAGED_SKILL_BYTES} bytes"
        )));
    }
    let lower = body.to_ascii_lowercase();
    for token in FORBIDDEN_SKILL_TOKENS {
        if lower.contains(token) {
            return Err(MemoryError::UnsafeSkillBody(format!(
                "forbidden sandbox escape token '{token}' detected"
            )));
        }
    }
    Ok(())
}

/// Computes a readable line diff between `baseline` and `staged`.
///
/// Time complexity: $O(M \cdot N)$ worst-case LCS over line counts $M, N$ (bounded by small skill scripts).
/// Space complexity: $O(M \cdot N)$ DP table.
#[must_use]
pub fn compute_unified_diff(skill_id: &str, baseline: &str, staged: &str) -> String {
    let a: Vec<&str> = baseline.lines().collect();
    let b: Vec<&str> = staged.lines().collect();
    let m = a.len();
    let n = b.len();

    let mut dp = vec![vec![0_usize; n + 1]; m + 1];
    for i in (0..m).rev() {
        for j in (0..n).rev() {
            dp[i][j] = if a[i] == b[j] {
                dp[i + 1][j + 1] + 1
            } else {
                dp[i + 1][j].max(dp[i][j + 1])
            };
        }
    }

    let mut out = format!("--- a/{skill_id}\n+++ b/{skill_id}\n");
    let mut i = 0;
    let mut j = 0;
    while i < m && j < n {
        if a[i] == b[j] {
            let _ = writeln!(out, " {}", a[i]);
            i += 1;
            j += 1;
        } else if dp[i + 1][j] >= dp[i][j + 1] {
            let _ = writeln!(out, "-{}", a[i]);
            i += 1;
        } else {
            let _ = writeln!(out, "+{}", b[j]);
            j += 1;
        }
    }
    while i < m {
        let _ = writeln!(out, "-{}", a[i]);
        i += 1;
    }
    while j < n {
        let _ = writeln!(out, "+{}", b[j]);
        j += 1;
    }
    out
}

/// Converts arbitrary user query text into a safe FTS5 `MATCH` expression by
/// quoting alphanumeric/underscore tokens joined with ` OR `.
#[must_use]
pub fn sanitize_fts5_query(raw: &str) -> String {
    let tokens: Vec<String> = raw
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|tok| !tok.is_empty())
        .map(|tok| format!("\"{tok}\""))
        .collect();
    tokens.join(" OR ")
}

/// Expands a leading `~/` prefix using `$HOME` if present.
#[must_use]
pub fn expand_tilde_path(path: &Path) -> PathBuf {
    let s = path.to_string_lossy();
    if let Some(rest) = s.strip_prefix("~/")
        && let Ok(home) = std::env::var("HOME")
    {
        return PathBuf::from(home).join(rest);
    }
    path.to_path_buf()
}

fn sanitize_skill_id(raw: &str) -> Result<String, MemoryError> {
    let trimmed = raw.trim();
    if trimmed.is_empty()
        || trimmed.contains("..")
        || trimmed.contains('/')
        || trimmed.contains('\\')
        || !trimmed
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'))
    {
        return Err(MemoryError::UnsafeSkillBody(format!(
            "invalid skill_id '{raw}'"
        )));
    }
    Ok(trimmed.to_string())
}

fn init_sqlite_schema(conn: &Connection, enable_wal: bool) -> Result<(), rusqlite::Error> {
    if enable_wal {
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
    }
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS l1_app_quirks (
            app_id TEXT PRIMARY KEY,
            notes TEXT NOT NULL,
            updated_at_ms INTEGER NOT NULL
         );

         CREATE TABLE IF NOT EXISTS l2_episodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            workspace TEXT NOT NULL,
            goal TEXT NOT NULL,
            route TEXT NOT NULL,
            summary TEXT NOT NULL,
            actions_json TEXT NOT NULL,
            confidence REAL NOT NULL,
            duration_ms INTEGER NOT NULL,
            created_at_ms INTEGER NOT NULL
         );

         CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
            workspace,
            goal,
            summary,
            content='l2_episodes',
            content_rowid='id'
         );

         CREATE TRIGGER IF NOT EXISTS l2_episodes_ai AFTER INSERT ON l2_episodes BEGIN
            INSERT INTO episodes_fts(rowid, workspace, goal, summary)
            VALUES (new.id, new.workspace, new.goal, new.summary);
         END;

         CREATE TRIGGER IF NOT EXISTS l2_episodes_ad AFTER DELETE ON l2_episodes BEGIN
            INSERT INTO episodes_fts(episodes_fts, rowid, workspace, goal, summary)
            VALUES ('delete', old.id, old.workspace, old.goal, old.summary);
         END;

         CREATE TRIGGER IF NOT EXISTS l2_episodes_au AFTER UPDATE ON l2_episodes BEGIN
            INSERT INTO episodes_fts(episodes_fts, rowid, workspace, goal, summary)
            VALUES ('delete', old.id, old.workspace, old.goal, old.summary);
            INSERT INTO episodes_fts(rowid, workspace, goal, summary)
            VALUES (new.id, new.workspace, new.goal, new.summary);
         END;

         CREATE TABLE IF NOT EXISTS staged_skills (
            skill_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            body TEXT NOT NULL,
            baseline_body TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at_ms INTEGER NOT NULL,
            approved_at_ms INTEGER
         );",
    )?;
    Ok(())
}

fn load_skill_by_id(
    mutex: &Mutex<Connection>,
    skill_id: &str,
) -> Result<StagedSkillRecord, MemoryError> {
    let guard = mutex.lock().expect("memory mutex poisoned");
    guard
        .query_row(
            "SELECT skill_id, title, body, baseline_body, status, created_at_ms, approved_at_ms
             FROM staged_skills WHERE skill_id = ?1",
            params![skill_id],
            map_skill_row,
        )
        .optional()?
        .ok_or_else(|| MemoryError::SkillNotFound(skill_id.to_string()))
}

fn map_episode_row(row: &rusqlite::Row<'_>) -> Result<EpisodeRecord, rusqlite::Error> {
    Ok(EpisodeRecord {
        id: row.get(0)?,
        workspace: row.get(1)?,
        goal: row.get(2)?,
        route: row.get(3)?,
        summary: row.get(4)?,
        actions_json: row.get(5)?,
        confidence: row.get::<_, f64>(6)? as f32,
        duration_ms: row.get::<_, i64>(7)?.max(0) as u64,
        created_at_ms: row.get::<_, i64>(8)?.max(0) as u64,
    })
}

fn map_skill_row(row: &rusqlite::Row<'_>) -> Result<StagedSkillRecord, rusqlite::Error> {
    let status_str: String = row.get(4)?;
    let approved: Option<i64> = row.get(6)?;
    Ok(StagedSkillRecord {
        skill_id: row.get(0)?,
        title: row.get(1)?,
        body: row.get(2)?,
        baseline_body: row.get(3)?,
        status: StagedSkillStatus::parse(&status_str),
        created_at_ms: row.get::<_, i64>(5)?.max(0) as u64,
        approved_at_ms: approved.map(|v| v.max(0) as u64),
    })
}

fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dhash_detects_identical_vs_flipped_gradient_frames() {
        // Horizontal brightening gradient left->right (40x20 RGBA).
        let mut left_to_right = vec![0_u8; 40 * 20 * 4];
        let mut right_to_left = vec![0_u8; 40 * 20 * 4];
        for y in 0..20 {
            for x in 0..40 {
                let off = (y * 40 + x) * 4;
                let v = ((x * 255) / 39) as u8;
                left_to_right[off..off + 4].copy_from_slice(&[v, v, v, 255]);
                let inv = 255 - v;
                right_to_left[off..off + 4].copy_from_slice(&[inv, inv, inv, 255]);
            }
        }

        let h1 = compute_rgba_dhash64(&left_to_right, 40, 20).unwrap();
        let h1_dup = compute_rgba_dhash64(&left_to_right, 40, 20).unwrap();
        let h2 = compute_rgba_dhash64(&right_to_left, 40, 20).unwrap();

        assert_eq!(hamming_distance64(h1, h1_dup), 0);
        assert!(hamming_distance64(h1, h2) >= 48);

        let mut ring = L0FrameRing::new(2);
        ring.push(L0FrameEntry {
            epoch: 1,
            workspace: "ws_01".to_string(),
            width: 40,
            height: 20,
            dhash: h1,
            timestamp_ms: 100,
        });
        assert!(ring.is_unchanged("ws_01", h1_dup, 4));
        assert!(!ring.is_unchanged("ws_01", h2, 4));
    }

    #[test]
    fn sqlite_wal_fts5_stores_l1_l2_and_promotes_sanitized_skills() {
        let dir = std::env::temp_dir().join(format!(
            "jiku-mem-test-{}",
            now_unix_ms() ^ u64::from(std::process::id())
        ));
        let db_path = dir.join("state.db");
        let promote_dir = dir.join("skills");

        let store = SqliteMemoryStore::open(&db_path, 24).unwrap();
        let quirk = store
            .upsert_app_quirk(
                "org.PWMT.zathura",
                "Use j/k navigation instead of scrollbars",
            )
            .unwrap();
        assert_eq!(quirk.notes.chars().count(), 24);
        assert_eq!(
            store
                .get_app_quirk("org.PWMT.zathura")
                .unwrap()
                .unwrap()
                .notes,
            quirk.notes
        );

        let ep1 = store
            .record_episode(&NewEpisode {
                workspace: "ws_01".to_string(),
                goal: "Open foot terminal and inspect git status".to_string(),
                route: "fast_path_koma".to_string(),
                summary: "Clicked foot window at cell G7 and verified git branch.".to_string(),
                actions_json: r#"["vseat.click G7"]"#.to_string(),
                confidence: 0.94,
                duration_ms: 180,
            })
            .unwrap();
        assert!(ep1 > 0);
        store
            .record_episode(&NewEpisode {
                workspace: "ws_02".to_string(),
                goal: "Sort receipts folder".to_string(),
                route: "turn_planner".to_string(),
                summary: "Classified three PNG invoices.".to_string(),
                actions_json: "[]".to_string(),
                confidence: 0.72,
                duration_ms: 1200,
            })
            .unwrap();

        let hits = store
            .search_episodes("foot terminal git-status!", 5)
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, ep1);

        // Unsafe Lua primitives must fail sanitization before staging.
        assert!(matches!(
            store.stage_skill("bad_skill", "Unsafe", "os.execute('id')", ""),
            Err(MemoryError::UnsafeSkillBody(_))
        ));

        let staged = store
            .stage_skill(
                "zathura_next_page",
                "Advance page",
                "-- Safe macro\njiku.notify({ summary = \"Next page\" })\n",
                "-- Safe macro\n",
            )
            .unwrap();
        assert_eq!(staged.status, StagedSkillStatus::Staged);
        let diff = store.diff_skill("zathura_next_page").unwrap();
        assert!(diff.contains("+jiku.notify"));

        let (approved, promoted_file) = store
            .approve_skill("zathura_next_page", Some(&promote_dir))
            .unwrap();
        assert_eq!(approved.status, StagedSkillStatus::Approved);
        let promoted_file = promoted_file.expect("promoted skill file");
        assert!(promoted_file.is_file());

        let _ = fs::remove_dir_all(&dir);
    }
}
