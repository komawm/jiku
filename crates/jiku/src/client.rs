//! Unified `jiku` controller client preferring `jikud`'s Unix domain socket (`jiku.sock`)
//! with automatic in-process [`JikuEngine`] fallback when no daemon is running.

use std::{
    env,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use anyhow::anyhow;
use jiku_memory::{EpisodeRecord, StagedSkillRecord, expand_tilde_path};
use jikud::{
    config::JikuConfig,
    daemon::{
        DaemonRpcRequest, DaemonStatusReport, JikuEngine, RpcDecisionQuestion, RunTaskReport,
        send_daemon_rpc,
    },
    drivers::{decisions::DecisionBatchResponse, turn::TurnResponse},
    jikubox::JikuBoxReport,
};
use serde_json::{Value, json};
use tokio::net::UnixStream;

/// Connection options shared by the `jiku` CLI and MCP server.
#[derive(Debug, Clone, Default)]
pub struct JikuClientOptions {
    /// Explicit `--config` path.
    pub config_path: Option<PathBuf>,
    /// Explicit `--policy` path.
    pub policy_path: Option<PathBuf>,
    /// Explicit `--socket` path overriding `$JIKU_SOCKET` and config defaults.
    pub socket_path: Option<PathBuf>,
    /// Open an ephemeral in-memory SQLite store when running in-process (for tests).
    pub in_memory_db: bool,
}

/// Controller client talking either to a live `jikud` socket or directly to [`JikuEngine`].
pub struct JikuClient {
    options: JikuClientOptions,
    socket_path: PathBuf,
    in_process: Option<JikuEngine>,
    next_id: AtomicU64,
}

impl JikuClient {
    /// Resolves socket configuration from `options` without prematurely spawning a daemon.
    pub fn new(options: JikuClientOptions) -> anyhow::Result<Self> {
        let (cfg, _) = JikuConfig::load_resolved(options.config_path.as_deref())?;
        let socket_path = options
            .socket_path
            .clone()
            .or_else(|| {
                env::var_os("JIKU_SOCKET")
                    .filter(|v| !v.is_empty())
                    .map(PathBuf::from)
            })
            .map(|p| expand_tilde_path(&p))
            .unwrap_or_else(|| cfg.daemon_socket_path());

        Ok(Self {
            options,
            socket_path,
            in_process: None,
            next_id: AtomicU64::new(1),
        })
    }

    async fn try_daemon_rpc(&self, method: &str, params: Value) -> Option<anyhow::Result<Value>> {
        if !self.socket_path.exists() {
            return None;
        }
        if UnixStream::connect(&self.socket_path).await.is_err() {
            return None;
        }
        let req = DaemonRpcRequest {
            id: self.next_id.fetch_add(1, Ordering::Relaxed),
            method: method.to_string(),
            params,
        };
        Some(match send_daemon_rpc(&self.socket_path, &req).await {
            Ok(resp) => {
                if let Some(err) = resp.error {
                    Err(anyhow!("jikud error [{}]: {}", err.code, err.message))
                } else {
                    Ok(resp.result.unwrap_or(Value::Null))
                }
            }
            Err(e) => Err(e),
        })
    }

    fn engine_mut(&mut self) -> anyhow::Result<&mut JikuEngine> {
        if self.in_process.is_none() {
            let eng = JikuEngine::from_resolved(
                self.options.config_path.as_deref(),
                self.options.policy_path.as_deref(),
                self.options.in_memory_db,
            )?;
            self.in_process = Some(eng);
        }
        Ok(self.in_process.as_mut().expect("engine initialized"))
    }

    /// Queries daemon, driver, memory, and Koma compositor health.
    pub async fn status(&mut self) -> anyhow::Result<DaemonStatusReport> {
        if let Some(res) = self.try_daemon_rpc("status", json!({})).await {
            return Ok(serde_json::from_value(res?)?);
        }
        self.engine_mut()?.status()
    }

    /// Evaluates a calibrated batch of `Noul` / `Choice` questions.
    pub async fn decide(
        &mut self,
        image_path: Option<&Path>,
        prompt: &str,
        questions: &[RpcDecisionQuestion],
        force_emulate: bool,
    ) -> anyhow::Result<DecisionBatchResponse> {
        if let Some(res) = self
            .try_daemon_rpc(
                "decide",
                json!({
                    "image_path": image_path.map(|p| p.display().to_string()),
                    "prompt": prompt,
                    "questions": questions,
                    "force_emulate": force_emulate,
                }),
            )
            .await
        {
            return Ok(serde_json::from_value(res?)?);
        }
        self.engine_mut()?
            .decide(image_path, prompt, questions, force_emulate)
            .await
    }

    /// Runs a planner turn against the configured `TurnModelDriver`.
    pub async fn plan(
        &mut self,
        image_path: Option<&Path>,
        system: Option<&str>,
        prompt: &str,
        json_object: bool,
        stream: bool,
    ) -> anyhow::Result<TurnResponse> {
        if let Some(res) = self
            .try_daemon_rpc(
                "plan",
                json!({
                    "image_path": image_path.map(|p| p.display().to_string()),
                    "system": system,
                    "prompt": prompt,
                    "json_object": json_object,
                    "stream": stream,
                }),
            )
            .await
        {
            return Ok(serde_json::from_value(res?)?);
        }
        self.engine_mut()?
            .plan(image_path, system, prompt, json_object, stream)
            .await
    }

    /// Runs a full Observe -> Memory/Lua -> Decide -> Plan -> Koma Action episode.
    pub async fn run_task(
        &mut self,
        workspace: Option<&str>,
        goal: &str,
        dry_run: bool,
        force_emulate: bool,
    ) -> anyhow::Result<RunTaskReport> {
        if let Some(res) = self
            .try_daemon_rpc(
                "run_task",
                json!({
                    "workspace": workspace,
                    "goal": goal,
                    "dry_run": dry_run,
                    "force_emulate": force_emulate,
                }),
            )
            .await
        {
            return Ok(serde_json::from_value(res?)?);
        }
        self.engine_mut()?
            .run_task(workspace, goal, dry_run, force_emulate)
            .await
    }

    /// Runs multimodal `DecisionsDriver` triage over a dropbox directory.
    pub async fn box_classify(
        &mut self,
        dir: Option<&Path>,
        force_emulate: bool,
    ) -> anyhow::Result<JikuBoxReport> {
        if let Some(res) = self
            .try_daemon_rpc(
                "box.classify",
                json!({
                    "directory": dir.map(|p| p.display().to_string()),
                    "force_emulate": force_emulate,
                }),
            )
            .await
        {
            return Ok(serde_json::from_value(res?)?);
        }
        self.engine_mut()?.box_classify(dir, force_emulate).await
    }

    /// Searches historical L2 episodes using SQLite FTS5 BM25 ranking.
    pub async fn memory_search(
        &mut self,
        query: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<EpisodeRecord>> {
        if let Some(res) = self
            .try_daemon_rpc(
                "memory.search",
                json!({
                    "query": query,
                    "limit": limit,
                }),
            )
            .await
        {
            return Ok(serde_json::from_value(res?)?);
        }
        self.engine_mut()?.memory_search(query, limit)
    }

    /// Lists staged, approved, and rejected skills.
    pub async fn skills_list(&mut self) -> anyhow::Result<Vec<StagedSkillRecord>> {
        if let Some(res) = self.try_daemon_rpc("skills.list", json!({})).await {
            return Ok(serde_json::from_value(res?)?);
        }
        self.engine_mut()?.skills_list()
    }

    /// Produces a unified line diff for a staged skill candidate.
    pub async fn skills_diff(&mut self, skill_id: &str) -> anyhow::Result<String> {
        if let Some(res) = self
            .try_daemon_rpc("skills.diff", json!({ "skill_id": skill_id }))
            .await
        {
            let val = res?;
            return Ok(val
                .get("diff")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string());
        }
        self.engine_mut()?.skills_diff(skill_id)
    }

    /// Stages a sanitized Lua or Markdown SOP skill candidate for review.
    pub async fn skills_stage(
        &mut self,
        skill_id: &str,
        title: &str,
        body: &str,
    ) -> anyhow::Result<StagedSkillRecord> {
        if let Some(res) = self
            .try_daemon_rpc(
                "skills.stage",
                json!({
                    "skill_id": skill_id,
                    "title": title,
                    "body": body,
                }),
            )
            .await
        {
            return Ok(serde_json::from_value(res?)?);
        }
        self.engine_mut()?.skills_stage(skill_id, title, body)
    }

    /// Approves a staged skill candidate and optionally promotes it into `promote_dir`.
    pub async fn skills_approve(
        &mut self,
        skill_id: &str,
        promote_dir: Option<&Path>,
    ) -> anyhow::Result<(StagedSkillRecord, Option<PathBuf>)> {
        if let Some(res) = self
            .try_daemon_rpc(
                "skills.approve",
                json!({
                    "skill_id": skill_id,
                    "promote_dir": promote_dir.map(|p| p.display().to_string()),
                }),
            )
            .await
        {
            let val = res?;
            let record: StagedSkillRecord = serde_json::from_value(
                val.get("record")
                    .cloned()
                    .ok_or_else(|| anyhow!("missing `record` in skills.approve response"))?,
            )?;
            let promoted = val
                .get("promoted_path")
                .and_then(Value::as_str)
                .map(PathBuf::from);
            return Ok((record, promoted));
        }
        self.engine_mut()?.skills_approve(skill_id, promote_dir)
    }
}
