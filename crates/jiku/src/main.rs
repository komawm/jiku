#![forbid(unsafe_code)]

//! CLI controller and MCP bridge binary (`jiku`) for `jikud`.

use std::{
    env,
    io::{self, BufReader, BufWriter},
    path::PathBuf,
    process,
};

use jiku::{
    avatar_stage::{
        AvatarStageConfig, AvatarStageSnapshot, DEFAULT_AVATAR_STAGE_ADDR, serve_avatar_stage,
    },
    client::{JikuClient, JikuClientOptions},
    mcp::{run_mcp_stdio, tool_definitions},
};
mod completions;

use jikud::{config::JikuConfig, daemon::RpcDecisionQuestion, lua_policy::JikuLuaPolicy};
use serde_json::{Value, json};
use tokio::sync::{broadcast, oneshot};

const VERSION: &str = env!("CARGO_PKG_VERSION");

fn usage_text() -> String {
    format!(
        "jiku {VERSION} — Jiku desktop agent CLI & MCP controller\n\n\
USAGE:\n    \
    jiku [GLOBAL OPTIONS] <COMMAND> [ARGS]\n\n\
GLOBAL OPTIONS:\n    \
    --config <PATH>         Path to config.toml or config.lua\n    \
    --policy <PATH>         Path to init.lua\n    \
    --socket <PATH>         Override jikud Unix socket path\n    \
    --json                  Emit machine-readable JSON output\n    \
    -V, --version           Print version information\n    \
    -h, --help              Print help\n\n\
COMMANDS:\n    \
    status                  Show daemon, model driver, memory DB, and Koma status\n    \
    validate-config         Validate config.toml/config.lua and init.lua offline\n    \
    migrate-config          Emit an equivalent config.lua (--out <PATH> [--force])\n    \
    decide                  Evaluate calibrated Noul/Choice questions (gemini-decision-1)\n    \
    plan                    Run a TurnModelDriver call (OpenCode Go deepseek-v4.1-flash)\n    \
    run                     Execute Observe -> Decide -> Plan -> Koma Action task loop\n    \
    box classify [<DIR>]    Triage files in directory (default ~/JikuBox)\n    \
    memory search <QUERY>   Search L2 episodes with SQLite FTS5 BM25 ranking\n    \
    skills <SUBCMD>         Manage staged skills (list | diff | stage | approve)\n    \
    tts <say|stop>          Synthesize/play speech or trigger immediate barge-in stop\n    \
    avatar serve            Serve companion HTTP + WebSocket Live2D/VRM stage\n    \
    a11y <snapshot|act>     Inspect normalized AT-SPI tree or execute semantic action\n    \
    context inspect         Inspect context budget and progressive waterfall compaction\n    \
    bench run               Run 2x2 Decision x Grounding VirtualPaint ablation benchmark\n    \
    mcp                     Run stdio JSON-RPC 2.0 MCP server (--stateless | --print-tools)\n    \
    completions <bash|zsh|fish>   Print a shell completion script and exit"
    )
}

fn print_usage() {
    println!("{}", usage_text());
}

#[derive(Debug, Default)]
struct GlobalOpts {
    config: Option<PathBuf>,
    policy: Option<PathBuf>,
    socket: Option<PathBuf>,
    json: bool,
}

fn extract_global_opts(raw: Vec<String>) -> Result<(GlobalOpts, Vec<String>), i32> {
    let mut globals = GlobalOpts::default();
    let mut rest = Vec::new();
    let mut it = raw.into_iter();

    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--json" => globals.json = true,
            "--config" => {
                let Some(val) = it.next() else {
                    eprintln!("error: --config requires a value");
                    return Err(2);
                };
                globals.config = Some(PathBuf::from(val));
            }
            "--policy" => {
                let Some(val) = it.next() else {
                    eprintln!("error: --policy requires a value");
                    return Err(2);
                };
                globals.policy = Some(PathBuf::from(val));
            }
            "--socket" => {
                let Some(val) = it.next() else {
                    eprintln!("error: --socket requires a value");
                    return Err(2);
                };
                globals.socket = Some(PathBuf::from(val));
            }
            s if s.starts_with("--config=") => {
                globals.config = Some(PathBuf::from(s.trim_start_matches("--config=")));
            }
            s if s.starts_with("--policy=") => {
                globals.policy = Some(PathBuf::from(s.trim_start_matches("--policy=")));
            }
            s if s.starts_with("--socket=") => {
                globals.socket = Some(PathBuf::from(s.trim_start_matches("--socket=")));
            }
            _ => rest.push(arg),
        }
    }

    Ok((globals, rest))
}

fn parse_noul_flag(raw: &str) -> anyhow::Result<RpcDecisionQuestion> {
    let (id, rest) = raw
        .split_once('=')
        .ok_or_else(|| anyhow::anyhow!("expected `id=noul:instructions`, got `{raw}`"))?;
    let body = rest.strip_prefix("noul:").unwrap_or(rest);
    if id.trim().is_empty() || body.trim().is_empty() {
        anyhow::bail!("question id and instructions must be non-empty");
    }
    Ok(RpcDecisionQuestion::Noul {
        id: id.trim().to_string(),
        instructions: body.trim().to_string(),
    })
}

fn parse_choice_flag(raw: &str) -> anyhow::Result<RpcDecisionQuestion> {
    let (id, rest) = raw
        .split_once('=')
        .ok_or_else(|| anyhow::anyhow!("expected `id=c1,c2:instructions`, got `{raw}`"))?;
    let (choices_str, instructions) = rest
        .split_once(':')
        .ok_or_else(|| anyhow::anyhow!("expected `c1,c2:instructions` in `{raw}`"))?;
    let choices: Vec<String> = choices_str
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if id.trim().is_empty() || choices.is_empty() || instructions.trim().is_empty() {
        anyhow::bail!("choice id, candidates, and instructions must be non-empty");
    }
    Ok(RpcDecisionQuestion::Choice {
        id: id.trim().to_string(),
        instructions: instructions.trim().to_string(),
        choices,
        scope_gate_question_id: None,
    })
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let raw_args: Vec<String> = env::args().skip(1).collect();
    if raw_args.is_empty() || raw_args.iter().any(|a| a == "-h" || a == "--help") {
        print_usage();
        return;
    }
    if raw_args.iter().any(|a| a == "-V" || a == "--version") {
        println!("jiku {VERSION}");
        return;
    }

    let (globals, rest) = match extract_global_opts(raw_args) {
        Ok(pair) => pair,
        Err(code) => process::exit(code),
    };
    let Some((subcmd, tail)) = rest.split_first() else {
        print_usage();
        return;
    };

    if subcmd.starts_with('-') {
        eprintln!("error: unknown option `{subcmd}`\n");
        print_usage();
        process::exit(2);
    }

    if let Err(err) = dispatch(globals, subcmd, tail).await {
        match err {
            CmdError::Usage(msg) => {
                eprintln!("error: {msg}");
                process::exit(2);
            }
            CmdError::Runtime(e) => {
                eprintln!("error: {e:#}");
                process::exit(1);
            }
        }
    }
}

enum CmdError {
    Usage(String),
    Runtime(anyhow::Error),
}

impl From<anyhow::Error> for CmdError {
    fn from(value: anyhow::Error) -> Self {
        Self::Runtime(value)
    }
}

async fn dispatch(globals: GlobalOpts, subcmd: &str, tail: &[String]) -> Result<(), CmdError> {
    if subcmd == "validate-config" {
        if !tail.is_empty() {
            return Err(CmdError::Usage(format!(
                "unexpected arguments for validate-config: {}",
                tail.join(" ")
            )));
        }
        let (cfg, cfg_path) =
            JikuConfig::load_resolved(globals.config.as_deref()).map_err(|e| anyhow::anyhow!(e))?;
        let (lua, pol_path) = JikuLuaPolicy::load_resolved(globals.policy.as_deref())
            .map_err(|e| anyhow::anyhow!(e))?;
        let (planner_provider, planner_model, planner_role) = cfg.planner_binding();
        let thinking = cfg
            .providers
            .get(&planner_provider)
            .is_some_and(|p| cfg.resolve_thinking_enabled(planner_role, p));
        let val = json!({
            "valid": true,
            "config_path": cfg_path.map(|p| p.display().to_string()),
            "policy_path": pol_path.map(|p| p.display().to_string()),
            "planner_provider": planner_provider,
            "planner_model": planner_model,
            "planner_thinking_enabled": thinking,
            "decision_model": cfg.decision_router.decision_model_id,
            "lua_hook_counts": lua.hook_counts(),
        });
        if globals.json {
            println!("{}", serde_json::to_string_pretty(&val).unwrap_or_default());
        } else {
            println!(
                "valid=true planner={planner_provider}/{planner_model} thinking={thinking} decision_model={}",
                cfg.decision_router
                    .decision_model_id
                    .as_deref()
                    .unwrap_or("emulated")
            );
        }
        return Ok(());
    }

    if subcmd == "migrate-config" {
        let mut out: Option<PathBuf> = None;
        let mut force = false;
        let mut it = tail.iter();
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--force" => force = true,
                "--out" => {
                    let Some(val) = it.next() else {
                        return Err(CmdError::Usage(
                            "--out requires a path argument".to_string(),
                        ));
                    };
                    out = Some(PathBuf::from(val));
                }
                other if other.starts_with("--out=") => {
                    out = Some(PathBuf::from(other.trim_start_matches("--out=")));
                }
                other => {
                    return Err(CmdError::Usage(format!(
                        "unknown migrate-config option `{other}`"
                    )));
                }
            }
        }
        let (cfg, _) =
            JikuConfig::load_resolved(globals.config.as_deref()).map_err(|e| anyhow::anyhow!(e))?;
        match out {
            Some(path) => {
                let written = jikud::migrate::migrate_to_path(&cfg, &path, force)
                    .map_err(|e| anyhow::anyhow!(e))?;
                println!("wrote {}", written.display());
                if let Some(toml) = jikud::migrate::shadowed_toml(&written) {
                    eprintln!(
                        "note: {} is still loaded first and fully overlaid by {}; remove or rename it to avoid editing dead configuration",
                        toml.display(),
                        written.display()
                    );
                }
            }
            None => {
                let rendered = jikud::migrate::render_checked_lua_config(&cfg)
                    .map_err(|e| anyhow::anyhow!(e))?;
                print!("{rendered}");
            }
        }
        return Ok(());
    }

    if subcmd == "completions" {
        let Some(name) = tail.first() else {
            eprintln!("usage: jiku completions <bash|zsh|fish>");
            std::process::exit(2);
        };
        let Some(shell) = completions::Shell::parse(name) else {
            eprintln!(
                "unknown shell `{name}` (expected {}, {}, or {})",
                completions::Shell::Bash.name(),
                completions::Shell::Zsh.name(),
                completions::Shell::Fish.name()
            );
            std::process::exit(2);
        };
        print!("{}", completions::render(shell));
        return Ok(());
    }

    let opts = JikuClientOptions {
        config_path: globals.config,
        policy_path: globals.policy,
        socket_path: globals.socket,
        in_memory_db: false,
    };

    match subcmd {
        "mcp" => {
            let mut stateless = false;
            let mut print_tools = false;
            for arg in tail {
                match arg.as_str() {
                    "--stateless" => stateless = true,
                    "--print-tools" => print_tools = true,
                    other => {
                        return Err(CmdError::Usage(format!("unknown mcp flag `{other}`")));
                    }
                }
            }
            if print_tools {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&tool_definitions()).unwrap_or_default()
                );
                return Ok(());
            }
            let mut client = JikuClient::new(opts)?;
            let stdin = BufReader::new(io::stdin().lock());
            let stdout = BufWriter::new(io::stdout().lock());
            run_mcp_stdio(&mut client, stdin, stdout, stateless).await?;
            Ok(())
        }
        "status" => {
            if !tail.is_empty() {
                return Err(CmdError::Usage(format!(
                    "unexpected arguments for status: {}",
                    tail.join(" ")
                )));
            }
            let mut client = JikuClient::new(opts)?;
            let status = client.status().await?;
            if globals.json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&status).unwrap_or_default()
                );
            } else {
                println!("version: {}", status.version);
                println!("socket: {}", status.daemon_socket_path);
                println!(
                    "planner: {}/{} (thinking={})",
                    status.planner_provider, status.planner_model, status.planner_thinking_enabled
                );
                println!(
                    "decision_router: {} ({})",
                    status.decision_mode,
                    status.decision_model_id.as_deref().unwrap_or("emulated")
                );
                println!(
                    "memory: {} (episodes={}, skills={})",
                    status.memory_db_path, status.episode_count, status.skill_count
                );
                println!(
                    "koma: connected={} workspaces={}",
                    status.koma_connected,
                    status.koma_workspaces.len()
                );
            }
            Ok(())
        }
        "decide" => {
            let mut image: Option<PathBuf> = None;
            let mut emulate = false;
            let mut questions: Vec<RpcDecisionQuestion> = Vec::new();
            let mut prompt_parts: Vec<String> = Vec::new();
            let mut it = tail.iter().cloned();
            while let Some(arg) = it.next() {
                match arg.as_str() {
                    "--image" => {
                        let Some(val) = it.next() else {
                            return Err(CmdError::Usage("--image requires a path".to_string()));
                        };
                        image = Some(PathBuf::from(val));
                    }
                    "--emulate" => emulate = true,
                    "--question" => {
                        let Some(val) = it.next() else {
                            return Err(CmdError::Usage("--question requires spec".to_string()));
                        };
                        questions.push(parse_noul_flag(&val)?);
                    }
                    "--choice" => {
                        let Some(val) = it.next() else {
                            return Err(CmdError::Usage("--choice requires spec".to_string()));
                        };
                        questions.push(parse_choice_flag(&val)?);
                    }
                    other if other.starts_with('-') => {
                        return Err(CmdError::Usage(format!("unknown decide option `{other}`")));
                    }
                    _ => prompt_parts.push(arg),
                }
            }
            let prompt = prompt_parts.join(" ");
            if prompt.trim().is_empty() && questions.is_empty() {
                return Err(CmdError::Usage(
                    "decide requires a prompt or --question / --choice".to_string(),
                ));
            }
            let mut client = JikuClient::new(opts)?;
            let resp = client
                .decide(image.as_deref(), &prompt, &questions, emulate)
                .await?;
            if globals.json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&resp).unwrap_or_default()
                );
            } else {
                println!(
                    "mode={} model={} tokens=({} in / {} out)",
                    resp.execution_mode,
                    resp.model,
                    resp.usage.total_input_tokens,
                    resp.usage.total_output_tokens
                );
                for (qid, ans) in &resp.answers {
                    println!("  {qid}: {ans:?}");
                }
            }
            Ok(())
        }
        "plan" => {
            let mut image: Option<PathBuf> = None;
            let mut system: Option<String> = None;
            let mut json_object = false;
            let mut stream = false;
            let mut prompt_parts: Vec<String> = Vec::new();
            let mut it = tail.iter().cloned();
            while let Some(arg) = it.next() {
                match arg.as_str() {
                    "--image" => {
                        let Some(val) = it.next() else {
                            return Err(CmdError::Usage("--image requires a path".to_string()));
                        };
                        image = Some(PathBuf::from(val));
                    }
                    "--system" => {
                        let Some(val) = it.next() else {
                            return Err(CmdError::Usage("--system requires text".to_string()));
                        };
                        system = Some(val);
                    }
                    "--json-object" => json_object = true,
                    "--stream" => stream = true,
                    other if other.starts_with('-') => {
                        return Err(CmdError::Usage(format!("unknown plan option `{other}`")));
                    }
                    _ => prompt_parts.push(arg),
                }
            }
            let prompt = prompt_parts.join(" ");
            if prompt.trim().is_empty() {
                return Err(CmdError::Usage("plan requires a prompt".to_string()));
            }
            let mut client = JikuClient::new(opts)?;
            let resp = client
                .plan(
                    image.as_deref(),
                    system.as_deref(),
                    &prompt,
                    json_object,
                    stream,
                )
                .await?;
            if globals.json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&resp).unwrap_or_default()
                );
            } else {
                println!("{}", resp.text);
            }
            Ok(())
        }
        "run" => {
            let mut workspace: Option<String> = None;
            let mut dry_run = false;
            let mut emulate = false;
            let mut goal_parts: Vec<String> = Vec::new();
            let mut it = tail.iter().cloned();
            while let Some(arg) = it.next() {
                match arg.as_str() {
                    "--workspace" => {
                        let Some(val) = it.next() else {
                            return Err(CmdError::Usage("--workspace requires an ID".to_string()));
                        };
                        workspace = Some(val);
                    }
                    "--dry-run" => dry_run = true,
                    "--emulate" => emulate = true,
                    other if other.starts_with('-') => {
                        return Err(CmdError::Usage(format!("unknown run option `{other}`")));
                    }
                    _ => goal_parts.push(arg),
                }
            }
            let goal = goal_parts.join(" ");
            if goal.trim().is_empty() {
                return Err(CmdError::Usage("run requires a goal".to_string()));
            }
            let mut client = JikuClient::new(opts)?;
            let report = client
                .run_task(workspace.as_deref(), &goal, dry_run, emulate)
                .await?;
            if globals.json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&report).unwrap_or_default()
                );
            } else {
                println!(
                    "episode={} route={} koma={} steps={} duration={}ms",
                    report.episode_id,
                    report.route,
                    report.koma_connected,
                    report.executed_steps.len(),
                    report.duration_ms
                );
                println!("summary: {}", report.planner_summary);
            }
            Ok(())
        }
        "box" => {
            let Some((action, box_tail)) = tail.split_first() else {
                return Err(CmdError::Usage(
                    "expected `jiku box classify [<DIR>]`".to_string(),
                ));
            };
            if action != "classify" {
                return Err(CmdError::Usage(format!(
                    "unknown box subcommand `{action}`"
                )));
            }
            let mut emulate = false;
            let mut dir: Option<PathBuf> = None;
            for arg in box_tail {
                if arg == "--emulate" {
                    emulate = true;
                } else if arg.starts_with('-') {
                    return Err(CmdError::Usage(format!(
                        "unknown box classify option `{arg}`"
                    )));
                } else {
                    dir = Some(PathBuf::from(arg));
                }
            }
            let mut client = JikuClient::new(opts)?;
            let report = client.box_classify(dir.as_deref(), emulate).await?;
            if globals.json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&report).unwrap_or_default()
                );
            } else {
                print!("{}", report.to_markdown());
            }
            Ok(())
        }
        "memory" => {
            let Some((action, mem_tail)) = tail.split_first() else {
                return Err(CmdError::Usage(
                    "expected `jiku memory search [--limit N] <QUERY>`".to_string(),
                ));
            };
            if action != "search" {
                return Err(CmdError::Usage(format!(
                    "unknown memory subcommand `{action}`"
                )));
            }
            let mut limit = 10_usize;
            let mut query_parts = Vec::new();
            let mut it = mem_tail.iter().cloned();
            while let Some(arg) = it.next() {
                if arg == "--limit" {
                    let Some(val) = it.next() else {
                        return Err(CmdError::Usage("--limit requires a number".to_string()));
                    };
                    limit = val
                        .parse::<usize>()
                        .map_err(|_| CmdError::Usage(format!("invalid --limit `{val}`")))?;
                } else if arg.starts_with('-') {
                    return Err(CmdError::Usage(format!(
                        "unknown memory search option `{arg}`"
                    )));
                } else {
                    query_parts.push(arg);
                }
            }
            let query = query_parts.join(" ");
            if query.trim().is_empty() {
                return Err(CmdError::Usage(
                    "memory search requires a query".to_string(),
                ));
            }
            let mut client = JikuClient::new(opts)?;
            let hits = client.memory_search(&query, limit).await?;
            if globals.json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&hits).unwrap_or_default()
                );
            } else if hits.is_empty() {
                println!("no episodes matched `{query}`");
            } else {
                for ep in hits {
                    println!(
                        "#{} [{} | {}] {} -> {}",
                        ep.id, ep.workspace, ep.route, ep.goal, ep.summary
                    );
                }
            }
            Ok(())
        }
        "skills" => {
            let Some((action, sk_tail)) = tail.split_first() else {
                return Err(CmdError::Usage(
                    "expected `jiku skills <list|diff|stage|approve>`".to_string(),
                ));
            };
            let mut client = JikuClient::new(opts)?;
            match action.as_str() {
                "list" => {
                    if !sk_tail.is_empty() {
                        return Err(CmdError::Usage(
                            "skills list takes no arguments".to_string(),
                        ));
                    }
                    let skills = client.skills_list().await?;
                    if globals.json {
                        println!(
                            "{}",
                            serde_json::to_string_pretty(&skills).unwrap_or_default()
                        );
                    } else if skills.is_empty() {
                        println!("no skills staged");
                    } else {
                        for sk in skills {
                            println!("{} [{}] {}", sk.skill_id, sk.status.as_str(), sk.title);
                        }
                    }
                }
                "diff" => {
                    let Some(id) = sk_tail.first().filter(|_| sk_tail.len() == 1) else {
                        return Err(CmdError::Usage(
                            "usage: jiku skills diff <skill_id>".to_string(),
                        ));
                    };
                    let diff = client.skills_diff(id).await?;
                    if globals.json {
                        println!(
                            "{}",
                            serde_json::to_string_pretty(&json!({ "skill_id": id, "diff": diff }))
                                .unwrap_or_default()
                        );
                    } else {
                        print!("{diff}");
                    }
                }
                "stage" => {
                    if sk_tail.len() != 3 {
                        return Err(CmdError::Usage(
                            "usage: jiku skills stage <skill_id> <title> <body>".to_string(),
                        ));
                    }
                    let rec = client
                        .skills_stage(&sk_tail[0], &sk_tail[1], &sk_tail[2])
                        .await?;
                    if globals.json {
                        println!("{}", serde_json::to_string_pretty(&rec).unwrap_or_default());
                    } else {
                        println!("staged {} ({})", rec.skill_id, rec.status.as_str());
                    }
                }
                "approve" => {
                    let mut id: Option<String> = None;
                    let mut promote_dir: Option<PathBuf> = None;
                    let mut it = sk_tail.iter().cloned();
                    while let Some(arg) = it.next() {
                        if arg == "--promote-dir" {
                            let Some(val) = it.next() else {
                                return Err(CmdError::Usage(
                                    "--promote-dir requires a path".to_string(),
                                ));
                            };
                            promote_dir = Some(PathBuf::from(val));
                        } else if arg.starts_with('-') {
                            return Err(CmdError::Usage(format!(
                                "unknown skills approve option `{arg}`"
                            )));
                        } else if id.is_none() {
                            id = Some(arg);
                        } else {
                            return Err(CmdError::Usage(
                                "usage: jiku skills approve <skill_id> [--promote-dir <dir>]"
                                    .to_string(),
                            ));
                        }
                    }
                    let Some(skill_id) = id else {
                        return Err(CmdError::Usage(
                            "usage: jiku skills approve <skill_id> [--promote-dir <dir>]"
                                .to_string(),
                        ));
                    };
                    let (rec, promoted) = client
                        .skills_approve(&skill_id, promote_dir.as_deref())
                        .await?;
                    if globals.json {
                        println!(
                            "{}",
                            serde_json::to_string_pretty(&json!({
                                "record": rec,
                                "promoted_path": promoted.map(|p| p.display().to_string()),
                            }))
                            .unwrap_or_default()
                        );
                    } else {
                        println!("approved {} ({})", rec.skill_id, rec.status.as_str());
                        if let Some(p) = promoted {
                            println!("promoted_path: {}", p.display());
                        }
                    }
                }
                other => {
                    return Err(CmdError::Usage(format!(
                        "unknown skills subcommand `{other}`"
                    )));
                }
            }
            Ok(())
        }
        "tts" => {
            let Some((action, tts_tail)) = tail.split_first() else {
                return Err(CmdError::Usage(
                    "usage: jiku tts <say <text> [--emotion <e>] [--no-play] [--wav-out <PATH>] | stop>"
                        .to_string(),
                ));
            };
            let mut client = JikuClient::new(opts)?;
            match action.as_str() {
                "say" => {
                    let mut emotion: Option<String> = None;
                    let mut play = true;
                    let mut wav_out: Option<PathBuf> = None;
                    let mut text_parts = Vec::new();
                    let mut it = tts_tail.iter().cloned();
                    while let Some(arg) = it.next() {
                        match arg.as_str() {
                            "--emotion" => {
                                emotion = Some(it.next().ok_or_else(|| {
                                    CmdError::Usage("--emotion requires a value".to_string())
                                })?);
                            }
                            "--no-play" => play = false,
                            "--wav-out" => {
                                let val = it.next().ok_or_else(|| {
                                    CmdError::Usage("--wav-out requires a file path".to_string())
                                })?;
                                wav_out = Some(PathBuf::from(val));
                            }
                            s if s.starts_with('-') => {
                                return Err(CmdError::Usage(format!(
                                    "unknown tts say option `{s}`"
                                )));
                            }
                            _ => text_parts.push(arg),
                        }
                    }
                    let text = text_parts.join(" ");
                    if text.trim().is_empty() {
                        return Err(CmdError::Usage(
                            "usage: jiku tts say <text> [--emotion <e>] [--no-play] [--wav-out <PATH>]"
                                .to_string(),
                        ));
                    }
                    let val = client
                        .tts_say(&text, emotion.as_deref(), play, wav_out.as_deref())
                        .await?;
                    if globals.json {
                        println!("{}", serde_json::to_string_pretty(&val).unwrap_or_default());
                    } else {
                        println!(
                            "provider={} duration_ms={} bytes={} played={}",
                            val["provider"].as_str().unwrap_or("unknown"),
                            val["duration_ms"].as_u64().unwrap_or(0),
                            val["wav_bytes_len"].as_u64().unwrap_or(0),
                            val["played"].as_bool().unwrap_or(false)
                        );
                    }
                    Ok(())
                }
                "stop" => {
                    if !tts_tail.is_empty() {
                        return Err(CmdError::Usage("usage: jiku tts stop".to_string()));
                    }
                    let val = client.tts_stop().await?;
                    if globals.json {
                        println!("{}", serde_json::to_string_pretty(&val).unwrap_or_default());
                    } else {
                        println!("stopped={}", val["stopped"].as_bool().unwrap_or(false));
                    }
                    Ok(())
                }
                other => Err(CmdError::Usage(format!(
                    "unknown tts subcommand `{other}` (expected `say` or `stop`)"
                ))),
            }
        }
        "avatar" => {
            let Some((action, av_tail)) = tail.split_first() else {
                return Err(CmdError::Usage(
                    "usage: jiku avatar serve [--bind <ADDR>] [--vrm <PATH>]".to_string(),
                ));
            };
            if action != "serve" {
                return Err(CmdError::Usage(format!(
                    "unknown avatar subcommand `{action}` (expected `serve`)"
                )));
            }
            let mut bind_addr = DEFAULT_AVATAR_STAGE_ADDR.to_string();
            let mut vrm_path: Option<PathBuf> = None;
            let mut it = av_tail.iter().cloned();
            while let Some(arg) = it.next() {
                match arg.as_str() {
                    "--bind" => {
                        bind_addr = it.next().ok_or_else(|| {
                            CmdError::Usage("--bind requires `<host>:<port>`".to_string())
                        })?;
                    }
                    "--vrm" => {
                        let p = it.next().ok_or_else(|| {
                            CmdError::Usage("--vrm requires a file path".to_string())
                        })?;
                        vrm_path = Some(PathBuf::from(p));
                    }
                    other => {
                        return Err(CmdError::Usage(format!(
                            "unknown avatar serve option `{other}`"
                        )));
                    }
                }
            }
            let mut client = JikuClient::new(opts)?;
            let mut snapshot = AvatarStageSnapshot::default();
            if let Ok(state) = client.avatar_state().await
                && let Some(cue_val) = state.get("cue").filter(|v| !v.is_null())
                && let Ok(cue) = serde_json::from_value(cue_val.clone())
            {
                snapshot.cue = cue;
            }
            let cfg = AvatarStageConfig {
                bind_addr,
                vrm_path,
                companion_name: "Jiku".to_string(),
            };
            let (tx, _rx) = broadcast::channel(64);
            let (ready_tx, ready_rx) = oneshot::channel();
            let emit_json = globals.json;
            tokio::spawn(async move {
                if let Ok(addr) = ready_rx.await {
                    if emit_json {
                        println!(
                            "{}",
                            serde_json::to_string_pretty(&json!({
                                "http_url": format!("http://{addr}/"),
                                "ws_url": format!("ws://{addr}/ws"),
                            }))
                            .unwrap_or_default()
                        );
                    } else {
                        println!("avatar stage listening on http://{addr}/ (ws://{addr}/ws)");
                    }
                }
            });
            serve_avatar_stage(cfg, snapshot, tx, Some(ready_tx), async {
                let _ = tokio::signal::ctrl_c().await;
            })
            .await?;
            Ok(())
        }
        "a11y" => {
            let Some((action, a11y_tail)) = tail.split_first() else {
                return Err(CmdError::Usage(
                    "usage: jiku a11y <snapshot [--skeleton] [--root <REF>] | act <click|focus|set-value|type|read-value> <REF> [TEXT]>"
                        .to_string(),
                ));
            };
            let mut client = JikuClient::new(opts)?;
            match action.as_str() {
                "snapshot" => {
                    let mut skeleton = false;
                    let mut root_ref: Option<String> = None;
                    let mut app_id = String::new();
                    let mut window_title = String::new();
                    let mut it = a11y_tail.iter().cloned();
                    while let Some(arg) = it.next() {
                        match arg.as_str() {
                            "--skeleton" => skeleton = true,
                            "--root" => {
                                root_ref = Some(it.next().ok_or_else(|| {
                                    CmdError::Usage("--root requires a `@snap:eN` ref".to_string())
                                })?);
                            }
                            "--app-id" => {
                                app_id = it.next().ok_or_else(|| {
                                    CmdError::Usage("--app-id requires a value".to_string())
                                })?;
                            }
                            "--title" => {
                                window_title = it.next().ok_or_else(|| {
                                    CmdError::Usage("--title requires a value".to_string())
                                })?;
                            }
                            other => {
                                return Err(CmdError::Usage(format!(
                                    "unknown a11y snapshot option `{other}`"
                                )));
                            }
                        }
                    }
                    let val = client
                        .a11y_snapshot(skeleton, root_ref.as_deref(), &app_id, &window_title)
                        .await?;
                    if globals.json {
                        println!("{}", serde_json::to_string_pretty(&val).unwrap_or_default());
                    } else {
                        println!("{}", val["text_tree"].as_str().unwrap_or(""));
                    }
                    Ok(())
                }
                "act" => {
                    let Some((act_kind, rest_args)) = a11y_tail.split_first() else {
                        return Err(CmdError::Usage(
                            "usage: jiku a11y act <click|focus|set-value|type|read-value> <REF> [TEXT]"
                                .to_string(),
                        ));
                    };
                    let Some((target_ref, text_args)) = rest_args.split_first() else {
                        return Err(CmdError::Usage(
                            "usage: jiku a11y act <click|focus|set-value|type|read-value> <REF> [TEXT]"
                                .to_string(),
                        ));
                    };
                    let text_val = if text_args.is_empty() {
                        None
                    } else {
                        Some(text_args.join(" "))
                    };
                    let val = client
                        .a11y_act(act_kind, target_ref, text_val.as_deref())
                        .await?;
                    if globals.json {
                        println!("{}", serde_json::to_string_pretty(&val).unwrap_or_default());
                    } else {
                        println!(
                            "ok={} ref={} action={} detail={}",
                            val["ok"].as_bool().unwrap_or(false),
                            val["ref_id"].as_str().unwrap_or(target_ref),
                            val["action"].as_str().unwrap_or(act_kind),
                            val["value"]
                                .as_str()
                                .or_else(|| val["message"].as_str())
                                .unwrap_or("ok")
                        );
                    }
                    Ok(())
                }
                other => Err(CmdError::Usage(format!(
                    "unknown a11y subcommand `{other}` (expected `snapshot` or `act`)"
                ))),
            }
        }
        "context" => {
            let Some((action, ctx_tail)) = tail.split_first() else {
                return Err(CmdError::Usage(
                    "usage: jiku context inspect [--max-tokens <N>] [--sample-text <TEXT>]"
                        .to_string(),
                ));
            };
            if action != "inspect" {
                return Err(CmdError::Usage(format!(
                    "unknown context subcommand `{action}` (expected `inspect`)"
                )));
            }
            let mut max_tokens: Option<u32> = None;
            let mut sample_text: Option<String> = None;
            let mut it = ctx_tail.iter().cloned();
            while let Some(arg) = it.next() {
                match arg.as_str() {
                    "--max-tokens" => {
                        let val = it.next().ok_or_else(|| {
                            CmdError::Usage("--max-tokens requires an integer".to_string())
                        })?;
                        max_tokens = Some(val.parse::<u32>().map_err(|_| {
                            CmdError::Usage(format!("invalid --max-tokens `{val}`"))
                        })?);
                    }
                    "--sample-text" => {
                        sample_text = Some(it.next().ok_or_else(|| {
                            CmdError::Usage("--sample-text requires a string".to_string())
                        })?);
                    }
                    other => {
                        return Err(CmdError::Usage(format!(
                            "unknown context inspect option `{other}`"
                        )));
                    }
                }
            }
            let mut client = JikuClient::new(opts)?;
            let val = client
                .context_inspect(max_tokens, sample_text.as_deref())
                .await?;
            if globals.json {
                println!("{}", serde_json::to_string_pretty(&val).unwrap_or_default());
            } else {
                let sim = &val["simulated_report"];
                println!(
                    "max_ctx={} raw_tok={} compacted_tok={} clamped_out={} dropped_imgs={} truncated_a11y={} stages={}",
                    sim["max_context_tokens"].as_u64().unwrap_or(0),
                    sim["raw_prompt_tokens"].as_u64().unwrap_or(0),
                    sim["compacted_prompt_tokens"].as_u64().unwrap_or(0),
                    sim["clamped_max_output_tokens"].as_u64().unwrap_or(0),
                    sim["dropped_images"].as_u64().unwrap_or(0),
                    sim["truncated_a11y_nodes"].as_u64().unwrap_or(0),
                    sim["stages_applied"]
                        .as_array()
                        .map(|arr| {
                            arr.iter()
                                .filter_map(Value::as_str)
                                .collect::<Vec<_>>()
                                .join(",")
                        })
                        .unwrap_or_default()
                );
            }
            Ok(())
        }
        "bench" => {
            let Some((action, bench_tail)) = tail.split_first() else {
                return Err(CmdError::Usage(
                    "usage: jiku bench run [--offline|--live] [--endpoint <URL>] [--model <NAME>] [--provider <P>] [--max-context-tokens <N>] [--out-dir <DIR>]"
                        .to_string(),
                ));
            };
            if action != "run" {
                return Err(CmdError::Usage(format!(
                    "unknown bench subcommand `{action}` (expected `run`)"
                )));
            }
            let mut offline = true;
            let mut endpoint: Option<String> = None;
            let mut model: Option<String> = None;
            let mut provider: Option<String> = None;
            let mut max_context_tokens: Option<u32> = None;
            let mut out_dir: Option<PathBuf> = None;
            let mut it = bench_tail.iter().cloned();
            while let Some(arg) = it.next() {
                match arg.as_str() {
                    "--offline" => offline = true,
                    "--live" => offline = false,
                    "--endpoint" => {
                        endpoint = Some(it.next().ok_or_else(|| {
                            CmdError::Usage("--endpoint requires a URL".to_string())
                        })?);
                    }
                    "--model" => {
                        model = Some(it.next().ok_or_else(|| {
                            CmdError::Usage("--model requires a model name".to_string())
                        })?);
                    }
                    "--provider" => {
                        provider = Some(it.next().ok_or_else(|| {
                            CmdError::Usage("--provider requires a provider key".to_string())
                        })?);
                    }
                    "--max-context-tokens" => {
                        let val = it.next().ok_or_else(|| {
                            CmdError::Usage("--max-context-tokens requires an integer".to_string())
                        })?;
                        max_context_tokens = Some(val.parse::<u32>().map_err(|_| {
                            CmdError::Usage(format!("invalid --max-context-tokens `{val}`"))
                        })?);
                    }
                    "--out-dir" => {
                        let val = it.next().ok_or_else(|| {
                            CmdError::Usage("--out-dir requires a directory path".to_string())
                        })?;
                        out_dir = Some(PathBuf::from(val));
                    }
                    other => {
                        return Err(CmdError::Usage(format!(
                            "unknown bench run option `{other}`"
                        )));
                    }
                }
            }
            let mut client = JikuClient::new(opts)?;
            let val = client
                .bench_run(
                    offline,
                    max_context_tokens,
                    out_dir.as_deref(),
                    provider.as_deref(),
                    model.as_deref(),
                    endpoint.as_deref(),
                )
                .await?;
            if globals.json {
                println!("{}", serde_json::to_string_pretty(&val).unwrap_or_default());
            } else {
                println!("{}", val["markdown"].as_str().unwrap_or(""));
                if let Some(dir) = val["artifacts"]["out_dir"].as_str() {
                    println!("Artifacts exported to: {dir}");
                }
            }
            Ok(())
        }
        other => Err(CmdError::Usage(format!("unknown command `{other}`"))),
    }
}
