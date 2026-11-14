//! CLI controller and MCP bridge binary (`jiku`) for `jikud`.

use std::{
    env,
    io::{self, BufReader, BufWriter},
    path::PathBuf,
    process,
};

use jiku::client::{JikuClient, JikuClientOptions};
use jiku::mcp::{run_mcp_loop, tool_definitions};
use jikud::{config::JikuConfig, daemon::RpcDecisionQuestion, lua_policy::JikuLuaPolicy};
use serde_json::json;

const VERSION: &str = env!("CARGO_PKG_VERSION");

fn print_usage() {
    println!(
        "jiku {VERSION} — Jiku desktop agent CLI & MCP controller\n\n\
USAGE:\n    \
    jiku [GLOBAL OPTIONS] <COMMAND> [ARGS]\n\n\
GLOBAL OPTIONS:\n    \
    --config <PATH>         Path to config.toml\n    \
    --policy <PATH>         Path to init.lua\n    \
    --socket <PATH>         Override jikud Unix socket path\n    \
    --json                  Emit machine-readable JSON output\n    \
    -V, --version           Print version information\n    \
    -h, --help              Print help\n\n\
COMMANDS:\n    \
    status                  Show daemon, model driver, memory DB, and Koma status\n    \
    validate-config         Validate config.toml and init.lua offline\n    \
    decide                  Evaluate calibrated Noul/Choice questions (gemini-decision-1)\n    \
    plan                    Run a TurnModelDriver call (OpenCode Go deepseek-v4.1-flash)\n    \
    run                     Execute Observe -> Decide -> Plan -> Koma Action task loop\n    \
    box classify [<DIR>]    Triage files in directory (default ~/JikuBox)\n    \
    memory search <QUERY>   Search L2 episodes with SQLite FTS5 BM25 ranking\n    \
    skills <SUBCMD>         Manage staged skills (list | diff | stage | approve)\n    \
    mcp                     Run stdio JSON-RPC 2.0 MCP server (--stateless | --print-tools)"
    );
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
            run_mcp_loop(&mut client, stdin, stdout, stateless).await?;
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
        other => Err(CmdError::Usage(format!("unknown command `{other}`"))),
    }
}
