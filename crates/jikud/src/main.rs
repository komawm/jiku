#![forbid(unsafe_code)]

//! Executable entrypoint for the `jikud` desktop orchestration daemon.

use std::{env, path::PathBuf, process};

use jiku_memory::{ContextProvider as _, NewEpisode, SqliteMemoryStore};
use jiku_tts::{AvatarEmotion, NoopTtsProvider, TtsProvider as _};
use jikud::{
    ControlPlaneRoute, DecisionCompilerConfig,
    config::JikuConfig,
    daemon::{JikuEngine, run_daemon_server},
    lua_policy::JikuLuaPolicy,
};
use serde_json::json;
use tracing_subscriber::EnvFilter;

const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Debug, Default)]
struct CliArgs {
    daemon: bool,
    validate_config: bool,
    headless_check: bool,
    config: Option<PathBuf>,
    policy: Option<PathBuf>,
    socket: Option<PathBuf>,
    migrate_config: bool,
    out: Option<PathBuf>,
    force: bool,
}

fn print_usage() {
    println!(
        "jikud {VERSION} — Jiku desktop orchestration daemon\n\n\
USAGE:\n    \
    jikud [OPTIONS]\n\n\
OPTIONS:\n    \
    --daemon                Run persistent $XDG_RUNTIME_DIR/jiku.sock RPC daemon (default)\n    \
    --socket <PATH>         Override Unix domain socket path\n    \
    --config <PATH>         Path to config.toml or config.lua (default: $JIKU_CONFIG or ~/.config/jiku/config.toml)\n    \
    --policy <PATH>         Path to init.lua (default: $JIKU_POLICY or ~/.config/jiku/init.lua)\n    \
    --validate-config       Validate configuration and Lua policy, then exit\n    \
    --migrate-config        Emit an equivalent config.lua (stdout, or --out <PATH>)\n    \
    --out <PATH>            Destination for --migrate-config output\n    \
    --force                 Allow --migrate-config to overwrite an existing file\n    \
    --headless-check        Run offline engine, Phodopus sandbox, SQLite FTS5, and compiler checks\n    \
    -V, --version           Print version information\n    \
    -h, --help              Print help"
    );
}

fn parse_args() -> Result<CliArgs, i32> {
    let mut args = env::args().skip(1).peekable();
    let mut out = CliArgs::default();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print_usage();
                return Err(0);
            }
            "-V" | "--version" => {
                println!("jikud {VERSION}");
                return Err(0);
            }
            "--daemon" => out.daemon = true,
            "--validate-config" => out.validate_config = true,
            "--headless-check" => out.headless_check = true,
            "--migrate-config" => out.migrate_config = true,
            "--force" => out.force = true,
            "--out" => {
                let Some(val) = args.next() else {
                    eprintln!("error: --out requires a path argument");
                    return Err(2);
                };
                out.out = Some(PathBuf::from(val));
            }
            "--config" => {
                let Some(val) = args.next() else {
                    eprintln!("error: --config requires a path argument");
                    return Err(2);
                };
                out.config = Some(PathBuf::from(val));
            }
            "--policy" => {
                let Some(val) = args.next() else {
                    eprintln!("error: --policy requires a path argument");
                    return Err(2);
                };
                out.policy = Some(PathBuf::from(val));
            }
            "--socket" => {
                let Some(val) = args.next() else {
                    eprintln!("error: --socket requires a path argument");
                    return Err(2);
                };
                out.socket = Some(PathBuf::from(val));
            }
            other if other.starts_with("--config=") => {
                out.config = Some(PathBuf::from(other.trim_start_matches("--config=")));
            }
            other if other.starts_with("--policy=") => {
                out.policy = Some(PathBuf::from(other.trim_start_matches("--policy=")));
            }
            other if other.starts_with("--socket=") => {
                out.socket = Some(PathBuf::from(other.trim_start_matches("--socket=")));
            }
            other if other.starts_with("--out=") => {
                out.out = Some(PathBuf::from(other.trim_start_matches("--out=")));
            }
            unknown => {
                eprintln!("error: unknown argument `{unknown}`\n");
                print_usage();
                return Err(2);
            }
        }
    }

    Ok(out)
}

async fn run_headless_check(args: &CliArgs) -> anyhow::Result<()> {
    let engine = JikuEngine::from_resolved(args.config.as_deref(), args.policy.as_deref(), true)?;

    // Exercise in-memory SQLite FTS5 roundtrip.
    let mem = SqliteMemoryStore::open_in_memory(engine.config().memory.app_quirk_max_chars)?;
    mem.record_episode(&NewEpisode {
        workspace: engine.config().koma.default_workspace.clone(),
        goal: "headless self check".to_string(),
        route: "fast_path_koma".to_string(),
        summary: "verified FTS5 episode indexing".to_string(),
        actions_json: "[]".to_string(),
        confidence: 0.99,
        duration_ms: 1,
    })?;
    let hits = mem.search_episodes("headless", 5)?;
    anyhow::ensure!(!hits.is_empty(), "expected SQLite FTS5 search hit");

    // Exercise DecisionCompilerConfig deterministic routing.
    let compiler = DecisionCompilerConfig::default();
    let route = compiler.route(0.95, "click_target", 0.92, false);
    anyhow::ensure!(
        matches!(route, ControlPlaneRoute::FastPathKoma { .. }),
        "expected FastPathKoma route in headless self-check"
    );

    // Exercise NoopTtsProvider + skill staging inside ephemeral engine memory.
    let staged = engine.skills_stage(
        "headless_smoke",
        "Headless Smoke",
        "-- offline check\nreturn true",
    )?;
    anyhow::ensure!(staged.skill_id == "headless_smoke");
    let noop = NoopTtsProvider::default();
    let tts_res = noop.synthesize("check", AvatarEmotion::Neutral).await?;
    anyhow::ensure!(tts_res.provider == "noop");

    let (planner_provider, planner_model, planner_role) = engine.config().planner_binding();
    let thinking = engine
        .config()
        .providers
        .get(&planner_provider)
        .is_some_and(|p| engine.config().resolve_thinking_enabled(planner_role, p));
    println!(
        "jikud headless check passed (planner={planner_provider}/{planner_model}, thinking={thinking}, decision_model={})",
        engine
            .config()
            .decision_router
            .decision_model_id
            .as_deref()
            .unwrap_or("emulated")
    );
    Ok(())
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let args = match parse_args() {
        Ok(parsed) => parsed,
        Err(code) => process::exit(code),
    };

    if args.validate_config {
        match (
            JikuConfig::load_resolved(args.config.as_deref()),
            JikuLuaPolicy::load_resolved(args.policy.as_deref()),
        ) {
            (Ok((cfg, cfg_path)), Ok((lua, pol_path))) => {
                let (planner_provider, planner_model, planner_role) = cfg.planner_binding();
                let thinking = cfg
                    .providers
                    .get(&planner_provider)
                    .is_some_and(|p| cfg.resolve_thinking_enabled(planner_role, p));
                let summary = json!({
                    "valid": true,
                    "config_path": cfg_path.map(|p| p.display().to_string()),
                    "policy_path": pol_path.map(|p| p.display().to_string()),
                    "socket_path": cfg.daemon_socket_path().display().to_string(),
                    "planner": {
                        "provider": planner_provider,
                        "model": planner_model,
                        "thinking_enabled": thinking,
                    },
                    "decision_router": {
                        "provider": cfg.decision_router.decision_provider,
                        "model": cfg.decision_router.decision_model_id,
                    },
                    "lua_hook_counts": lua.hook_counts(),
                });
                println!(
                    "{}",
                    serde_json::to_string_pretty(&summary).unwrap_or_default()
                );
                return;
            }
            (Err(e), _) => {
                eprintln!("configuration error: {e}");
                process::exit(1);
            }
            (_, Err(e)) => {
                eprintln!("Lua policy error: {e}");
                process::exit(1);
            }
        }
    }

    if args.migrate_config {
        let (cfg, _) = match JikuConfig::load_resolved(args.config.as_deref()) {
            Ok(loaded) => loaded,
            Err(e) => {
                eprintln!("configuration error: {e}");
                process::exit(1);
            }
        };
        match &args.out {
            Some(out) => match jikud::migrate::migrate_to_path(&cfg, out, args.force) {
                Ok(path) => println!("wrote {}", path.display()),
                Err(e) => {
                    eprintln!("migration error: {e}");
                    process::exit(1);
                }
            },
            None => match jikud::migrate::render_checked_lua_config(&cfg) {
                Ok(rendered) => print!("{rendered}"),
                Err(e) => {
                    eprintln!("migration error: {e}");
                    process::exit(1);
                }
            },
        }
        return;
    }

    if args.headless_check {
        if let Err(e) = run_headless_check(&args).await {
            eprintln!("headless check failed: {e:#}");
            process::exit(1);
        }
        return;
    }

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .try_init();

    if let Err(e) = run_daemon_server(
        args.socket.as_deref(),
        args.config.as_deref(),
        args.policy.as_deref(),
        false,
    )
    .await
    {
        eprintln!("jikud daemon error: {e:#}");
        process::exit(1);
    }
}
