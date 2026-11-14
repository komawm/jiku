//! Sandboxed `Phodopus` Lua 5.4 runtime for `~/.config/jiku/init.lua` policy hooks.
//!
//! Strips `io`, `os`, `debug`, and `package`, enforces hard memory limits via
//! `gc-arena`, and steps every hook under deterministic instruction `Fuel`.

use std::{cell::RefCell, collections::BTreeMap, fs, path::Path, rc::Rc, str::FromStr};

use jiku_memory::expand_tilde_path;
use jiku_tts::{AvatarEmotion, NotificationSpec};
use phodopus::{
    Callback, CallbackReturn, Closure, Context, Error as LuaError, Executor, Fuel, Function, Lua,
    RuntimeError, StashedFunction, Table, Value,
};
use phodopus_util::serde::{LuaSerdeExt as _, to_value};
use serde_json::Value as JsonValue;
use thiserror::Error;

/// Hard heap limit for Jiku's embedded Phodopus Lua sandbox (8 MiB).
pub const DEFAULT_LUA_HEAP_LIMIT_BYTES: usize = 8 * 1024 * 1024;

/// Instruction budget allocated to evaluating `init.lua` and individual hook calls.
pub const POLICY_FUEL_BUDGET: i32 = 200_000;

/// Errors raised when initializing or invoking `init.lua` inside Phodopus.
#[derive(Debug, Error)]
pub enum LuaPolicyError {
    /// Failed reading `init.lua` from disk.
    #[error("failed reading policy file: {0}")]
    Io(#[from] std::io::Error),

    /// Lua syntax or runtime execution error (including fuel exhaustion).
    #[error("Phodopus Lua policy error: {0}")]
    Lua(String),
}

/// Collected side-effects emitted by `jiku.notify`, `jiku.tts.say`, and `jiku.avatar.set_emotion`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PolicyEffects {
    /// Queued desktop notifications.
    pub notifications: Vec<NotificationSpec>,
    /// Queued speech utterances.
    pub tts_utterances: Vec<String>,
    /// Companion emotion transitions triggered by hooks.
    pub avatar_emotions: Vec<AvatarEmotion>,
}

#[derive(Default)]
struct PolicyInner {
    prefetch_hooks: Vec<StashedFunction>,
    event_hooks: BTreeMap<String, Vec<StashedFunction>>,
    effects: PolicyEffects,
}

/// Encapsulated `Phodopus` Lua 5.4 policy VM executing Jiku's `init.lua` hooks.
pub struct JikuLuaPolicy {
    lua: Lua,
    inner: Rc<RefCell<PolicyInner>>,
}

impl Default for JikuLuaPolicy {
    fn default() -> Self {
        Self::new(DEFAULT_LUA_HEAP_LIMIT_BYTES).expect("default Phodopus policy init")
    }
}

impl JikuLuaPolicy {
    /// Creates an isolated Phodopus Lua VM with zero `io`/`os`/`debug` libraries.
    pub fn new(heap_limit_bytes: usize) -> Result<Self, LuaPolicyError> {
        let mut builder = Lua::builder();
        builder.memory_limit(heap_limit_bytes);
        let mut lua = builder.build();
        lua.load_serde();

        let inner = Rc::new(RefCell::new(PolicyInner::default()));
        let inner_ref = inner.clone();
        lua.try_enter(|ctx| install_jiku_globals(ctx, inner_ref))
            .map_err(|e| LuaPolicyError::Lua(format!("failed installing jiku.* globals: {e}")))?;

        Ok(Self { lua, inner })
    }

    /// Compiles and evaluates Lua policy source code under [`POLICY_FUEL_BUDGET`].
    pub fn eval_source(&mut self, source: &str) -> Result<(), LuaPolicyError> {
        let stashed = self
            .lua
            .try_enter(|ctx| {
                let closure = Closure::load(ctx, Some("=init.lua"), source.as_bytes())?;
                Ok(ctx.stash(Executor::start(ctx, closure.into(), ())))
            })
            .map_err(|e| LuaPolicyError::Lua(format!("syntax error: {e}")))?;

        self.lua
            .try_enter(|ctx| {
                let exec = ctx.fetch(&stashed);
                let mut fuel = Fuel::with(POLICY_FUEL_BUDGET);
                let done = exec.step(ctx, &mut fuel).map_err(RuntimeError::new)?;
                if !done && !fuel.should_continue() {
                    return Err(LuaError::from(RuntimeError::new(std::io::Error::other(
                        "init.lua exceeded POLICY_FUEL_BUDGET",
                    ))));
                }
                exec.take_result::<()>(ctx).map_err(RuntimeError::new)??;
                Ok(())
            })
            .map_err(|e| LuaPolicyError::Lua(format!("evaluation error: {e}")))
    }

    /// Reads and evaluates `~/.config/jiku/init.lua` if `path` exists.
    pub fn load_file(&mut self, path: &Path) -> Result<(), LuaPolicyError> {
        let resolved = expand_tilde_path(path);
        let source = fs::read_to_string(resolved)?;
        self.eval_source(&source)
    }

    /// Resolves and loads `--policy`, `$JIKU_POLICY`, or `~/.config/jiku/init.lua` when present.
    pub fn load_resolved(
        explicit: Option<&Path>,
    ) -> Result<(Self, Option<std::path::PathBuf>), LuaPolicyError> {
        let mut vm = Self::default();
        if let Some(p) = explicit {
            let resolved = expand_tilde_path(p);
            vm.load_file(&resolved)?;
            return Ok((vm, Some(resolved)));
        }
        if let Some(env_p) = std::env::var_os("JIKU_POLICY").filter(|v| !v.is_empty()) {
            let resolved = expand_tilde_path(Path::new(&env_p));
            vm.load_file(&resolved)?;
            return Ok((vm, Some(resolved)));
        }
        let default_candidate = std::env::var_os("XDG_CONFIG_HOME")
            .filter(|v| !v.is_empty())
            .map(std::path::PathBuf::from)
            .or_else(|| {
                std::env::var_os("HOME")
                    .filter(|v| !v.is_empty())
                    .map(|h| std::path::PathBuf::from(h).join(".config"))
            })
            .map(|dir| dir.join("jiku").join("init.lua"));
        if let Some(p) = default_candidate
            && p.is_file()
        {
            vm.load_file(&p)?;
            return Ok((vm, Some(p)));
        }
        Ok((vm, None))
    }

    /// Invokes every registered `jiku.memory.on_prefetch` hook and returns injected prompt notes.
    pub fn run_prefetch(
        &mut self,
        active_app_id: &str,
        workspace: &str,
        goal: &str,
    ) -> Result<Vec<String>, LuaPolicyError> {
        let hooks: Vec<StashedFunction> = self.inner.borrow().prefetch_hooks.clone();
        if hooks.is_empty() {
            return Ok(Vec::new());
        }

        let notes = Rc::new(RefCell::new(Vec::<String>::new()));
        for func_handle in hooks {
            let notes_ref = notes.clone();
            let app_owned = active_app_id.to_string();
            let ws_owned = workspace.to_string();
            let goal_owned = goal.to_string();

            self.lua
                .try_enter(move |ctx| {
                    let tbl = Table::new(&ctx);
                    let app_val = to_value(ctx, &app_owned).map_err(RuntimeError::new)?;
                    let ws_val = to_value(ctx, &ws_owned).map_err(RuntimeError::new)?;
                    let goal_val = to_value(ctx, &goal_owned).map_err(RuntimeError::new)?;
                    tbl.set(ctx, "active_app_id", app_val)?;
                    tbl.set(ctx, "workspace", ws_val)?;
                    tbl.set(ctx, "goal", goal_val)?;
                    let append_cb = Callback::from_fn(&ctx, move |ctx, _, mut stack| {
                        let (first, second): (Value, Option<Value>) = stack.consume(ctx)?;
                        let text_val = second.unwrap_or(first);
                        if let Value::String(s) = text_val
                            && let Ok(utf8) = s.to_str()
                        {
                            notes_ref.borrow_mut().push(utf8.to_string());
                        }
                        Ok(CallbackReturn::Return)
                    });
                    tbl.set(ctx, "append_note", append_cb)?;

                    let f = ctx.fetch(&func_handle);
                    let exec = Executor::start(ctx, f, tbl);
                    let mut fuel = Fuel::with(POLICY_FUEL_BUDGET);
                    let _ = exec.step(ctx, &mut fuel).map_err(RuntimeError::new)?;
                    exec.take_result::<()>(ctx).map_err(RuntimeError::new)??;
                    Ok(())
                })
                .map_err(|e| LuaPolicyError::Lua(format!("on_prefetch error: {e}")))?;
        }

        Ok(notes.borrow().clone())
    }

    /// Emits a lifecycle event (such as `"episode.completed"` or `"security.verification_needed"`)
    /// to registered `jiku.on(event, fn)` listeners and drains accumulated effects.
    pub fn emit_event(
        &mut self,
        event_name: &str,
        payload: &JsonValue,
    ) -> Result<PolicyEffects, LuaPolicyError> {
        let hooks: Vec<StashedFunction> = self
            .inner
            .borrow()
            .event_hooks
            .get(event_name)
            .cloned()
            .unwrap_or_default();

        for func_handle in hooks {
            let payload_clone = payload.clone();
            self.lua
                .try_enter(move |ctx| {
                    let lua_arg = to_value(ctx, &payload_clone).map_err(RuntimeError::new)?;
                    let f = ctx.fetch(&func_handle);
                    let exec = Executor::start(ctx, f, lua_arg);
                    let mut fuel = Fuel::with(POLICY_FUEL_BUDGET);
                    let _ = exec.step(ctx, &mut fuel).map_err(RuntimeError::new)?;
                    exec.take_result::<()>(ctx).map_err(RuntimeError::new)??;
                    Ok(())
                })
                .map_err(|e| {
                    LuaPolicyError::Lua(format!("event `{event_name}` hook error: {e}"))
                })?;
        }

        let mut guard = self.inner.borrow_mut();
        Ok(std::mem::take(&mut guard.effects))
    }

    /// Number of registered event hook handlers across all event topics.
    #[must_use]
    pub fn hook_counts(&self) -> (usize, usize) {
        let guard = self.inner.borrow();
        let event_count: usize = guard.event_hooks.values().map(Vec::len).sum();
        (guard.prefetch_hooks.len(), event_count)
    }
}

fn install_jiku_globals<'gc>(
    ctx: Context<'gc>,
    inner: Rc<RefCell<PolicyInner>>,
) -> Result<(), LuaError<'gc>> {
    let jiku = Table::new(&ctx);

    // jiku.memory.on_prefetch(fn)
    let memory_tbl = Table::new(&ctx);
    {
        let st = inner.clone();
        memory_tbl.set_field(
            ctx,
            "on_prefetch",
            Callback::from_fn(&ctx, move |ctx, _, mut stack| {
                let func: Function = stack.consume(ctx)?;
                st.borrow_mut().prefetch_hooks.push(ctx.stash(func));
                Ok(CallbackReturn::Return)
            }),
        );
    }
    jiku.set_field(ctx, "memory", memory_tbl);

    // jiku.on(event_name, fn)
    {
        let st = inner.clone();
        jiku.set_field(
            ctx,
            "on",
            Callback::from_fn(&ctx, move |ctx, _, mut stack| {
                let (event_name, func): (String, Function) = stack.consume(ctx)?;
                st.borrow_mut()
                    .event_hooks
                    .entry(event_name)
                    .or_default()
                    .push(ctx.stash(func));
                Ok(CallbackReturn::Return)
            }),
        );
    }

    // jiku.notify({ summary = ..., body = ..., urgency = ... })
    {
        let st = inner.clone();
        jiku.set_field(
            ctx,
            "notify",
            Callback::from_fn(&ctx, move |ctx, _, mut stack| {
                let tbl: Table = stack.consume(ctx)?;
                let summary: String = tbl.get(ctx, "summary").unwrap_or_default();
                let body: String = tbl.get(ctx, "body").unwrap_or_default();
                let urgency: String = tbl
                    .get(ctx, "urgency")
                    .unwrap_or_else(|_| "normal".to_string());
                st.borrow_mut()
                    .effects
                    .notifications
                    .push(NotificationSpec {
                        summary,
                        body,
                        urgency,
                    });
                Ok(CallbackReturn::Return)
            }),
        );
    }

    // jiku.tts.say(text)
    let tts_tbl = Table::new(&ctx);
    {
        let st = inner.clone();
        tts_tbl.set_field(
            ctx,
            "say",
            Callback::from_fn(&ctx, move |ctx, _, mut stack| {
                let text: String = stack.consume(ctx)?;
                st.borrow_mut().effects.tts_utterances.push(text);
                Ok(CallbackReturn::Return)
            }),
        );
    }
    jiku.set_field(ctx, "tts", tts_tbl);

    // jiku.avatar.set_emotion(emotion)
    let avatar_tbl = Table::new(&ctx);
    {
        let st = inner;
        avatar_tbl.set_field(
            ctx,
            "set_emotion",
            Callback::from_fn(&ctx, move |ctx, _, mut stack| {
                let raw: String = stack.consume(ctx)?;
                let emotion = AvatarEmotion::from_str(&raw).unwrap_or(AvatarEmotion::Neutral);
                st.borrow_mut().effects.avatar_emotions.push(emotion);
                Ok(CallbackReturn::Return)
            }),
        );
    }
    jiku.set_field(ctx, "avatar", avatar_tbl);

    ctx.globals().set_field(ctx, "jiku", jiku);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::path::PathBuf;

    #[test]
    fn evaluates_repository_example_init_lua_hooks_safely() {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let example_lua = manifest_dir.join("../../examples/init.lua");
        let mut policy = JikuLuaPolicy::default();
        policy
            .load_file(&example_lua)
            .expect("examples/init.lua must load under Phodopus sandbox");

        assert_eq!(policy.hook_counts(), (1, 2));

        let notes = policy
            .run_prefetch("org.PWMT.zathura", "ws_01", "Scroll PDF")
            .unwrap();
        assert_eq!(notes.len(), 1);
        assert!(notes[0].contains("keyboard navigation"));

        let ep_fx = policy
            .emit_event(
                "episode.completed",
                &json!({ "workspace": "ws_01", "summary": "Done", "duration_ms": 6200 }),
            )
            .unwrap();
        assert_eq!(ep_fx.notifications.len(), 1);
        assert_eq!(ep_fx.tts_utterances.len(), 1);

        let sec_fx = policy
            .emit_event(
                "security.verification_needed",
                &json!({ "workspace": "ws_01" }),
            )
            .unwrap();
        assert_eq!(sec_fx.avatar_emotions, vec![AvatarEmotion::Alert]);
        assert_eq!(sec_fx.notifications[0].urgency, "critical");
    }

    #[test]
    fn strips_unsafe_io_and_os_libraries_from_sandbox() {
        let mut policy = JikuLuaPolicy::default();
        assert!(policy.eval_source("io.open('/etc/passwd', 'r')").is_err());
        assert!(policy.eval_source("os.execute('id')").is_err());
    }
}
