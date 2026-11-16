//! Sandboxed Lua configuration front end (`config.lua`).
//!
//! `config.lua` populates the same typed [`JikuConfig`] that `config.toml`
//! deserializes into. The first file is merged over built-in defaults (or over a
//! previously loaded `config.toml`), so an overlay only needs to state the keys
//! it changes.
//!
//! # Sandbox
//!
//! The VM is built with the same profile as the `init.lua` policy VM: hard
//! `gc-arena` heap ceiling, deterministic instruction [`Fuel`], and **no**
//! `io`, `os`, `package`, `dofile`, or `loadfile`. In particular the sandbox
//! cannot read the environment, so `api_key_env` stays a variable *name* and
//! secrets never enter the configuration file or the Lua heap.
//!
//! # Surface
//!
//! ```lua
//! jiku.config{
//!     daemon = { locale = "zh-CN", socket_name = "jiku.sock" },
//!     providers = {
//!         primary = {
//!             kind = "openai_compatible",
//!             base_url = "https://your-endpoint.example/v1",
//!             api_key_env = "JIKU_PLANNER_API_KEY",
//!         },
//!     },
//!     roles = {
//!         planner = { provider = "primary", model = "<primary-text-model-id>", driver = "turn_sse" },
//!     },
//! }
//! ```
//!
//! Tables merge field by field across calls; arrays and scalars replace the
//! value that was there.

use std::{cell::RefCell, fs, path::Path, rc::Rc};

use phodopus::{
    Callback, CallbackReturn, Closure, Context, Error as LuaError, Executor, Fuel, Lua,
    RuntimeError, Stack, Table, Value,
};
use phodopus_util::serde::{LuaSerdeExt as _, from_value};
use serde_json::Value as JsonValue;

use crate::config::{ConfigError, JikuConfig};

/// Hard heap ceiling for the configuration VM (8 MiB), matching the policy VM.
pub const CONFIG_LUA_HEAP_LIMIT_BYTES: usize = 8 * 1024 * 1024;

/// Instruction budget for evaluating `config.lua`.
pub const CONFIG_LUA_FUEL_BUDGET: i32 = 200_000;

/// Upper bound on the accumulated JSON configuration document (4 MiB). The Lua
/// heap ceiling does not cover the Rust-side accumulator, so patches are measured
/// after each call and rejected when the document exceeds this size.
pub const CONFIG_LUA_JSON_LIMIT_BYTES: usize = 4 * 1024 * 1024;

/// Loads `config.lua` from `path` and merges it over `base`.
pub fn load_lua_config(base: JikuConfig, path: &Path) -> Result<JikuConfig, ConfigError> {
    let source = fs::read_to_string(path).map_err(|source| ConfigError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    apply_lua_source(base, &source, &path.display().to_string())
}

/// Evaluates `source` in a fresh sandbox and merges every `jiku.config{...}`
/// patch over `base`, then runs [`JikuConfig::validate`].
///
/// # Complexity
///
/// Time `O(S + E)` for `S` source bytes and `E` configuration entries; space
/// `O(E)` for the merged JSON document plus the VM heap ceiling.
pub fn apply_lua_source(
    base: JikuConfig,
    source: &str,
    label: &str,
) -> Result<JikuConfig, ConfigError> {
    let merged = Rc::new(RefCell::new(serde_json::to_value(&base).map_err(|e| {
        ConfigError::Lua(format!("failed serializing base configuration: {e}"))
    })?));

    let mut builder = Lua::builder();
    builder.memory_limit(CONFIG_LUA_HEAP_LIMIT_BYTES);
    let mut lua = builder.build();
    lua.load_serde();

    lua.try_enter(|ctx| {
        let jiku = Table::new(&ctx);
        {
            let acc = merged.clone();
            jiku.set_field(
                ctx,
                "config",
                Callback::from_fn(&ctx, move |ctx, _, mut stack| {
                    let patch = consume_object(ctx, &mut stack, "jiku.config{...}")?;
                    {
                        let mut document = acc.borrow_mut();
                        deep_merge(&mut document, &patch);
                        ensure_document_bound(&document)?;
                    }
                    Ok(CallbackReturn::Return)
                }),
            );
        }
        {
            let acc = merged.clone();
            jiku.set_field(
                ctx,
                "config_replace",
                Callback::from_fn(&ctx, move |ctx, _, mut stack| {
                    let patch = consume_object(ctx, &mut stack, "jiku.config_replace{...}")?;
                    {
                        let mut document = acc.borrow_mut();
                        replace_sections(&mut document, &patch);
                        ensure_document_bound(&document)?;
                    }
                    Ok(CallbackReturn::Return)
                }),
            );
        }
        ctx.globals().set_field(ctx, "jiku", jiku);
        Ok(())
    })
    .map_err(|e| ConfigError::Lua(format!("failed installing `jiku.config` in {label}: {e}")))?;

    let stashed = lua
        .try_enter(|ctx| {
            let closure = Closure::load(ctx, Some(label), source.as_bytes())?;
            Ok(ctx.stash(Executor::start(ctx, closure.into(), ())))
        })
        .map_err(|e| ConfigError::Lua(format!("syntax error in {label}: {e}")))?;

    lua.try_enter(|ctx| {
        let exec = ctx.fetch(&stashed);
        let mut fuel = Fuel::with(CONFIG_LUA_FUEL_BUDGET);
        let done = exec.step(ctx, &mut fuel).map_err(RuntimeError::new)?;
        if !done && !fuel.should_continue() {
            return Err(LuaError::from(RuntimeError::new(std::io::Error::other(
                "config.lua exceeded CONFIG_LUA_FUEL_BUDGET",
            ))));
        }
        exec.take_result::<()>(ctx).map_err(RuntimeError::new)??;
        Ok(())
    })
    .map_err(|e| ConfigError::Lua(format!("evaluation error in {label}: {e}")))?;

    let document = merged.borrow().clone();

    let cfg: JikuConfig = serde_json::from_value(document).map_err(|e| {
        ConfigError::Lua(format!(
            "schema mismatch in {label} (unknown keys or wrong value types): {e}"
        ))
    })?;
    cfg.validate()?;
    Ok(cfg)
}

/// Consumes a Lua argument as a JSON object, rejecting scalars and arrays.
fn consume_object<'gc>(
    ctx: Context<'gc>,
    stack: &mut Stack<'gc, '_>,
    name: &'static str,
) -> Result<JsonValue, LuaError<'gc>> {
    let value: Value<'gc> = stack.consume(ctx)?;
    let patch: JsonValue = from_value(value).map_err(RuntimeError::new)?;
    if !patch.is_object() {
        return Err(LuaError::from(RuntimeError::new(std::io::Error::other(
            format!("{name} expects a table of named configuration sections"),
        ))));
    }
    Ok(patch)
}

/// Rejects accumulator documents that exceed [`CONFIG_LUA_JSON_LIMIT_BYTES`].
fn ensure_document_bound<'gc>(document: &JsonValue) -> Result<(), LuaError<'gc>> {
    let len = serde_json::to_string(document)
        .map(|text| text.len())
        .unwrap_or(usize::MAX);
    if len > CONFIG_LUA_JSON_LIMIT_BYTES {
        return Err(LuaError::from(RuntimeError::new(std::io::Error::other(
            "config.lua produced a configuration document larger than 4 MiB",
        ))));
    }
    Ok(())
}

/// Replaces whole top-level sections with the provided values.
fn replace_sections(base: &mut JsonValue, patch: &JsonValue) {
    let (Some(base_map), JsonValue::Object(patch_map)) = (base.as_object_mut(), patch) else {
        return;
    };
    for (key, value) in patch_map {
        base_map.insert(key.clone(), value.clone());
    }
}

/// Recursively merges `patch` into `base`: objects merge key by key, every other
/// JSON shape replaces the existing value.
fn deep_merge(base: &mut JsonValue, patch: &JsonValue) {
    match (base, patch) {
        (JsonValue::Object(base_map), JsonValue::Object(patch_map)) => {
            for (key, value) in patch_map {
                match base_map.get_mut(key) {
                    Some(slot) => deep_merge(slot, value),
                    None => {
                        base_map.insert(key.clone(), value.clone());
                    }
                }
            }
        }
        (slot, value) => *slot = value.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LUA_ONLY: &str = r#"
        jiku.config{
            daemon = { locale = "ja-JP" },
            providers = {
                primary = {
                    kind = "openai_compatible",
                    base_url = "https://planner.example/v1",
                    api_key_env = "JIKU_PLANNER_API_KEY",
                },
            },
            roles = {
                planner = {
                    provider = "primary",
                    model = "<primary-text-model-id>",
                    driver = "turn_sse",
                },
            },
        }
    "#;

    #[test]
    fn lua_only_config_populates_typed_ir() {
        let cfg = apply_lua_source(JikuConfig::default(), LUA_ONLY, "=config.lua").unwrap();
        assert_eq!(cfg.daemon.locale, "ja-JP");
        // Untouched sections keep their defaults.
        assert_eq!(
            cfg.daemon.socket_name,
            JikuConfig::default().daemon.socket_name
        );
        let planner = cfg.roles.get("planner").unwrap();
        assert_eq!(planner.provider, "primary");
        assert_eq!(planner.model, "<primary-text-model-id>");
    }

    #[test]
    fn lua_overlay_only_overrides_named_keys() {
        let base = apply_lua_source(JikuConfig::default(), LUA_ONLY, "=base.lua").unwrap();
        let overlay = r#"
            jiku.config{
                daemon = { dropbox_dir = "~/OverlayBox" },
                roles = { planner = { temperature = 0.9 } },
            }
        "#;
        let cfg = apply_lua_source(base, overlay, "=overlay.lua").unwrap();
        assert_eq!(cfg.daemon.dropbox_dir, "~/OverlayBox");
        // Field-level merge keeps the base values that the overlay omitted.
        assert_eq!(cfg.daemon.locale, "ja-JP");
        assert_eq!(cfg.roles["planner"].model, "<primary-text-model-id>");
        assert_eq!(cfg.roles["planner"].temperature, Some(0.9));
    }

    #[test]
    fn unknown_keys_are_rejected() {
        let err = apply_lua_source(
            JikuConfig::default(),
            r#"jiku.config{ daemon = { locale = "en-US", not_a_key = true } }"#,
            "=bad.lua",
        )
        .unwrap_err();
        let message = err.to_string();
        assert!(message.contains("not_a_key"), "unexpected error: {message}");
    }

    #[test]
    fn non_table_patch_is_rejected() {
        let err =
            apply_lua_source(JikuConfig::default(), "jiku.config(7)", "=bad.lua").unwrap_err();
        assert!(err.to_string().contains("expects a table"));
    }

    #[test]
    fn runaway_config_exhausts_fuel() {
        let err = apply_lua_source(
            JikuConfig::default(),
            "local i = 0 while true do i = i + 1 end",
            "=loop.lua",
        )
        .unwrap_err();
        assert!(err.to_string().contains("CONFIG_LUA_FUEL_BUDGET"));
    }

    #[test]
    fn invalid_provider_reference_fails_validation() {
        let err = apply_lua_source(
            JikuConfig::default(),
            r#"jiku.config{ roles = { planner = { provider = "missing", model = "m" } } }"#,
            "=bad.lua",
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("unknown provider"),
            "unexpected: {err}"
        );
    }

    #[test]
    fn parses_repository_example_lua_config() {
        let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let path = manifest_dir.join("../../examples/config.lua");
        let cfg = load_lua_config(JikuConfig::default(), &path)
            .expect("examples/config.lua must validate");
        assert_eq!(cfg.roles["planner"].provider, "primary");
        assert_eq!(cfg.roles["decision_router"].driver, "decisions");
        assert_eq!(cfg.daemon.locale, "zh-CN");
        assert!(!cfg.tts.enabled);
    }

    #[test]
    fn sandbox_has_no_io_or_os() {
        for global in ["io", "os", "dofile", "loadfile"] {
            let src =
                format!("if {global} ~= nil then error('{global} leaked into config.lua') end");
            apply_lua_source(JikuConfig::default(), &src, "=sandbox.lua")
                .unwrap_or_else(|e| panic!("`{global}` should be nil: {e}"));
        }
    }
}
