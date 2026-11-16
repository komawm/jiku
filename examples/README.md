# Jiku configuration templates

[config.toml](https://github.com/komawm/jiku/blob/main/examples/config.toml) and [init.lua](https://github.com/komawm/jiku/blob/main/examples/init.lua) accompany the [configuration specification](https://github.com/komawm/docs/blob/main/docs/JIKU_CONFIG_SPEC.md) and [architecture](https://github.com/komawm/docs/blob/main/docs/JIKU_ARCHITECTURE.md).

Both files are generic templates kept in sync with the `v0.0.1` baseline: provider endpoints, model IDs, and `api_key_env` variable names are placeholders (`<...-model-id>`, `JIKU_*_API_KEY`) that you replace with your own bindings. `config.toml` validates offline via `jikud --validate-config` without any credentials, and `init.lua` loads in the sandboxed Phodopus Lua runtime.

Keep real credentials and private endpoints outside tracked examples. Never inline raw API keys into `config.toml`; store them in the environment variable named by `api_key_env`.
