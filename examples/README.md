# Jiku configuration templates

[config.toml](https://github.com/komawm/jiku/blob/main/examples/config.toml), [config.lua](https://github.com/komawm/jiku/blob/main/examples/config.lua), and [init.lua](https://github.com/komawm/jiku/blob/main/examples/init.lua) accompany the [configuration specification](https://github.com/komawm/docs/blob/main/docs/JIKU_CONFIG_SPEC.md) and [architecture](https://github.com/komawm/docs/blob/main/docs/JIKU_ARCHITECTURE.md).

`config.toml` and `config.lua` are equivalent front ends for the same typed configuration; pick either extension, or keep TOML as the base and let `config.lua` overlay the keys it changes. `init.lua` holds the sandboxed policy hooks. All three files carry placeholders (`<...-model-id>`, `JIKU_*_API_KEY`) that you replace with your own bindings.

Both configuration files validate offline via `jikud --validate-config` without any credentials, and `init.lua` loads in the sandboxed Phodopus Lua runtime.

Keep real credentials and private endpoints outside tracked examples. Never inline raw API keys into a configuration file; store them in the environment variable named by `api_key_env`.
