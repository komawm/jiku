# Jiku

Jiku is the proposed out-of-process desktop GUI orchestration companion for [Koma](https://github.com/komawm/koma). This independent repository establishes its ownership boundary and preserves the existing configuration and service templates. It contains four library-only scaffolding crates (`jikud`, `jiku`, `jiku-memory`, `jiku-tts`), all non-runnable and unpublished. There is no `jiku`/`jikud` executable, provider integration or functioning daemon yet.

- [Canonical architecture](https://github.com/komawm/docs/blob/main/docs/JIKU_ARCHITECTURE.md) and [configuration proposal](https://github.com/komawm/docs/blob/main/docs/JIKU_CONFIG_SPEC.md) live in koma-docs.
- [Crate scaffolding](https://github.com/komawm/jiku/tree/main/crates) reserves the daemon, controller, memory and speech boundaries.
- [Examples](https://github.com/komawm/jiku/tree/main/examples) contain the imported provider and Lua-policy templates.
- [Resources](https://github.com/komawm/jiku/tree/main/resources) contain the proposed `jikud.service`; it cannot run without an implementation and validated installation.
- [Skills](https://github.com/komawm/jiku/tree/main/skills) is an internal Jiku directory. There is no separate jiku-skills repository or implemented skill runtime.

Provider names, routes, memory/voice settings, sandbox restrictions and CLI commands in the templates are design inputs. Keep real credentials out of examples and do not treat a service file as security or runtime acceptance. Avatar integration is deferred and does not have a repository.

The incubation baseline is 0.0.1 on the 0.0.x line before a first formal 0.1.0 release line. The Rust scaffolding targets Rust 1.100.0 and Rust 2024; its local checks pass under that target compiler, with no runtime tests. A scaffold build does not establish an agent runtime. All Jiku crates use [GPL-3.0-or-later](https://github.com/komawm/jiku/blob/main/LICENSE). Preserve third-party attribution when implementation is added.

Read [AGENTS.md](https://github.com/komawm/jiku/blob/main/AGENTS.md), use the repository justfile for its actual documentation/template gates, and track work in its local CarryCtx project. Local initialization does not publish or release the component.
