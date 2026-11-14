# Jiku service template

[jikud.service](https://github.com/komawm/jiku/blob/main/resources/jikud.service) is the imported proposed service unit for the [Jiku architecture](https://github.com/komawm/docs/blob/main/docs/JIKU_ARCHITECTURE.md). This repository currently builds no `jikud` binary. The template's filesystem restrictions require dedicated validation when an implementation exists.

Do not install or activate the unit during workspace initialization. Coordinate actual service lifecycle and Koma IPC integration in explicit implementation and runtime-test tasks.
