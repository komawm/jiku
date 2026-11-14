# Changelog

All notable changes to Jiku are recorded here. Versions follow the `0.0.x`
incubation line before a formal `0.1.0` release.

## [Unreleased]

## [0.0.1] - 2026-11-14

Source-only development prerelease of the library-only scaffold. There is no
`jiku` or `jikud` executable, functioning daemon, or network provider
integration in this release.

### Added

- Three-Driver model contracts in `jikud`: `DecisionsDriver`,
  `TurnModelDriver`, and optional `LiveSessionDriver` as pure-data types with
  unit-tested invariants.
- `DecisionExecutionMode` selecting `NativeDecisionsApi` when a dedicated
  decision model is configured and falling back to
  `EmulatedStructuredTextModel` (structured prompt plus JSON Schema
  constraints on the primary text/VLM model) otherwise.
- `DecisionCompiler` primitives compiling decision distributions into
  `TimeSlotMacro`, `BernoulliExpectedStep`, and `ThresholdProbe` steps, plus
  sub-cell grid centroid interpolation.
- Configurable defaults via `Default` implementations for `GridSpec`,
  `DecisionCompilerConfig`, and `LiveSessionConfig`, documented in
  `examples/config.toml`.
- Source release tooling: `scripts/package-source.sh`, `just package-source`,
  and a tag-triggered prerelease workflow.

### Known limits

- Library-only scaffold: no daemon, no CLI binary, no installable service.
- No network provider integration; provider settings in `examples/` are
  design inputs only.
- Memory and speech crates are documentation-only boundaries.

[Unreleased]: https://github.com/komawm/jiku/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/komawm/jiku/releases/tag/v0.0.1
