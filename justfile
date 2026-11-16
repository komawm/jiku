set shell := ["bash", "-euo", "pipefail", "-c"]

default:
    @just --list

# Quality gates: metadata, skills, actionlint, shellcheck, format, typecheck, clippy, and tests.
check: infra-check skills-check fmt-check typecheck clippy test

infra-check: metadata actionlint scripts-check

metadata:
    python3 scripts/check-infrastructure.py

skills-check:
    python3 scripts/check-skills.py

cargo-metadata:
    cargo metadata --format-version 1 --no-deps > /dev/null

actionlint:
    actionlint -color

scripts-check:
    shellcheck scripts/*.sh

fmt:
    cargo fmt --all

fmt-check:
    cargo fmt --all -- --check

typecheck:
    cargo check --workspace --all-targets --locked

clippy:
    cargo clippy --workspace --all-targets --locked -- -D warnings

test:
    cargo test --workspace --all-targets --locked

hooks-install:
    lefthook install pre-commit
    carryctx hooks install --compose


# Source-only development prerelease archive, manifest and checksums.
package-source version out="dist":
    bash scripts/package-source.sh --version "{{version}}" --out "{{out}}"

# Prebuilt binary development prerelease archive, manifest and checksums.
package-release version target="x86_64-unknown-linux-gnu" out="dist":
    bash scripts/package-release.sh --version "{{version}}" --target "{{target}}" --out "{{out}}"
