set shell := ["bash", "-euo", "pipefail", "-c"]

default:
    @just --list

# These gates validate the scaffold; they do not establish agent functionality.
check: infra-check fmt-check typecheck clippy test

infra-check: metadata actionlint scripts-check

metadata:
    python3 scripts/check-infrastructure.py

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
