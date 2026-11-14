#!/usr/bin/env bash
# Build a source-only development prerelease archive for Jiku.
# Produces jiku-<version>-src.tar.gz, RELEASE-MANIFEST.toml and SHA256SUMS.
# Pre-v0.1.0 runtime archive containing jikud, jiku CLI/MCP, jiku-memory, and jiku-tts.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/package-source.sh --version X.Y.Z --out DIR

  --version X.Y.Z  Release version; must equal workspace.package.version.
  --out DIR        Output directory (created if missing). Existing release
                   files are never overwritten.
EOF
}

version=""
out=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --version)
            [[ $# -ge 2 ]] || { echo "error: --version needs a value" >&2; exit 2; }
            version="$2"
            shift 2
            ;;
        --version=*)
            version="${1#*=}"
            shift
            ;;
        --out)
            [[ $# -ge 2 ]] || { echo "error: --out needs a value" >&2; exit 2; }
            out="$2"
            shift 2
            ;;
        --out=*)
            out="${1#*=}"
            shift
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$version" || -z "$out" ]]; then
    usage >&2
    exit 2
fi
if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "error: version must be X.Y.Z without a leading 'v': $version" >&2
    exit 2
fi

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$root"

workspace_version="$(python3 - <<'PY'
import tomllib
with open("Cargo.toml", "rb") as source:
    print(tomllib.load(source)["workspace"]["package"]["version"])
PY
)"
if [[ "$version" != "$workspace_version" ]]; then
    echo "error: version $version does not match workspace.package.version $workspace_version" >&2
    exit 1
fi

crates="$(python3 - <<'PY'
import tomllib
from pathlib import Path
with open("Cargo.toml", "rb") as source:
    members = tomllib.load(source)["workspace"]["members"]
print(", ".join(f'"{Path(member).name}"' for member in members))
PY
)"

revision="$(git rev-parse HEAD)"
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "warning: working tree has uncommitted changes; the archive contains HEAD ($revision) only" >&2
fi

name="jiku-${version}"
archive="${name}-src.tar.gz"
manifest="RELEASE-MANIFEST.toml"
sums="SHA256SUMS"

mkdir -p -- "$out"
for file in "$archive" "$manifest" "$sums"; do
    if [[ -e "$out/$file" ]]; then
        echo "error: refusing to overwrite existing $out/$file" >&2
        exit 1
    fi
done

git archive --format=tar.gz --prefix="${name}/" -o "$out/$archive" HEAD

cat >"$out/$manifest" <<EOF
# Jiku source-only development prerelease manifest.
[release]
name = "jiku"
version = "${version}"
revision = "${revision}"
prerelease = true
channel = "development"
kind = "source"
archive = "${archive}"
license = "GPL-3.0-or-later"
crates = [${crates}]
note = "pre-v0.1.0 development release containing jikud, jiku CLI/MCP, jiku-memory, and jiku-tts"
EOF

(cd -- "$out" && sha256sum -- "$archive" "$manifest" >"$sums")

echo "Wrote $out/$archive"
echo "Wrote $out/$manifest"
echo "Wrote $out/$sums"
