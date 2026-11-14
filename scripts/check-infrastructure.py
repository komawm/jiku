#!/usr/bin/env python3
"""Check Jiku's declared scaffold topology without requiring a compiler."""
from pathlib import Path
import tomllib

root = Path(__file__).resolve().parent.parent

def load(relative):
    with (root / relative).open("rb") as source:
        return tomllib.load(source)

repo = load("repo.toml")["repo"]
workspace = load("Cargo.toml")["workspace"]
package = workspace["package"]
toolchain = load("rust-toolchain.toml")["toolchain"]
assert repo["name"] == "jiku", "repository identity drift"
assert repo["repository"] == package["repository"], "repository URL drift"
assert repo["docs"].startswith("https://github.com/komawm/"), "canonical docs URL required"
assert package["edition"] == "2024", "unexpected Rust edition"
assert toolchain["channel"].startswith(package["rust-version"] + "."), "MSRV/toolchain drift"
assert package["license"] == "GPL-3.0-or-later", "product license drift"
assert "license-file" not in package, "use SPDX license only; license-file conflicts with license"
assert (root / "LICENSE").read_text().startswith("GNU GENERAL PUBLIC LICENSE"), "GPL license text missing"
assert list(root.glob("LICENSE*")) == [root / "LICENSE"], "keep one root LICENSE"
for member in workspace["members"]:
    crate = load(Path(member) / "Cargo.toml")["package"]
    assert crate["name"] == Path(member).name, f"crate identity drift: {member}"
    assert crate["version"] == {"workspace": True}, f"version drift: {member}"
    assert crate["license"] == {"workspace": True}, f"license drift: {member}"
    assert "license-file" not in crate, f"use SPDX license only: {member}"
    assert not list((root / member).glob("LICENSE*")), f"use the shared root LICENSE: {member}"
    assert (root / member / "src/lib.rs").is_file(), f"crate source missing: {member}"
load(".carryctx/config.toml")
assert not (root / ".gitmodules").exists(), "Jiku does not use Git submodules"
print(f"Infrastructure metadata valid: {len(workspace['members'])} Cargo crates")

