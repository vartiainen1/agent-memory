#!/usr/bin/env python3
"""_check_drift.py - drift guards for agent-memory (family parity).

Prevents the whole class of "docs claim X while the code does Y" bugs
(the stale-count bug appeared 4 times in this workspace family). Guards:

  1. README test counts match the ACTUAL suites (unit + external-API audit).
  2. SPEC-pinned constants match the implementation (format_version,
     mem_ prefix, recall/search limits, exit-code contract).
  3. SPEC command table matches the implemented CLI subcommands.
  4. pyproject pins Python >= 3.11 (the family floor is 3.9; agent-memory
     deliberately requires 3.11 for stdlib tomllib - ROUND 2).
  5. pyproject declares zero runtime dependencies (stdlib-only claim).
  6. EOL discipline: every .py file uses LF-only line endings
     (byte-identical determinism contract, V0.1_SPEC.md 13/15.8).

Usage: python _check_drift.py    (exit 0 = all guards pass)
"""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE / "agent_memory.py"
SPEC = HERE / "V0.1_SPEC.md"
README = HERE / "README.md"
PYPROJECT = HERE / "pyproject.toml"

failures: list[str] = []


def guard(name: str, cond: bool, detail: str = "") -> None:
    if not cond:
        failures.append(f"{name}: {detail}")
        print(f"drift-guard FAIL: {name} - {detail}")


def suite_checks(test_py: Path) -> int:
    """Run a suite and return its check count (both suites print 'N checks')."""
    out = subprocess.run(
        [sys.executable, str(test_py)], capture_output=True, text=True, timeout=300
    )
    m = re.search(r"(\d+) checks?, 0 failure", out.stdout)
    if not m:
        guard("suite parser", False, f"could not parse result from {test_py}: {out.stdout[-200:]!r}")
        return -1
    return int(m.group(1))


def spec_command_table() -> list[str]:
    """Commands from the spec's CLI command table (section 6.2 only)."""
    m = re.search(r"### 6\.2 Commands(.*?)### 6\.3", SPEC.read_text(encoding="utf-8"), re.S)
    if not m:
        return []
    return re.findall(r"\| `(\w+)` \|", m.group(1))


def main() -> int:
    # 1. README counts vs actual suites
    unit_actual = suite_checks(HERE / "_test_agent_memory.py")
    audit_actual = suite_checks(HERE / "_audit_cli.py")
    readme = README.read_text(encoding="utf-8", errors="replace")
    stated = [int(n) for n in re.findall(r"\((\d+) checks\)", readme)]
    if not stated:
        guard("README test counts", False, "README states no test count - state them explicitly.")
    for n in stated:
        if n not in (unit_actual, audit_actual):
            guard("README test counts", False,
                  f"README states {n} but the suites report unit={unit_actual} audit={audit_actual}")

    # 2. SPEC constants vs implementation (code-form vs prose-form)
    code = CODE.read_text(encoding="utf-8")
    spec = SPEC.read_text(encoding="utf-8")
    code_checks = [
        ("FORMAT_VERSION = 1", r"FORMAT_VERSION = 1"),
        ("MEMORY_PREFIX mem_", r'MEMORY_PREFIX = "mem_"'),
        ("DEFAULT_RECALL_LIMIT = 10", r"DEFAULT_RECALL_LIMIT = 10"),
        ("DEFAULT_SEARCH_LIMIT = 50", r"DEFAULT_SEARCH_LIMIT = 50"),
    ]
    spec_checks = [
        ("format_version is 1", r"format_version.*MUST be `1`", re.S),
        ("mem_<uuid> ids", r"`mem_<uuid", 0),
        ("recall limit 10", r"DEFAULT_RECALL_LIMIT = 10", 0),
        ("search default limit 50", r"Default limit: 50", 0),
        ("exit codes 0/1/2", r"\| `0` \| success", 0),
    ]
    for name, pat in code_checks:
        guard(f"impl: {name}", bool(re.search(pat, code)), f"{pat} not found in agent_memory.py")
    for name, pat, flags in spec_checks:
        guard(f"spec doc: {name}", bool(re.search(pat, spec, flags)),
              f"{pat!r} not stated in V0.1_SPEC.md")

    # 3. SPEC command table vs CLI subcommands
    commands = spec_command_table()
    implemented = re.findall(r'add_parser\("(\w+)"', code)
    for cmd in commands:
        guard(f"spec cmd: {cmd}", cmd in implemented,
              f"command {cmd!r} is in the spec table but not implemented")
    for cmd in implemented:
        guard(f"impl cmd: {cmd}", cmd in commands,
              f"command {cmd!r} is implemented but missing from the spec table")

    # 4. Python >= 3.11 in pyproject
    try:
        with open(PYPROJECT, "rb") as fh:
            meta = tomllib.load(fh)
        requires = meta.get("project", {}).get("requires-python", "")
        guard("python pin", ">=3.11" in requires,
              f"requires-python = {requires!r}, expected >=3.11")
        guard("readme field", meta.get("project", {}).get("readme") == "README.md",
              "pyproject readme field drifted")
        deps = meta.get("project", {}).get("dependencies", [])
        guard("zero deps", deps == [], f"dependencies = {deps!r}")
    except Exception as exc:  # noqa: BLE001
        guard("pyproject parse", False, f"could not parse pyproject.toml: {exc}")

    # 6. EOL discipline: all .py files LF-only
    for py in sorted(HERE.glob("*.py")):
        data = py.read_bytes()
        guard(f"eol: {py.name}", b"\r\n" not in data,
              "file contains CRLF line endings (byte-determinism contract)")

    if failures:
        print(f"drift-guard FAIL: {len(failures)} violation(s)")
        return 1
    print("drift-guard OK: README, SPEC, pyproject, and code agree.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
