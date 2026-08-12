# Changelog

All notable changes to agent-memory are documented here. The version at the
top of this file is the single source of truth for releases (the Release
workflow tags whatever the top `## [X.Y.Z]` header says).

## [Unreleased]

### Added (v0.2 Tier 1, retrieval - EVIDENCE-012/013)

- IDF/common-token downweighting + phrase bonus in deterministic
  search/recall scoring; honest zero results preserved.
- Equal-score tie-break contract tightened to (score, created_at, title,
  id) desc so ordering is stable across store clones (no UUID exposure).

## [0.1.0] - 2026-08-12

### Added

- v0.1 initial implementation of the memory engine:
  - schema (`format_version: 1`, `mem_<uuid>` ids, typed records)
  - per-project `.agent/` storage (git-style walk-up discovery)
  - validation (pinned enums for type/provenance/trust/status/severity)
  - trust lifecycle (`promote`, human-only, never `system`, no downgrades)
  - memory lifecycle (`supersede` bidirectional links, no chains; tombstone
    `delete`; `--purge` untrusted-only)
  - append-only audit log (`MEMORY_CREATED/ACCESSED/REJECTED/SUPERSEDED/
    DELETED`, `TRUST_PROMOTED`, `SECRET_OVERRIDE`)
  - deterministic secret detection (8 pattern families + entropy heuristic;
    `--allow-secret` explicit + audited override)
  - operator `search` (limit 50) and agent `recall` (limit 10, path bonus,
    trusted-only by default)
  - JSON output contract; exit codes 0/1/2; `--json` errors on stdout
  - interactive `add` mode (friction fix) alongside the flag-driven path
- Family parity: `_test_agent_memory.py` (136 checks), `_audit_cli.py`
  external-API audit (127 checks), `_check_drift.py` drift guards, CI
  workflow (3 OS x Python matrix + audit + drift + packaging), release
  workflow (tag + draft from CHANGELOG.md), MIT license.
