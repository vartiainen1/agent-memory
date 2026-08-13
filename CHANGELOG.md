# Changelog

All notable changes to agent-memory are documented here. The version at the
top of this file is the single source of truth for releases (the Release
workflow tags whatever the top `## [X.Y.Z]` header says).

## [Unreleased]
### Added (v0.3 Tier 1, agent interface - MCP)
- `agent_memory_mcp.py`: stdio MCP server (official MCP SDK, optional
  `[mcp]` extra; core stays stdlib-only). Tools: memory_recall,
  memory_search, memory_get, memory_history, memory_suggest,
  memory_create, memory_validate.
- Permission boundary enforced by the tool surface: delete/promote/
  supersede/import are NOT exposed to agent sessions; provenance forced
  to agent; secret detection never overridable; audit actor = agent.
- `memory_suggest` returns a validated candidate preview (no
  persistence) - the AI proposes, the system decides.
- 63-check companion suite (incl. stdio round-trip over the real server);
  `_check_drift.py` now validates the third suite's README count and
  cross-checks the README tool list against ALLOWED_TOOLS.
### Added (v0.2 Tier 2.3, rule-log source - EVIDENCE-023)
- `import --source rule-log`: the numbered RULES OF ENGAGEMENT (sections
  1-6 of agent-log-ai/rules.txt) as `constraint` memories - the T9
  coverage fix (the authoritative layer above the lesson drafts).
### Added (v0.2 Tier 2.3 + 2.2, family import - EVIDENCE-003/016/017)

- `import` command: canonical-fingerprint import of agent-error-log,
  agent-decision-log and agent-log-ai (rules.txt lesson drafts) logs
  (untrusted, provenance-tracked, idempotent, secret-detected,
  SUPERSEDES-wired, dry-run).

### Added (v0.2 Tier 2.1, retrieval - EVIDENCE-010/015)

- Recall-only relevance floor (SCORE_FLOOR_RATIO = 0.25): weak
  tails dropped, path matches exempt, honest zero results
  preserved; search stays inclusive.


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
