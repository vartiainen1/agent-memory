# agent-memory

Local-first knowledge and governance layer for AI coding agents.

agent-memory gives agents persistent project memory while keeping trust,
history, and authority under system and human control. **The AI can use
memory, but the AI must not have unrestricted authority over memory.**

- **Local-first** — no cloud, no accounts, no telemetry, no external LLMs.
- **Structured** — DECISION / ERROR / LESSON / CONSTRAINT / ARCHITECTURE / PATTERN
  memories with typed metadata (provenance, trust, severity, paths, tags).
- **Trusted** — AI/imported knowledge starts `untrusted`; only a human can
  promote trust. Agents can never promote themselves.
- **Auditable** — immutable history (supersede, never rewrite), append-only
  audit log, secrets rejected before storage.
- **Agent- and model-agnostic** — a boring, deterministic CLI any agent can
  drive; works with zero LLM calls.

## Quickstart

```bash
# in your project root
agent-memory init

agent-memory add --type constraint \
  --title "Auth via AuthService" \
  --content "All authentication logic must use AuthService" \
  --paths "src/auth/**"

agent-memory promote <mem_id> --trust approved   # human-authorized

# operator search
agent-memory search "auth"

# agent context assembly (trusted memories only, path-aware)
agent-memory recall "auth" --path src/auth/login.py
```

## Commands

| Command | Purpose |
|---|---|
| `init [--project NAME]` | create the `.agent/` store in the current project |
| `add --type T --title S --content S [...]` | create a memory (flags required; `--allow-secret` for audited override) |
| `list [--type T] [--status S] [--json]` | list memories |
| `show <mem_id> [--json]` | show one memory |
| `search QUERY [--type T] [--status S] [--limit N] [--json]` | operator textual search (limit 50) |
| `recall QUERY [--path P] [--limit N] [--json]` | agent context assembly (limit 10, trusted + active only) |
| `promote <mem_id> --trust verified\|approved` | human trust promotion (never `system`) |
| `supersede <old_id> <new_id>` | bidirectional supersession; no chains in v0.1 |
| `delete <mem_id> [--purge]` | tombstone delete; `--purge` physically removes untrusted only |
| `status [--json]` | store health + counts |

Exit codes: `0` success · `1` runtime error · `2` usage error.
Every command supports `--json` for machine-readable output.

## Storage layout

```
.agent/
├── config.toml      # project, recall_limit, include_untrusted
├── audit.jsonl      # append-only audit trail
└── memory/
    ├── decisions/   # mem_<uuid>.json
    ├── errors/
    ├── lessons/
    ├── constraints/
    ├── architecture/
    └── patterns/
```

Memory files are the authoritative record; there is no index database in
v0.1.

## Security model

- **Trust ladder:** `untrusted` → `verified` → `approved` (`system` is
  application-internal; agents can never create it). No downgrades in v0.1.
- **Secret detection:** deterministic patterns + entropy heuristic run BEFORE
  storage. Default is reject; `--allow-secret` is an explicit, audited override.
- **Immutable history:** superseding sets bidirectional links
  (`supersedes` / `superseded_by`); historical records are retained.
- **Audit:** every mutation and access writes an append-only event.

## Ecosystem role

agent-memory is the persistent knowledge layer of the agent-tool family:

- `agent-error-log` — captures errors
- `agent-decision-log` — captures decisions
- `agent-log-ai` — distills lessons from logs
- **`agent-memory` — persists, organizes, trusts, retrieves, relates knowledge**
- `agent-diff-gate` — enforces constraints against changes

Family-repo import is v0.2; v0.1 ships provenance/fingerprint fields
schema-ready but unpopulated.

## Requirements

- Python >= 3.11 (stdlib only — `tomllib`, `uuid`, `hashlib`, `re`)
- No pip dependencies

## Development

```bash
python _test_agent_memory.py   # unit + CLI integration tests (136 checks)
python _audit_cli.py           # external-API audit, real binary via subprocess (127 checks)
```

`_audit_cli.py` treats the CLI as an external API: every check runs the real
binary in an isolated scratch project and asserts on stdout, stderr, exit
codes and output values — including malformed input, corrupt files, secret
detection, trust boundaries, the `--json` error contract, Unicode, and
byte-identical determinism.

## Status

v0.1 (alpha). `V0.1_SPEC.md` is the implementation contract.

## License

MIT
