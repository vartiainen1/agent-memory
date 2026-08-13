# agent-memory

Local-first knowledge and governance layer for AI coding agents.

[![CI](https://github.com/vartiainen1/agent-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/vartiainen1/agent-memory/actions/workflows/ci.yml)
[![checks on master](https://img.shields.io/github/checks-status/vartiainen1/agent-memory/master)](https://github.com/vartiainen1/agent-memory/actions)
[![release](https://img.shields.io/github/v/release/vartiainen1/agent-memory)](https://github.com/vartiainen1/agent-memory/releases)
[![license](https://img.shields.io/github/license/vartiainen1/agent-memory)](https://github.com/vartiainen1/agent-memory/blob/master/LICENSE)
[![python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-3776AB)](https://github.com/vartiainen1/agent-memory/actions)
[![dependencies-0](https://img.shields.io/badge/dependencies-0-brightgreen)](https://github.com/vartiainen1/agent-memory)
[![Visitors](https://visitor-badge.laobi.icu/badge?page_id=vartiainen1.agent-memory&left_text=Visitors&right_color=2F80ED)](https://github.com/vartiainen1/agent-memory)

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

## Why isn't this just RAG?

RAG retrieves *source text* (docs, code) and lets an LLM do the rest.
agent-memory stores *curated, typed knowledge about the project* — what was
decided, what broke before, what the constraints are — and, critically,
**who is allowed to create, trust, and modify that knowledge**:

- Memories are born `untrusted`; only a human can promote them.
- Nothing is ever rewritten: history is superseded, never edited.
- Every create/access/delete is written to an append-only audit log.
- Secrets are detected and rejected before storage.

A vector database answers *"what text is similar to this query?"* — it has no
opinion on whether the answer is *true*, *trusted*, or *current*. That is the
problem agent-memory is built around.

## Install

Requires Python >= 3.11. Zero runtime dependencies for the core.
The MCP agent interface (v0.3 Tier 1) is an optional extra.

```bash
# from this repository (the current distribution path - not yet on PyPI)
pip install git+https://github.com/vartiainen1/agent-memory.git

# or clone and install locally
git clone https://github.com/vartiainen1/agent-memory.git
cd agent-memory
pip install .

# optional: MCP (Model Context Protocol) agent interface
pip install .[mcp]
```

Verify:

```bash
agent-memory --version   # agent-memory 0.1.0
```

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
| `import PATH --source error-log\|decision-log\|lesson-log\|rule-log` | family import (untrusted, provenance-tracked, idempotent) |
| `supersede <old_id> <new_id>` | bidirectional supersession; a memory supersedes at most one other (linear chains allowed) |
| `delete <mem_id> [--purge]` | tombstone delete; `--purge` physically removes untrusted only |
| `status [--json]` | store health + counts |

Exit codes: `0` success · `1` runtime error · `2` usage error.
Every command supports `--json` for machine-readable output.

## How an AI agent uses it

An agent asks *"what should I know before doing this task?"* — it does **not**
get unrestricted access to the store:

```bash
agent-memory recall "modify authentication" --path src/auth/login.py --json
```

```json
{
  "results": [
    {
      "id": "mem_...",
      "type": "constraint",
      "title": "Authentication must use AuthService",
      "trust": "approved",
      "content": "All authentication logic must use AuthService",
      "paths": ["src/auth/**"]
    }
  ],
  "count": 1
}
```

By default `recall` only returns `active` memories that are **not untrusted**,
so an agent cannot be steered by unverified or superseded knowledge. `promote`
is a human-only command; there is no agent-accessible path to raise trust or
alter history. Everything the agent reads or the system mutates is recorded in
`audit.jsonl`.

`recall` also applies a **relevance floor** (`SCORE_FLOOR_RATIO = 0.25`):
results whose text score is below 25% of the best match are dropped,
unless they match `--path` (explicit intent). `search` stays
inclusive — operators see everything that matches; agents get only
confident context.

`--path` is a **tiered ranking signal**, not an auto-win (v0.3 T2.1): a
memory whose path pattern covers the touched file gets a bonus by
specificity — exact file (10) > bare directory (6) > glob/prefix (3) —
so `src/auth/**` outranks a generally-relevant auth memory when editing
`src/auth/session.py`, while a strong text match without a path can
still outrank a weak-text exact-path match. A bare directory pattern
(`src/auth`) now routes to files under it.

### Family import (cold-start)

A brand-new project can adopt the family's accumulated knowledge instead of
starting empty:

```
agent-memory init
agent-memory import ../agent-error-log --source error-log
agent-memory import ../agent-decision-log --source decision-log
agent-memory import ../agent-log-ai --source lesson-log   # AI-derived lesson drafts
agent-memory import ../agent-log-ai --source rule-log    # numbered engagement rules (constraints)
agent-memory list            # imported memories (untrusted)
agent-memory promote <id> --trust approved   # human-curate what the agent may recall
```

Imported memories are born untrusted — `recall` ignores them until a human
promotes them, so import can never inject unverified context into an agent.
Re-running an import is safe: entries are deduplicated by a canonical
fingerprint (same source entry → same fingerprint, independent of line
endings).

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

## What happens to your data

Everything lives in `.agent/` inside your own project. Nothing is uploaded,
synced, or sent anywhere — agent-memory makes **zero network calls**. It runs
on Python's standard library only. Delete the `.agent/` directory and every
memory is gone; there is no server, no account, and no cloud copy.

## Ecosystem role

agent-memory is the persistent knowledge layer of the agent-tool family:

- `agent-error-log` — captures errors
- `agent-decision-log` — captures decisions
- `agent-log-ai` — distills lessons from logs
- **`agent-memory` — persists, organizes, trusts, retrieves, relates knowledge**
- `agent-diff-gate` — enforces constraints against changes

The family works as a loop: the log repositories record what happened,
agent-memory turns that history into trusted, retrievable knowledge, an
agent consumes it as context before changing code, and agent-diff-gate
enforces the stored constraints against the proposed change:

```
  error-log ──┐
decision-log ─┼──▶ agent-memory ──recall──▶ AI agent ──proposed change──▶ agent-diff-gate
   log-ai ────┘         │  ▲                     │                            │
                        │  └── trusted context    │                            ▼
                        └───────── constraints ───┴─────────────────────▶ PASS / BLOCK
```

(Import of the family logs into agent-memory is v0.2; v0.1 ships
provenance/fingerprint fields schema-ready but unpopulated, and seeding is
manual via `add --provenance import`.)

## Agent interface (MCP, v0.3 Tier 1)

`agent-memory-mcp` is a stdio MCP server that exposes the memory system
to AI coding agents through the official MCP SDK:

- Tools: `memory_recall`, `memory_search`, `memory_get`, `memory_history`,
  `memory_suggest`, `memory_create`, `memory_validate`
- The tool surface IS the permission boundary: `delete` and `promote` are
  never exposed to agents - promotion stays human-only, deletion stays
  CLI-only
- `memory_suggest` returns a validated candidate preview without
  persisting (the AI proposes; the system decides what becomes
  authoritative)
- Every call goes through the core pipeline: validation, secret
  detection (never overridable), trust rules, audit (actor = agent)

```bash
agent-memory-mcp            # stdio server
agent-memory-mcp --list-tools
agent-memory-mcp --version
```

## Requirements

- Python >= 3.11 (stdlib only — `tomllib`, `uuid`, `hashlib`, `re`)
- No pip dependencies for the core; the optional `[mcp]` extra adds the
  official MCP SDK for the agent interface

## Development

```bash
python _test_agent_memory.py   # unit + CLI integration tests (257 checks)
python _audit_cli.py           # external-API audit, real binary via subprocess (127 checks)
python _test_mcp.py            # MCP adapter: permissions, secrets, protocol (63 checks)
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
