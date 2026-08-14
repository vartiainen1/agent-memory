#!/usr/bin/env python3
"""agent_memory_mcp.py - MCP (Model Context Protocol) adapter for agent-memory.

v0.3 Tier 1 (roadmap locked 2026-08-13): a stdio MCP server that lets an AI
agent call the memory system through the official MCP SDK, with the core
security model enforced on the server side.

Architecture (reviewer-locked):
    MCP client
        | stdio
        v
    MCP adapter (this module) - the tool surface IS the permission boundary
        | calls core functions only
        v
    agent_memory.py (stdlib-only core: validation, permissions, trust,
                     secret detection, audit, storage)

The adapter NEVER bypasses the core security model. It never reads or writes
memory files directly; every operation goes through the core pipeline
(validation -> secret detection -> trust rules -> storage -> audit).

Rules enforced here, never in the agent's control:    - Tool surface (exactly these 7): memory_recall, memory_search,
    memory_get, memory_history, memory_suggest, memory_create,
    memory_validate. (T3 approve/reject stay CLI-only - the agent can
    propose, never approve.)
    NOT exposed to agent sessions: memory_delete, memory_promote, supersede,
    import, purge.
  - provenance is forced to "agent"; actor is forced to "agent" (the audit
    trail can tell agent actions from human CLI actions).
  - allow_secret is always False - an agent can never override secret
    detection.
  - memory_suggest ENQUEUES a validated, secret-screened PENDING SUGGESTION
    (T3 approve-to-persist loop, EVIDENCE-034): the AI proposes, a HUMAN
    approves. A suggestion is not a memory - it never enters recall/trust/
    lifecycle until approved. approval/rejection are CLI-only and never
    exposed on this surface, so an agent can never approve its own
    suggestion. memory_create remains a documented lower-level escape hatch
    (agent-created untrusted memory, promoted by humans).

The core module stays stdlib-only; `mcp` is an optional extra:
    pip install agent-memory[mcp]

Run:
    agent-memory-mcp                 # stdio server (default)
    agent-memory-mcp --list-tools    # print the agent tool surface, exit
    agent-memory-mcp --version
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import agent_memory as am

try:
    from mcp.server.fastmcp import FastMCP

    MCP_AVAILABLE = True
except ImportError:  # optional extra not installed; tools are still testable
    FastMCP = None  # type: ignore[assignment,misc]
    MCP_AVAILABLE = False

# The agent's entire permission set. Anything not listed here (delete,
# promote, supersede, import, purge, allow_secret, actor choice) is refused
# by construction - it is simply not registered.
ALLOWED_TOOLS = (
    "memory_recall",
    "memory_search",
    "memory_get",
    "memory_history",
    "memory_suggest",
    "memory_create",
    "memory_validate",
)

VERSION = am.VERSION  # single source of truth: the core constant


def _resolve_store() -> Path:
    """Locate the .agent store from the server's working directory (CLI parity)."""
    return am.find_store()


def _project_name(store: Path) -> str:
    """Project name = the directory the store lives in (CLI parity)."""
    return store.parent.name


def _candidate(
    store: Path,
    mem_type: str,
    title: str,
    content: str,
    tags: list[str] | None,
    paths: list[str] | None,
    severity: str,
) -> dict:
    """Build the record that WOULD be stored, exactly as the core shapes it.

    provenance is hard-coded to "agent" - the agent cannot claim to be a
    human, an import, or the system. trust starts "untrusted"; promotion is
    human-only and not exposed here.
    """
    now = am.now_utc()
    return {
        "format_version": am.FORMAT_VERSION,
        "id": am.new_id(),
        "type": mem_type,
        "title": title,
        "content": content,
        "project": _project_name(store),
        "provenance": "agent",
        "trust": "untrusted",
        "status": "active",
        "severity": severity,
        "tags": tags or [],
        "paths": paths or [],
        "created_at": now,
        "updated_at": now,
        "source": None,
        "supersedes": None,
        "superseded_by": None,
        "deleted_at": None,
        "deleted_by": None,
    }


def _check_candidate(store: Path, record: dict) -> dict:
    """Validation + secret scan for a candidate record.

    Returns a report dict; never raises for expected failures. Mirrors the
    core pipeline so `suggest`/`validate` see exactly what `create` enforces.
    """
    errors: list[str] = []
    try:
        am.validate_memory(record)
    except am.UsageError as exc:
        errors.append(str(exc))
    detected = am.detect_secret(record["title"], record["content"])
    report: dict = {
        "valid": not errors and detected is None,
        "errors": errors,
        "secret_detected": detected,
    }
    return report


# ---------------------------------------------------------------------------
# Tool handlers. Plain functions returning JSON strings (deterministic
# protocol); each is fully testable WITHOUT the mcp SDK installed. The
# registered surface above is enforced by build_server().
# ---------------------------------------------------------------------------


def tool_memory_recall(
    query: str, path: str | None = None, limit: int | None = None
) -> str:
    """Agent recall: relevant, trusted context for the current task."""
    try:
        store = _resolve_store()
        results = am.recall_memories(store, query, path=path, limit=limit)
        am.append_audit(store, "MEMORY_ACCESSED", None, "agent",
                        {"command": "recall", "count": len(results)})
        payload: dict = {"ok": True, "count": len(results), "results": results}
    except (am.AgentMemoryError, am.UsageError, OSError, ValueError) as exc:
        payload = {"ok": False, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001 - protocol contract: never leak
        payload = {"ok": False,
                   "error": f"internal error: {type(exc).__name__}: {exc}"}
    return json.dumps(payload, ensure_ascii=False, indent=2)


def tool_memory_search(
    query: str,
    mem_type: str | None = None,
    status: str | None = None,
    limit: int | None = None,
) -> str:
    """Operator-style textual search over the store (includes untrusted)."""
    try:
        store = _resolve_store()
        if limit is None:
            limit = am.DEFAULT_SEARCH_LIMIT  # core default, same as the CLI
        results = am.search_memories(store, query, mem_type=mem_type,
                                     status=status, limit=limit)
        am.append_audit(store, "MEMORY_ACCESSED", None, "agent",
                        {"command": "search", "count": len(results)})
        payload = {"ok": True, "count": len(results), "results": results}
    except (am.AgentMemoryError, am.UsageError, OSError, ValueError) as exc:
        payload = {"ok": False, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001 - protocol contract: never leak
        payload = {"ok": False,
                   "error": f"internal error: {type(exc).__name__}: {exc}"}
    return json.dumps(payload, ensure_ascii=False, indent=2)


def tool_memory_get(memory_id: str) -> str:
    """Fetch one memory by id (audited as MEMORY_ACCESSED)."""
    try:
        store = _resolve_store()
        record = am.load_memory(store, memory_id)
        am.append_audit(store, "MEMORY_ACCESSED", memory_id, "agent", {})
        payload = {"ok": True, "memory": record}
    except (am.AgentMemoryError, am.UsageError, OSError, ValueError) as exc:
        payload = {"ok": False, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001 - protocol contract: never leak
        payload = {"ok": False,
                   "error": f"internal error: {type(exc).__name__}: {exc}"}
    return json.dumps(payload, ensure_ascii=False, indent=2)


def tool_memory_history(memory_id: str | None = None, limit: int | None = None) -> str:
    """Audit history for one memory, or the whole store when no id is given."""
    try:
        store = _resolve_store()
        events = am.read_audit(store)
        if memory_id is not None:
            events = [e for e in events if e.get("memory_id") == memory_id]
        if limit is not None:
            if not isinstance(limit, int) or limit < 1:
                raise am.UsageError("limit must be a positive integer")
            events = events[:limit]
        payload = {"ok": True, "count": len(events), "events": events}
    except (am.AgentMemoryError, am.UsageError, OSError, ValueError) as exc:
        payload = {"ok": False, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001 - protocol contract: never leak
        payload = {"ok": False,
                   "error": f"internal error: {type(exc).__name__}: {exc}"}
    return json.dumps(payload, ensure_ascii=False, indent=2)


def tool_memory_suggest(
    mem_type: str,
    title: str,
    content: str,
    tags: list[str] | None = None,
    paths: list[str] | None = None,
    severity: str = "normal",
) -> str:
    """Propose a memory: enqueue a validated, secret-screened PENDING
    SUGGESTION (T3 approve-to-persist loop).

    The AI proposes; the SYSTEM decides. A suggestion is NOT a memory: it
    carries no trust/status, never enters recall or the memory lifecycle,
    and only a human can convert it into a memory (CLI: `suggestions
    approve <id> --trust verified|approved` - approval is never exposed on
    this surface, so an agent cannot approve its own suggestion). Secrets
    are rejected BEFORE the suggestion is written - rejected material
    (including any secret content) never travels back out.
    """
    try:
        store = _resolve_store()
        suggestion = am.propose_suggestion(
            store, mem_type, title, content,
            project=_project_name(store),
            severity=severity, tags=tags, paths=paths,
            actor="agent",
        )
        payload = {"ok": True, "suggestion": suggestion,
                   "note": "pending suggestion - a human must approve it "
                           "before it becomes memory (CLI: suggestions "
                           "approve <id> --trust verified|approved); approval "
                           "is never exposed to agents"}
    except (am.AgentMemoryError, am.UsageError, OSError, ValueError) as exc:
        report = None
        try:
            store = _resolve_store()
            record = _candidate(store, mem_type, title, content, tags, paths, severity)
            report = _check_candidate(store, record)
        except Exception:  # noqa: BLE001 - report is best-effort on failure
            report = None
        payload = {"ok": False, "error": str(exc), "report": report}
    except Exception as exc:  # noqa: BLE001 - protocol contract: never leak
        payload = {"ok": False,
                   "error": f"internal error: {type(exc).__name__}: {exc}"}
    return json.dumps(payload, ensure_ascii=False, indent=2)


def tool_memory_create(
    mem_type: str,
    title: str,
    content: str,
    tags: list[str] | None = None,
    paths: list[str] | None = None,
    severity: str = "normal",
) -> str:
    """Persist a validated memory through the core (provenance=agent,
    trust=untrusted, secret detection enforced - an agent can never override
    --allow-secret)."""
    try:
        store = _resolve_store()
        record = am.create_memory(
            store, mem_type, title, content,
            project=_project_name(store),
            provenance="agent",
            severity=severity,
            tags=tags,
            paths=paths,
            allow_secret=False,
            actor="agent",
        )
        payload = {"ok": True, "memory": record}
    except (am.AgentMemoryError, am.UsageError, OSError, ValueError) as exc:
        payload = {"ok": False, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001 - protocol contract: never leak
        payload = {"ok": False,
                   "error": f"internal error: {type(exc).__name__}: {exc}"}
    return json.dumps(payload, ensure_ascii=False, indent=2)


def tool_memory_validate(
    mem_type: str,
    title: str,
    content: str,
    tags: list[str] | None = None,
    paths: list[str] | None = None,
    severity: str = "normal",
) -> str:
    """Validate a proposed memory WITHOUT storing anything. Returns a
    {valid, errors, secret_detected} report."""
    try:
        store = _resolve_store()
        record = _candidate(store, mem_type, title, content, tags, paths, severity)
        report = _check_candidate(store, record)
        report["candidate"] = record
        payload = {"ok": True, "report": report}
    except (am.AgentMemoryError, am.UsageError, OSError, ValueError) as exc:
        payload = {"ok": False, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001 - protocol contract: never leak
        payload = {"ok": False,
                   "error": f"internal error: {type(exc).__name__}: {exc}"}
    return json.dumps(payload, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Server assembly + entry point (requires the optional mcp extra)
# ---------------------------------------------------------------------------


def build_server() -> FastMCP:
    """Register exactly the ALLOWED_TOOLS surface on a stdio FastMCP server."""
    if not MCP_AVAILABLE:
        raise am.AgentMemoryError(
            "the mcp extra is not installed; run: pip install agent-memory[mcp]"
        )
    mcp = FastMCP("agent-memory")
    mcp.add_tool(tool_memory_recall, name="memory_recall")
    mcp.add_tool(tool_memory_search, name="memory_search")
    mcp.add_tool(tool_memory_get, name="memory_get")
    mcp.add_tool(tool_memory_history, name="memory_history")
    mcp.add_tool(tool_memory_suggest, name="memory_suggest")
    mcp.add_tool(tool_memory_create, name="memory_create")
    mcp.add_tool(tool_memory_validate, name="memory_validate")
    return mcp


def main(argv: list[str] | None = None) -> int:
    """Entry point: `agent-memory-mcp` -> stdio server (exit 0/2 contract)."""
    args = list(sys.argv[1:] if argv is None else argv)
    if "--version" in args:
        print(f"agent-memory-mcp {VERSION}")
        return 0
    if "--list-tools" in args:
        for name in ALLOWED_TOOLS:
            print(name)
        return 0
    if not MCP_AVAILABLE:
        print("agent-memory-mcp: error: the mcp extra is not installed; "
              "run: pip install agent-memory[mcp]", file=sys.stderr)
        return 2
    build_server().run(transport="stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
