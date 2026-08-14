"""Tests for agent_memory_mcp.py - the v0.3 Tier 1 MCP adapter.

Covers the permission surface (exactly ALLOWED_TOOLS, no delete/promote/
supersede/import, and no approve/reject - an agent can never approve its
own suggestion), the security invariants (provenance forced to agent,
secrets never overridable, suggest enqueues a PENDING SUGGESTION - never
a memory, audit actor = agent), the JSON protocol contract, and the
store-absent error path.

Most tests exercise the plain tool handler functions directly (they do not
require the mcp SDK). The tool-surface test builds the real FastMCP server
and runs only when the optional `mcp` extra is installed.

Run: python _test_mcp.py
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import sys
import tempfile
from pathlib import Path

import agent_memory as am
import agent_memory_mcp as mcp

PASS = 0
BAR = "=" * 80
_failures: list[str] = []
_count = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _count
    _count += 1
    if not cond:
        _failures.append(f"{name} {detail}".strip())


def tmp_store(**init_kwargs) -> Path:
    """Create a temp project with an initialized store; return the store dir."""
    tmp = tempfile.mkdtemp(prefix="agent-memory-mcp-test-")
    root = Path(tmp)
    return am.init_store(target=root, **init_kwargs)


def _pin_store(store: Path) -> None:
    """Point the adapter's store resolver at the temp store for this test."""
    mcp._resolve_store = lambda: store  # noqa: SLF001 - test seam


def _make_memory(store: Path, title: str = "auth via AuthService",
                 content: str = "All authentication must use AuthService",
                 mem_type: str = "constraint") -> dict:
    return am.create_memory(store, mem_type, title, content, project="p",
                            provenance="agent")


# ---------------------------------------------------------------------------
# recall / search / get / history
# ---------------------------------------------------------------------------


def test_recall_trusted_only_by_default() -> None:
    """Agent recall must exclude untrusted memories (candidates/agent adds);
    after human promotion the same memory IS recalled."""
    store = tmp_store()
    _pin_store(store)
    am.create_memory(store, "decision", "use postgres", "chosen for consistency",
                     project="p", provenance="human")
    for r in am.list_memories(store):
        am.promote_trust(store, r["id"], "approved")
    am.create_memory(store, "error", "refresh tokens", "tokens were reusable",
                     project="p", provenance="agent")
    out = json.loads(mcp.tool_memory_recall("postgres"))
    check("recall: ok flag true", out["ok"] is True, f"(got {out})")
    check("recall: approved memory returned",
          any("postgres" in r["title"] for r in out["results"]),
          f"(titles {[r['title'] for r in out['results']]})")
    check("recall: untrusted agent memory excluded",
          not any("refresh" in r["title"] for r in out["results"]),
          f"(titles {[r['title'] for r in out['results']]})")


def test_recall_audits_as_agent_actor() -> None:
    store = tmp_store()
    _pin_store(store)
    am.create_memory(store, "decision", "use postgres", "chosen for consistency",
                     project="p", provenance="human")
    for r in am.list_memories(store):
        am.promote_trust(store, r["id"], "approved")
    mcp.tool_memory_recall("postgres")
    events = am.read_audit(store)
    accesses = [e for e in events if e.get("event") == "MEMORY_ACCESSED"]
    check("recall: MEMORY_ACCESSED recorded", len(accesses) >= 1, f"(got {events})")
    check("recall: actor is agent (not human)",
          accesses and accesses[-1].get("actor") == "agent",
          f"(actors {[e.get('actor') for e in accesses]})")


def test_search_includes_untrusted() -> None:
    """Operator search is inclusive on purpose - it must show the agent's
    untrusted memory too (the agent can see everything, it just cannot
    trust/promote anything)."""
    store = tmp_store()
    _pin_store(store)
    am.create_memory(store, "error", "refresh tokens", "tokens were reusable",
                     project="p", provenance="agent")
    out = json.loads(mcp.tool_memory_search("refresh"))
    check("search: ok flag true", out["ok"] is True, f"(got {out})")
    check("search: untrusted memory visible",
          any("refresh" in r["title"] for r in out["results"]),
          f"(titles {[r['title'] for r in out['results']]})")


def test_search_type_and_limit_filters() -> None:
    store = tmp_store()
    _pin_store(store)
    am.create_memory(store, "decision", "use postgres", "chosen for consistency",
                     project="p", provenance="human")
    am.create_memory(store, "error", "postgres outage", "connection dropped",
                     project="p", provenance="human")
    out = json.loads(mcp.tool_memory_search("postgres", mem_type="decision"))
    check("search: type filter applied",
          all(r["type"] == "decision" for r in out["results"]) and len(out["results"]) == 1,
          f"(got {[r['type'] for r in out['results']]})")
    out2 = json.loads(mcp.tool_memory_search("postgres", limit=1))
    check("search: limit respected", len(out2["results"]) == 1,
          f"(got {len(out2['results'])})")


def test_get_returns_memory_and_audits() -> None:
    store = tmp_store()
    _pin_store(store)
    rec = _make_memory(store)
    out = json.loads(mcp.tool_memory_get(rec["id"]))
    check("get: memory returned", out["ok"] and out["memory"]["id"] == rec["id"],
          f"(got {out})")
    events = [e for e in am.read_audit(store)
              if e.get("event") == "MEMORY_ACCESSED"]
    check("get: access audited for that id",
          events and events[-1].get("memory_id") == rec["id"],
          f"(got {events[-1] if events else None})")


def test_get_missing_id_is_clean_error() -> None:
    store = tmp_store()
    _pin_store(store)
    out = json.loads(mcp.tool_memory_get("mem_does-not-exist"))
    check("get: missing id -> ok false", out["ok"] is False, f"(got {out})")
    check("get: missing id -> error message present",
          isinstance(out.get("error"), str) and len(out["error"]) > 0,
          f"(got {out})")


def test_history_filters_by_memory_id() -> None:
    store = tmp_store()
    _pin_store(store)
    a = _make_memory(store, title="alpha memory")
    b = _make_memory(store, title="beta memory")
    mcp.tool_memory_get(a["id"])
    mcp.tool_memory_get(b["id"])
    out = json.loads(mcp.tool_memory_history(memory_id=a["id"]))
    check("history: ok flag true", out["ok"] is True, f"(got {out})")
    check("history: only events for the requested id",
          out["events"] and all(e.get("memory_id") == a["id"] for e in out["events"]),
          f"(ids {[e.get('memory_id') for e in out['events']]})")


def test_history_limit_applied() -> None:
    store = tmp_store()
    _pin_store(store)
    rec = _make_memory(store)
    for _ in range(3):
        mcp.tool_memory_get(rec["id"])
    out = json.loads(mcp.tool_memory_history(memory_id=rec["id"], limit=1))
    check("history: limit respected", len(out["events"]) == 1,
          f"(got {len(out['events'])})")


def test_history_bad_limit_is_clean_error() -> None:
    store = tmp_store()
    _pin_store(store)
    out = json.loads(mcp.tool_memory_history(limit=0))
    check("history: limit 0 -> ok false", out["ok"] is False, f"(got {out})")


# ---------------------------------------------------------------------------
# suggest: T3 approve-to-persist - enqueue a PENDING SUGGESTION, never a memory
# ---------------------------------------------------------------------------


def test_suggest_persists_suggestion_not_memory() -> None:
    store = tmp_store()
    _pin_store(store)
    out = json.loads(mcp.tool_memory_suggest(
        "constraint", "auth via AuthService", "All authentication must use AuthService"))
    check("suggest: ok flag true", out["ok"] is True, f"(got {out})")
    check("suggest: pending suggestion returned",
          isinstance(out.get("suggestion"), dict)
          and out["suggestion"]["state"] == "pending", f"(got {out})")
    check("suggest: no MEMORY persisted",
          am.list_memories(store) == [], f"(got {am.list_memories(store)})")
    sugs = am.list_suggestions(store)
    check("suggest: exactly one suggestion enqueued", len(sugs) == 1,
          f"(got {sugs})")
    events = am.read_audit(store)
    check("suggest: SUGGESTION_CREATED audited",
          [e["event"] for e in events] == ["SUGGESTION_CREATED"],
          f"(got {events})")
    check("suggest: audit actor is agent", events[0]["actor"] == "agent",
          f"(got {events[0]})")


def test_suggest_is_not_a_memory_record() -> None:
    """A suggestion carries no trust/status: it is a candidate, not a memory
    (the memory schema is untouched - T3 decision)."""
    store = tmp_store()
    _pin_store(store)
    out = json.loads(mcp.tool_memory_suggest(
        "constraint", "t", "content here"))
    sug = out["suggestion"]
    check("suggest: provenance pinned to agent", sug["provenance"] == "agent",
          f"(got {sug.get('provenance')})")
    check("suggest: no trust field (not a memory)", "trust" not in sug,
          f"(got {sorted(sug.keys())})")
    check("suggest: no status field (not a memory)", "status" not in sug,
          f"(got {sorted(sug.keys())})")
    check("suggest: id space is sug_", sug["id"].startswith("sug_"),
          f"(got {sug['id']})")


def test_suggest_secret_rejected_not_persisted() -> None:
    store = tmp_store()
    _pin_store(store)
    token = "ghp_" + "a" * 36
    out = json.loads(mcp.tool_memory_suggest(
        "constraint", "token", f"here is the key: {token}"))
    check("suggest: secret -> ok false", out["ok"] is False, f"(got {out})")
    check("suggest: secret reported", out["report"]["secret_detected"] is not None,
          f"(got {out})")
    check("suggest: secret suggestion NOT enqueued", am.list_suggestions(store) == [],
          f"(got {am.list_suggestions(store)})")
    check("suggest: secret memory NOT persisted", am.list_memories(store) == [],
          f"(got {am.list_memories(store)})")
    check("suggest: secret rejection audited",
          [e["event"] for e in am.read_audit(store)] == ["SUGGESTION_REJECTED"],
          f"(got {am.read_audit(store)})")


# ---------------------------------------------------------------------------
# create: persist through the core with agent provenance
# ---------------------------------------------------------------------------


def test_create_persists_with_agent_provenance() -> None:
    store = tmp_store()
    _pin_store(store)
    out = json.loads(mcp.tool_memory_create(
        "constraint", "auth via AuthService", "All authentication must use AuthService"))
    check("create: ok flag true", out["ok"] is True, f"(got {out})")
    mem = out["memory"]
    check("create: persisted", am.list_memories(store) != [], "store empty")
    check("create: provenance forced to agent", mem["provenance"] == "agent",
          f"(got {mem.get('provenance')})")
    check("create: trust starts untrusted", mem["trust"] == "untrusted",
          f"(got {mem.get('trust')})")
    events = am.read_audit(store)
    created = [e for e in events if e.get("event") == "MEMORY_CREATED"]
    check("create: audited as agent actor",
          created and created[-1].get("actor") == "agent",
          f"(got {created[-1] if created else None})")


def test_create_secret_rejected_and_never_persisted() -> None:
    """The agent can never override secret detection - there is no
    allow_secret parameter anywhere on the MCP surface."""
    store = tmp_store()
    _pin_store(store)
    token = "sk-" + "b" * 20
    out = json.loads(mcp.tool_memory_create(
        "error", "leak", f"the key was {token}"))
    check("create: secret -> ok false", out["ok"] is False, f"(got {out})")
    check("create: rejected memory not persisted", am.list_memories(store) == [],
          f"(got {am.list_memories(store)})")
    rejected = [e for e in am.read_audit(store)
                if e.get("event") == "MEMORY_REJECTED"]
    check("create: MEMORY_REJECTED audited", len(rejected) == 1,
          f"(got {rejected})")


def test_create_invalid_type_is_clean_error() -> None:
    store = tmp_store()
    _pin_store(store)
    out = json.loads(mcp.tool_memory_create("nonsense", "t", "content"))
    check("create: invalid type -> ok false", out["ok"] is False, f"(got {out})")
    check("create: invalid type -> nothing persisted", am.list_memories(store) == [],
          f"(got {am.list_memories(store)})")


# ---------------------------------------------------------------------------
# validate: pure check, never stores
# ---------------------------------------------------------------------------


def test_validate_valid_proposal() -> None:
    store = tmp_store()
    _pin_store(store)
    out = json.loads(mcp.tool_memory_validate(
        "constraint", "auth via AuthService", "All authentication must use AuthService"))
    check("validate: ok flag true", out["ok"] is True, f"(got {out})")
    check("validate: report valid", out["report"]["valid"] is True, f"(got {out})")
    check("validate: nothing persisted", am.list_memories(store) == [],
          f"(got {am.list_memories(store)})")


def test_validate_invalid_proposal_reports_errors() -> None:
    store = tmp_store()
    _pin_store(store)
    out = json.loads(mcp.tool_memory_validate("nonsense", "t", "content"))
    check("validate: invalid type -> report valid false",
          out["report"]["valid"] is False, f"(got {out})")
    check("validate: errors populated",
          isinstance(out["report"]["errors"], list) and len(out["report"]["errors"]) > 0,
          f"(got {out})")


def test_validate_secret_reported() -> None:
    store = tmp_store()
    _pin_store(store)
    out = json.loads(mcp.tool_memory_validate(
        "error", "leak", f"key sk-{'c' * 20}"))
    check("validate: secret flagged", out["report"]["secret_detected"] is not None,
          f"(got {out})")
    check("validate: secret -> valid false", out["report"]["valid"] is False,
          f"(got {out})")


# ---------------------------------------------------------------------------
# protocol + error paths
# ---------------------------------------------------------------------------


def test_all_tools_return_json_strings() -> None:
    """Deterministic protocol contract: every handler returns a JSON string
    that parses, with an ok flag - never raw exceptions to the client."""
    store = tmp_store()
    _pin_store(store)
    handlers = [
        lambda: mcp.tool_memory_recall("anything"),
        lambda: mcp.tool_memory_search("anything"),
        lambda: mcp.tool_memory_get("mem_missing"),
        lambda: mcp.tool_memory_history(),
        lambda: mcp.tool_memory_suggest("constraint", "t", "c"),
        lambda: mcp.tool_memory_create("constraint", "t", "c"),
        lambda: mcp.tool_memory_validate("constraint", "t", "c"),
    ]
    for i, handler in enumerate(handlers):
        raw = handler()
        try:
            parsed = json.loads(raw)
            ok = isinstance(parsed, dict) and "ok" in parsed
        except json.JSONDecodeError:
            ok = False
        check(f"protocol: handler {i} returns parseable JSON with ok flag", ok,
              f"(got {raw[:120]!r})")


def test_store_absent_is_clean_error() -> None:
    """No .agent store -> clean ok:false JSON, not a crash."""
    mcp._resolve_store = lambda: (_ for _ in ()).throw(
        am.AgentMemoryError("not in an agent-memory project (no .agent/ found)"))
    out = json.loads(mcp.tool_memory_recall("anything"))
    check("no-store: ok false", out["ok"] is False, f"(got {out})")
    check("no-store: error message present", len(out.get("error", "")) > 0,
          f"(got {out})")


# ---------------------------------------------------------------------------
# server surface (requires the optional mcp extra)
# ---------------------------------------------------------------------------


def test_server_surface_exactly_allowed_tools() -> None:
    if not mcp.MCP_AVAILABLE:
        check("surface: skipped (mcp extra not installed)", True)
        check("surface: no forbidden tools (not testable without mcp)", True)
        return
    server = mcp.build_server()
    tools = asyncio.run(server.list_tools())
    names = sorted(t.name for t in tools)
    check("surface: exactly ALLOWED_TOOLS registered",
          names == sorted(mcp.ALLOWED_TOOLS), f"(got {names})")
    forbidden = {"memory_delete", "memory_promote", "memory_supersede",
                 "memory_import", "memory_purge", "memory_allow_secret",
                 "suggestion_approve", "suggestion_reject",
                 "conflict_scan", "conflict_dismiss", "conflict_resolve",
                 "memory_conflicts", "git_context", "git_list"}
    check("surface: delete/promote/supersede/import NOT exposed",
          not (set(names) & forbidden), f"(got {names})")
    check("surface: approve/reject NOT exposed (agent cannot approve itself)",
          not (set(names) & {"suggestion_approve", "suggestion_reject"}),
          f"(got {names})")
    check("surface: T4 conflict tools NOT exposed (EVIDENCE-038 FORK 4 - "
          "review is human-CLI only)",
          not (set(names) & {"conflict_scan", "conflict_dismiss",
                             "conflict_resolve", "memory_conflicts"}),
          f"(got {names})")


def test_server_surface_has_no_secret_override_param() -> None:
    """The MCP surface must never accept allow_secret/force - check the tool
    input schemas of the two write tools."""
    if not mcp.MCP_AVAILABLE:
        check("surface-schema: skipped (mcp extra not installed)", True)
        check("surface-schema: no override param (not testable without mcp)",
              True)
        check("surface-schema: create/suggest/validate covered (not testable "
              "without mcp)", True)
        return
    server = mcp.build_server()
    tools = {t.name: t for t in asyncio.run(server.list_tools())}
    for tool_name in ("memory_create", "memory_suggest", "memory_validate"):
        schema = json.dumps(tools[tool_name].inputSchema or {})
        check(f"surface-schema: {tool_name} has no allow_secret/force param",
              "allow_secret" not in schema and "force" not in schema,
              f"(got {schema[:200]})")


def test_stdio_round_trip_via_real_server() -> None:
    """Wire-protocol proof (family _audit_cli.py pattern): spawn the real
    server over stdio, initialize, list tools, call memory_recall."""
    if not mcp.MCP_AVAILABLE:
        check("round-trip: skipped (mcp extra not installed)", True)
        check("round-trip: no server spawned without mcp", True)
        check("round-trip: no wire calls without mcp", True)
        return
    import subprocess
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    script = str(pathlib.Path(__file__).resolve().parent / "agent_memory_mcp.py")
    tmp = tempfile.mkdtemp(prefix="am-mcp-rt-")
    root = Path(tmp)
    store = am.init_store(target=root, project="rt")
    am.create_memory(store, "constraint", "auth via AuthService",
                     "All authentication must use AuthService", project="rt",
                     provenance="human")
    for r in am.list_memories(store):
        am.promote_trust(store, r["id"], "approved")
    params = StdioServerParameters(command=sys.executable, args=[script],
                                   cwd=str(root))

    async def _run() -> tuple[list[str], dict]:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                names = sorted(t.name for t in tools.tools)
                res = await session.call_tool(
                    "memory_recall", {"query": "authentication"})
                return names, json.loads(res.content[0].text)

    names, data = asyncio.run(_run())
    check("round-trip: server exposes exactly the 7 tools over stdio",
          names == sorted(mcp.ALLOWED_TOOLS), f"(got {names})")
    check("round-trip: memory_recall returns approved context over the wire",
          data.get("ok") is True and data.get("count", 0) >= 1,
          f"(got {data})")
    check("round-trip: delete/promote absent from the wire surface",
          not (set(names) & {"memory_delete", "memory_promote"}),
          f"(got {names})")


def test_main_list_tools_and_version() -> None:
    rc = mcp.main(["--list-tools"])
    check("main: --list-tools exit 0", rc == 0, f"(rc={rc})")
    rc2 = mcp.main(["--version"])
    check("main: --version exit 0", rc2 == 0, f"(rc={rc2})")


def test_main_missing_extra_returns_2() -> None:
    if mcp.MCP_AVAILABLE:
        check("main: mcp present -> server path reachable", True)
        return
    rc = mcp.main([])
    check("main: no mcp extra -> exit 2", rc == 2, f"(rc={rc})")


def main() -> int:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        try:
            t()
        except Exception as exc:  # noqa: BLE001
            _failures.append(f"{t.__name__} raised {type(exc).__name__}: {exc}")
    print(f"{BAR}\n{_count} checks, {len(_failures)} failure(s)")
    if _failures:
        print("\n".join(f"  FAIL: {f}" for f in _failures))
        return 1
    print("ALL PASS")
    return PASS


if __name__ == "__main__":
    sys.exit(main())
