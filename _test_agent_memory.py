"""Tests for agent_memory.py - schema, storage, validation, trust/lifecycle,
audit, search, recall, secrets, CLI.

Run: python _test_agent_memory.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import agent_memory as am

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
    tmp = tempfile.mkdtemp(prefix="agent-memory-test-")
    root = Path(tmp)
    store = am.init_store(target=root, **init_kwargs)
    return store


# --------------------------------------------------------------------------
# 1. Schema / validation
# --------------------------------------------------------------------------

def test_schema_validation() -> None:
    store = tmp_store()
    rec = am.create_memory(store, "decision", "Use Postgres", "Because ACID", project="p")
    check("schema: valid record passes", rec["format_version"] == 1)
    check("schema: id is mem_<uuid>", rec["id"].startswith("mem_") and len(rec["id"]) > 20)

    bad = dict(rec)
    bad["format_version"] = 2
    try:
        am.validate_memory(bad)
        check("schema: format_version 2 rejected", False)
    except am.UsageError:
        check("schema: format_version 2 rejected", True)

    for field in ("type", "trust", "status", "severity", "provenance"):
        bad = dict(rec)
        bad[field] = "not-a-value"
        try:
            am.validate_memory(bad)
            check(f"schema: bad {field} rejected", False)
        except am.UsageError:
            check(f"schema: bad {field} rejected", True)

    bad = dict(rec)
    bad["title"] = "   "
    try:
        am.validate_memory(bad)
        check("schema: blank title rejected", False)
    except am.UsageError:
        check("schema: blank title rejected", True)

    bad = dict(rec)
    bad["created_at"] = "not-a-timestamp"
    try:
        am.validate_memory(bad)
        check("schema: bad created_at rejected", False)
    except am.UsageError:
        check("schema: bad created_at rejected", True)


def test_required_fields() -> None:
    store = tmp_store()
    rec = am.create_memory(store, "error", "Refresh token reused", "It happened", project="p")
    for f in am.REQUIRED_FIELDS:
        check(f"schema: required field {f} present", f in rec)
    for f in am.OPTIONAL_FIELDS:
        check(f"schema: optional field {f} present", f in rec)
    check("schema: source is null in v0.1", rec["source"] is None)
    check("schema: born untrusted", rec["trust"] == "untrusted")
    check("schema: born active", rec["status"] == "active")


# --------------------------------------------------------------------------
# 2. Storage / init / discovery / config
# --------------------------------------------------------------------------

def test_init_and_discovery() -> None:
    tmp = tempfile.mkdtemp(prefix="agent-memory-test-")
    root = Path(tmp)
    (root / "sub" / "deep").mkdir(parents=True)
    store = am.init_store(target=root, project="myproj")
    check("storage: .agent exists", (root / ".agent").is_dir())
    check("storage: type dirs created", all((store / "memory" / t).is_dir() for t in am.MEMORY_TYPES))
    check("storage: config.toml written", (store / "config.toml").exists())
    check("storage: audit.jsonl written", (store / "audit.jsonl").exists())

    cfg = am.load_config(store)
    check("storage: config project", cfg["project"] == "myproj")
    check("storage: config recall_limit default", cfg["recall_limit"] == 10)

    # Discovery from a nested dir walks up.
    found = am.find_store(root / "sub" / "deep")
    check("storage: discovery walks up", found == store)

    # Not-initialized error.
    other = Path(tempfile.mkdtemp(prefix="agent-memory-test-"))
    try:
        am.find_store(other)
        check("storage: missing store raises", False)
    except am.AgentMemoryError as exc:
        check("storage: missing store raises", "agent-memory init" in str(exc))


def test_reinit_refused() -> None:
    store = tmp_store()
    try:
        am.init_store(target=store.parent)
        check("storage: re-init refused", False)
    except am.AgentMemoryError:
        check("storage: re-init refused", True)


def test_windows_eol_determinism() -> None:
    """Memory + audit files must be LF on every platform (byte-identical contract)."""
    store = tmp_store()
    rec = am.create_memory(store, "decision", "EOL check", "line one\nline two", project="p")
    data = (store / "memory" / "decision" / f"{rec['id']}.json").read_bytes()
    check("eol: memory file uses LF only", b"\r\n" not in data and data.count(b"\n") >= 3)
    audit = (store / "audit.jsonl").read_bytes()
    check("eol: audit uses LF only", b"\r\n" not in audit)


# --------------------------------------------------------------------------
# 3. CRUD + lifecycle
# --------------------------------------------------------------------------

def test_crud() -> None:
    store = tmp_store()
    rec = am.create_memory(store, "constraint", "Auth via AuthService", "All auth must go through AuthService", project="p")
    loaded = am.load_memory(store, rec["id"])
    check("crud: load by id", loaded["id"] == rec["id"])
    check("crud: loaded fields", loaded["title"] == "Auth via AuthService")

    records = am.list_memories(store)
    check("crud: list finds one", len(records) == 1)

    try:
        am.load_memory(store, "mem_nonexistent-0000")
        check("crud: missing id raises", False)
    except am.AgentMemoryError:
        check("crud: missing id raises", True)


def test_corrupt_memory_file() -> None:
    store = tmp_store()
    rec = am.create_memory(store, "decision", "Corrupt me", "content", project="p")
    path = am.find_memory_path(store, rec["id"])
    path.write_text("not json {", encoding="utf-8")
    try:
        am.load_memory(store, rec["id"])
        check("corrupt: load raises", False)
    except am.AgentMemoryError as exc:
        check("corrupt: load raises", "corrupt" in str(exc).lower())


def test_supersede() -> None:
    store = tmp_store()
    old = am.create_memory(store, "decision", "Use Redis", "caching", project="p")
    new = am.create_memory(store, "decision", "Remove Redis", "no longer needed", project="p")
    am.supersede(store, old["id"], new["id"])

    old_loaded = am.load_memory(store, old["id"])
    new_loaded = am.load_memory(store, new["id"])
    check("lifecycle: old superseded", old_loaded["status"] == "superseded")
    check("lifecycle: old.superseded_by", old_loaded["superseded_by"] == new["id"])
    check("lifecycle: new.supersedes", new_loaded["supersedes"] == old["id"])
    check("lifecycle: new still active", new_loaded["status"] == "active")

    events = am.read_audit(store)
    check("audit: MEMORY_SUPERSEDED written", any(e["event"] == "MEMORY_SUPERSEDED" for e in events))

    # Old is already superseded -> reject.
    third = am.create_memory(store, "decision", "Third", "third", project="p")
    try:
        am.supersede(store, old["id"], third["id"])
        check("lifecycle: superseded old rejected", False)
    except am.AgentMemoryError:
        check("lifecycle: superseded old rejected", True)

    # No chains: a memory that already supersedes another cannot supersede a
    # second one -> supersede(third, new) is the chain case (new already
    # supersedes old).
    try:
        am.supersede(store, third["id"], new["id"])
        check("lifecycle: chain rejected", False)
    except am.AgentMemoryError:
        check("lifecycle: chain rejected", True)


def test_delete_tombstone_and_purge() -> None:
    store = tmp_store()
    rec = am.create_memory(store, "lesson", "Lesson one", "learned something", project="p")
    am.delete_memory(store, rec["id"])
    loaded = am.load_memory(store, rec["id"])
    check("lifecycle: tombstone status", loaded["status"] == "deleted")
    check("lifecycle: deleted_at set", loaded["deleted_at"] is not None)
    check("lifecycle: record still on disk", am.find_memory_path(store, rec["id"]) is not None)

    events = am.read_audit(store)
    check("audit: MEMORY_DELETED written", any(e["event"] == "MEMORY_DELETED" for e in events))

    # Purge only untrusted.
    rec2 = am.create_memory(store, "lesson", "Lesson two", "another", project="p")
    am.promote_trust(store, rec2["id"], "approved")
    try:
        am.delete_memory(store, rec2["id"], purge=True)
        check("lifecycle: purge approved rejected", False)
    except am.AgentMemoryError:
        check("lifecycle: purge approved rejected", True)

    rec3 = am.create_memory(store, "lesson", "Lesson three", "third", project="p")
    am.delete_memory(store, rec3["id"], purge=True)
    check("lifecycle: purge untrusted removes file", am.find_memory_path(store, rec3["id"]) is None)


# --------------------------------------------------------------------------
# 4. Trust
# --------------------------------------------------------------------------

def test_trust_transitions() -> None:
    store = tmp_store()
    rec = am.create_memory(store, "decision", "D", "d", project="p")
    am.promote_trust(store, rec["id"], "verified")
    check("trust: untrusted->verified", am.load_memory(store, rec["id"])["trust"] == "verified")
    am.promote_trust(store, rec["id"], "approved")
    check("trust: verified->approved", am.load_memory(store, rec["id"])["trust"] == "approved")

    events = am.read_audit(store)
    check("audit: TRUST_PROMOTED written", sum(1 for e in events if e["event"] == "TRUST_PROMOTED") == 2)

    rec2 = am.create_memory(store, "decision", "E", "e", project="p")
    am.promote_trust(store, rec2["id"], "approved")
    check("trust: single jump untrusted->approved", am.load_memory(store, rec2["id"])["trust"] == "approved")

    # Never to system via CLI.
    try:
        am.promote_trust(store, rec2["id"], "system")
        check("trust: system promotion rejected", False)
    except am.UsageError:
        check("trust: system promotion rejected", True)

    # No downgrade.
    try:
        am.promote_trust(store, rec["id"], "verified")  # from approved
        check("trust: downgrade rejected", False)
    except am.AgentMemoryError:
        check("trust: downgrade rejected", True)

    # Promote on a deleted memory is rejected.
    rec3 = am.create_memory(store, "decision", "F", "f", project="p")
    am.delete_memory(store, rec3["id"])
    try:
        am.promote_trust(store, rec3["id"], "verified")
        check("trust: promote deleted rejected", False)
    except am.AgentMemoryError:
        check("trust: promote deleted rejected", True)


def test_provenance_system_rejected() -> None:
    store = tmp_store()
    try:
        am.create_memory(store, "decision", "Sys", "x", project="p", provenance="system")
        check("trust: provenance system rejected", False)
    except am.UsageError:
        check("trust: provenance system rejected", True)


# --------------------------------------------------------------------------
# 5. Secrets
# --------------------------------------------------------------------------

def test_secret_detection() -> None:
    samples = {
        "github pat": "token ghp_" + "A" * 30,
        "openai": "key sk-" + "B" * 20,
        "aws": "AKIAIOSFODNN7EXAMPLE",
        "private key": "-----BEGIN RSA PRIVATE KEY-----",
        "connection string": "postgres://user:pass@host:5432/db",
        "jwt": "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
        "inline password": "password = hunter2secret",
    }
    for name, content in samples.items():
        got = am.detect_secret("title", content)
        check(f"secrets: {name} detected", got is not None, f"(got {got})")

    clean = am.detect_secret("Use Postgres", "PostgreSQL was selected for ACID")
    check("secrets: clean content passes", clean is None)

    # Entropy heuristic: a long, non-repetitive alnum token (>4.5 bits/byte).
    high_ent = "kQ7mZ2xP9vR4nB6cL8dT3wY5hJ1sF0aG"  # 32 chars, no repetition
    got = am.detect_secret("title", high_ent)
    check("secrets: high-entropy token detected", got is not None)


def test_secret_rejection_and_override() -> None:
    store = tmp_store()
    try:
        am.create_memory(store, "decision", "Secret here", "ghp_" + "C" * 30, project="p")
        check("secrets: rejection raises", False)
    except am.AgentMemoryError as exc:
        check("secrets: rejection raises", "secret" in str(exc).lower())

    events = am.read_audit(store)
    check("secrets: MEMORY_REJECTED audited", any(e["event"] == "MEMORY_REJECTED" for e in events))

    rec = am.create_memory(
        store, "decision", "Allowed", "ghp_" + "D" * 30, project="p", allow_secret=True
    )
    check("secrets: --allow-secret stores", rec["trust"] == "untrusted")
    events = am.read_audit(store)
    check("secrets: SECRET_OVERRIDE audited", any(e["event"] == "SECRET_OVERRIDE" for e in events))


# --------------------------------------------------------------------------
# 6. Audit
# --------------------------------------------------------------------------

def test_audit_append_only() -> None:
    store = tmp_store()
    am.create_memory(store, "decision", "A", "a", project="p")
    am.create_memory(store, "decision", "B", "b", project="p")
    events = am.read_audit(store)
    check("audit: two MEMORY_CREATED", sum(1 for e in events if e["event"] == "MEMORY_CREATED") == 2)
    for e in events:
        check("audit: has at/event/memory_id/actor", all(k in e for k in ("at", "event", "memory_id", "actor")))
        check("audit: timestamp UTC Z", e["at"].endswith("Z"))


def test_audit_corrupt_line() -> None:
    store = tmp_store()
    (store / "audit.jsonl").write_text("garbage-not-json\n", encoding="utf-8")
    try:
        am.read_audit(store)
        check("audit: corrupt line raises", False)
    except am.AgentMemoryError:
        check("audit: corrupt line raises", True)


# --------------------------------------------------------------------------
# 7. Search + recall
# --------------------------------------------------------------------------

def _seed_for_search() -> Path:
    store = tmp_store()
    am.create_memory(store, "constraint", "Auth must use AuthService", "All authentication logic must use AuthService", project="p", paths=["src/auth/**"])
    am.create_memory(store, "error", "Refresh token reuse in auth", "Refresh tokens in the auth flow were reusable after rotation", project="p", paths=["src/auth/**"])
    am.create_memory(store, "decision", "Postgres for persistence", "PostgreSQL selected for ACID", project="p")
    return store


def test_search_semantics() -> None:
    store = _seed_for_search()
    results = am.search_memories(store, "auth")
    check("search: finds auth memories", len(results) == 2, f"(got {len(results)})")

    results = am.search_memories(store, "postgres")
    check("search: finds postgres", len(results) == 1 and results[0]["type"] == "decision")

    results = am.search_memories(store, "zzz-no-match")
    check("search: no match -> empty", results == [])

    # Empty query is a successful 0 results (V0.1_SPEC.md 7 / ROUND 4 contract).
    results = am.search_memories(store, "")
    check("search: empty query -> 0 results", results == [], f"(got {len(results)})")

    # Limit edge cases: N < 1 is a usage error (exit 2).
    try:
        am.search_memories(store, "auth", limit=0)
        check("search: limit 0 rejected", False)
    except am.UsageError:
        check("search: limit 0 rejected", True)
    try:
        am.search_memories(store, "auth", limit=-3)
        check("search: negative limit rejected", False)
    except am.UsageError:
        check("search: negative limit rejected", True)


def test_recall_semantics() -> None:
    store = _seed_for_search()
    # All memories are untrusted -> recall returns nothing by default.
    results = am.recall_memories(store, "auth")
    check("recall: untrusted excluded by default", results == [])

    # Promote the auth memories (constraint + error) and postgres.
    for r in am.list_memories(store):
        am.promote_trust(store, r["id"], "approved")

    results = am.recall_memories(store, "auth")
    check("recall: trusted auth found", len(results) == 2, f"(got {len(results)})")

    # Path bonus: constraint has higher score when path matches.
    results_path = am.recall_memories(store, "auth", path="src/auth/login.py")
    check("recall: path bonus ranks constraint first", results_path[0]["type"] == "constraint")

    # Limit.
    results_limited = am.recall_memories(store, "auth", limit=1)
    check("recall: limit honored", len(results_limited) == 1)

    # Default limit 10 (promote each created memory directly - ids, not ordering).
    store10 = tmp_store()
    for i in range(12):
        rec = am.create_memory(store10, "decision", f"Memory {i} auth", "about auth stuff", project="p")
        am.promote_trust(store10, rec["id"], "approved")
    results10 = am.recall_memories(store10, "auth")
    check("recall: default limit 10", len(results10) == 10, f"(got {len(results10)})")


def test_deterministic_ordering() -> None:
    """Same-score/same-second ties break on id - deterministic output."""
    store = tmp_store()
    ids = []
    for title in ("Alpha auth", "Beta auth", "Gamma auth"):
        rec = am.create_memory(store, "decision", title, "about auth", project="p")
        am.promote_trust(store, rec["id"], "approved")
        ids.append(rec["id"])
    r1 = am.recall_memories(store, "auth")
    r2 = am.recall_memories(store, "auth")
    check("determinism: identical across calls", [r["id"] for r in r1] == [r["id"] for r in r2])
    # Contract: ordering is (score, created_at, id) desc. Score is equal here,
    # so ids must be ordered by (created_at, id) desc - true even across
    # second boundaries (unlike the old id-only assertion, which was flaky).
    keys = [(r["created_at"], r["title"], r["id"]) for r in r1]
    check("determinism: (created_at, title, id) desc ordering", keys == sorted(keys, reverse=True), f"(got {keys})")

    # Explicit same-second tie -> id tie-break applies deterministically.
    store2 = tmp_store()
    for title in ("A auth", "B auth", "C auth"):
        rec = am.create_memory(store2, "decision", title, "about auth", project="p")
        rec["created_at"] = "2026-08-12T00:00:00Z"
        am.save_memory(store2, rec)
    tied = am.recall_memories(store2, "auth")
    got = [(r["title"], r["id"]) for r in tied]
    # v0.2 Tier 1: equal-score ties break on (created_at, title, id) desc -
    # content-derived title before the random id (cross-store stable).
    check("determinism: same-second tie breaks on (title, id) desc",
          got == sorted(got, reverse=True), f"(got {got})")





def test_idf_downweights_common_token() -> None:
    """v0.2 Tier 1: a token shared by many memories carries less signal."""
    store = tmp_store()
    # 4 memories share 'gate' (common token); only one also mentions 'merge'.
    for i in range(4):
        am.create_memory(store, "error", f"generic gate issue {i}", "gate handling",
                         project="p", tags=["gate"])
    rare = am.create_memory(store, "error", "merge-commit gate blocks push",
                            "gate rejected the merge", project="p", tags=["gate", "merge"])
    for r in am.list_memories(store):
        am.promote_trust(store, r["id"], "approved")
    # 'merge' is distinctive (1/5 docs); 'gate' is common (5/5).
    results = am.recall_memories(store, "merge gate")
    check("idf: distinctive token ranks the merge memory first",
          results[0]["id"] == rare["id"], f"(got {results[0]['title']})")


def test_phrase_bonus_ranks_contiguous_first() -> None:
    """v0.2 Tier 1: contiguous query terms in a field earn the phrase bonus."""
    store = tmp_store()
    a = am.create_memory(store, "decision", "deploy green blue to prod",
                         "green blue deploy", project="p")
    am.create_memory(store, "decision", "blue sky deploy green",
                     "deploy happens after blue and green", project="p")
    for r in am.list_memories(store):
        am.promote_trust(store, r["id"], "approved")
    results = am.recall_memories(store, "green blue")
    check("phrase: contiguous 'green blue' ranks first",
          results[0]["id"] == a["id"], f"(got {results[0]['title']})")


def test_idf_keeps_honest_zero_results() -> None:
    """v0.2 Tier 1 invariant: improved recall must NOT degrade honest zeros."""
    store = tmp_store()
    am.create_memory(store, "decision", "auth via service", "login flow",
                     project="p", tags=["auth"])
    for r in am.list_memories(store):
        am.promote_trust(store, r["id"], "approved")
    results = am.recall_memories(store, "deploy kubernetes cluster")
    check("tier1: unknown domain -> honest 0 results", results == [])


def test_idf_tie_break_still_deterministic() -> None:
    """v0.2 Tier 1: equal scores break on (created_at, title, id) desc."""
    store = tmp_store()
    for title in ("Alpha auth", "Beta auth", "Gamma auth"):
        rec = am.create_memory(store, "decision", title, "about auth", project="p")
        am.promote_trust(store, rec["id"], "approved")
    # All three contain 'auth' -> identical idf and hits -> equal scores.
    # Tie-break is (created_at, title, id) desc - title is stable across stores
    # (unlike random UUID ids).
    r1 = am.recall_memories(store, "auth")
    keys = [(r["created_at"], r["title"], r["id"]) for r in r1]
    check("tier1: equal-score tie breaks on (created_at, title, id) desc",
          keys == sorted(keys, reverse=True), f"(got {keys})")



def test_search_shares_idf_scoring() -> None:
    """v0.2 Tier 1: search_memories ranks a rare-token memory above a
    common-token one with equal hit counts (same scorer as recall)."""
    store = tmp_store()
    am.create_memory(store, "error", "gate handling", "gate pipeline",
                     project="p", tags=["gate"])
    rare = am.create_memory(store, "error", "merge handling", "merge pipeline",
                            project="p", tags=["merge"])
    results = am.search_memories(store, "merge gate")
    check("search: rare-token memory ranks first (IDF parity)",
          results[0]["id"] == rare["id"], f"(got {results[0]['title']})")


def test_idf_ubiquitous_term_still_ranked() -> None:
    """v0.2 Tier 1 acceptance: a term present in EVERY candidate still ranks
    (idf ~ 1, not 0) - no accidental honest-zero regression for ubiquitous
    vocabulary."""
    store = tmp_store()
    for i in range(5):
        am.create_memory(store, "decision", f"auth memory {i}", "about auth",
                         project="p", tags=["auth"])
    for r in am.list_memories(store):
        am.promote_trust(store, r["id"], "approved")
    results = am.recall_memories(store, "auth")
    check("tier1: ubiquitous term still returns results",
          len(results) == 5, f"(got {len(results)})")

# --------------------------------------------------------------------------
# 8. Path matcher
# --------------------------------------------------------------------------

def test_path_matcher() -> None:
    check("path: ** matches deep", am.path_matches("src/auth/**", "src/auth/login.py"))
    check("path: ** matches nested", am.path_matches("src/auth/**", "src/auth/deep/deeper.py"))
    check("path: * within segment", am.path_matches("src/*/x.py", "src/auth/x.py"))
    check("path: * no slash cross", not am.path_matches("src/*.py", "src/a/b.py"))
    check("path: exact", am.path_matches("src/auth.py", "src/auth.py"))
    check("path: no match", not am.path_matches("src/auth/**", "src/frontend/x.py"))
    check("path: ** zero segments", am.path_matches("src/auth/**", "src/auth"))
    check("path: ? one char", am.path_matches("src/?/x.py", "src/a/x.py"))
    try:
        am.validate_path_pattern("src/[ab]/x.py")
        check("path: char class rejected", False)
    except am.UsageError:
        check("path: char class rejected", True)


# --------------------------------------------------------------------------
# 9. CLI subprocess integration (real binary, output VALUES)
# --------------------------------------------------------------------------

def _run_cli(tmp: Path, *args: str, cwd: Path | None = None, stdin_text: str | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).parent)
    return subprocess.run(
        [sys.executable, str(Path(__file__).parent / "agent_memory.py"), *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=cwd or tmp, env=env, timeout=60, input=stdin_text,
    )


def test_cli_flow() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="agent-memory-cli-"))
    r = _run_cli(tmp, "init", "--project", "demo")
    check("cli: init exit 0", r.returncode == 0, f"(rc={r.returncode}, stderr={r.stderr})")
    check("cli: init message", "initialized" in r.stdout)

    r = _run_cli(tmp, "add", "--type", "constraint", "--title", "Auth via AuthService", "--content", "All auth must use AuthService", "--paths", "src/auth/**")
    check("cli: add exit 0", r.returncode == 0, f"(rc={r.returncode}, stderr={r.stderr})")
    mem_id = r.stdout.strip().split()[-1]
    check("cli: add prints created mem_", mem_id.startswith("mem_"))

    r = _run_cli(tmp, "list")
    check("cli: list finds 1", "1 result(s)" in r.stdout, f"(got {r.stdout!r})")

    r = _run_cli(tmp, "search", "auth")
    check("cli: search finds it", "Auth via AuthService" in r.stdout)

    r = _run_cli(tmp, "search", "")
    check("cli: empty search 0 results exit 0", r.returncode == 0 and "0 results" in r.stdout)

    r = _run_cli(tmp, "recall", "auth")
    check("cli: recall 0 (untrusted)", "0 results" in r.stdout)

    r = _run_cli(tmp, "promote", mem_id, "--trust", "approved")
    check("cli: promote exit 0", r.returncode == 0, f"(rc={r.returncode}, stderr={r.stderr})")

    r = _run_cli(tmp, "recall", "auth", "--path", "src/auth/login.py")
    check("cli: recall returns after promote", "RELEVANT CONTEXT" in r.stdout, f"(got {r.stdout!r})")

    r = _run_cli(tmp, "search", "nothing-here")
    check("cli: no-match search 0 results exit 0", r.returncode == 0 and "0 results" in r.stdout)

    r = _run_cli(tmp, "status", "--json")
    data = json.loads(r.stdout)
    check("cli: status json total 1", data["total"] == 1, f"(got {data})")


def test_cli_interactive_add() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="agent-memory-cli-"))
    _run_cli(tmp, "init")
    stdin_text = "constraint\nAuth via AuthService\nAll auth must use AuthService\n\nsrc/auth/**\n"
    r = _run_cli(tmp, "add", stdin_text=stdin_text)
    check("cli: interactive add exit 0", r.returncode == 0, f"(rc={r.returncode}, stderr={r.stderr})")
    check("cli: interactive add created", "created mem_" in r.stdout)


def test_cli_error_paths() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="agent-memory-cli-"))
    # Not initialized -> exit 1, clean message.
    r = _run_cli(tmp, "list")
    check("cli: not-init exit 1", r.returncode == 1, f"(rc={r.returncode})")
    check("cli: not-init clean message", "agent-memory init" in (r.stderr or r.stdout))

    _run_cli(tmp, "init")
    # Usage error -> exit 2.
    r = _run_cli(tmp, "add")
    check("cli: add no flags no-tty exit 2", r.returncode == 2, f"(rc={r.returncode})")
    r = _run_cli(tmp, "promote", "mem_x", "--trust", "system")
    check("cli: promote system exit 2", r.returncode == 2, f"(rc={r.returncode})")
    # Bad enum -> exit 2.
    r = _run_cli(tmp, "add", "--type", "bogus", "--title", "t", "--content", "c")
    check("cli: bad type exit 2", r.returncode == 2, f"(rc={r.returncode})")
    # --provenance system is not even a choice -> argparse exit 2.
    r = _run_cli(tmp, "add", "--type", "decision", "--title", "t", "--content", "c", "--provenance", "system")
    check("cli: provenance system exit 2", r.returncode == 2, f"(rc={r.returncode})")
    # --limit 0 -> usage error exit 2.
    r = _run_cli(tmp, "search", "auth", "--limit", "0")
    check("cli: search limit 0 exit 2", r.returncode == 2, f"(rc={r.returncode})")
    # Secret rejection -> exit 1.
    r = _run_cli(tmp, "add", "--type", "decision", "--title", "s", "--content", "ghp_" + "E" * 30)
    check("cli: secret reject exit 1", r.returncode == 1, f"(rc={r.returncode})")
    check("cli: secret message", "secret" in (r.stderr or r.stdout).lower())
    # JSON error output on stdout.
    r = _run_cli(tmp, "add", "--type", "decision", "--title", "s", "--content", "ghp_" + "F" * 30, "--json")
    check("cli: json error object", r.returncode == 1 and '"error"' in r.stdout)
    # JSON empty search shape.
    r = _run_cli(tmp, "search", "", "--json")
    data = json.loads(r.stdout)
    check("cli: json empty search shape", data == {"results": [], "count": 0}, f"(got {data})")


# --------------------------------------------------------------------------

def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
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
