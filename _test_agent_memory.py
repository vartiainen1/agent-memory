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


def test_tier21_floor_drops_weak_tail() -> None:
    """v0.2 Tier 2.1 (EVIDENCE-010/015): recall on a small store returns the
    genuinely relevant cluster, not the whole corpus. A memory matching the
    query once in content (idf-weight ~1, far below 0.25 x the top text
    score) is dropped by the floor."""
    store = tmp_store()
    am.create_memory(store, "decision", "config format: back to YAML",
                     "REASON: YAML is the canonical config format",
                     tags=["config", "yaml"], project="p")
    am.create_memory(store, "decision", "config format: TOML",
                     "REASON: TOML is stricter", tags=["config", "toml"],
                     project="p")
    am.create_memory(store, "decision", "filler unrelated",
                     "just one config mention nowhere else", tags=["misc"],
                     project="p")
    for r in am.list_memories(store):
        am.promote_trust(store, r["id"], "approved")
    results = am.recall_memories(store, "config format")
    titles = [r["title"] for r in results]
    check("tier2.1: weak single-mention memory dropped by the floor",
          "filler unrelated" not in titles, f"(got {titles})")
    check("tier2.1: relevant cluster kept",
          "config format: back to YAML" in titles and
          "config format: TOML" in titles, f"(got {titles})")


def test_tier21_floor_sparse_unique_match_survives() -> None:
    """v0.2 Tier 2.1: 'genuinely relevant low-score results' requirement - a
    sparse-but-unique match (low text score, sole match for the query) is
    never zeroed, because the top text score always passes its own relative
    floor (an ABSOLUTE threshold would drop it - this test discriminates)."""
    store = tmp_store()
    am.create_memory(store, "decision", "deploy kubernetes cluster",
                     "REASON: production runs on k8s", tags=["kubernetes"],
                     project="p")
    for r in am.list_memories(store):
        am.promote_trust(store, r["id"], "approved")
    results = am.recall_memories(store, "kubernetes")
    check("tier2.1: sparse unique match survives the floor",
          len(results) == 1 and results[0]["title"] == "deploy kubernetes cluster",
          f"(got {len(results)})")


def test_tier21_floor_honest_zero_preserved() -> None:
    """v0.2 Tier 2.1: unknown queries still return 0 results (the invariant
    must survive the floor - EVIDENCE-003)."""
    store = tmp_store()
    am.create_memory(store, "decision", "auth session state",
                     "server-side session", tags=["auth"], project="p")
    for r in am.list_memories(store):
        am.promote_trust(store, r["id"], "approved")
    results = am.recall_memories(store, "deploy kubernetes cluster")
    check("tier2.1: honest zero results preserved",
          results == [], f"(got {len(results)})")


def test_tier21_floor_tiny_store() -> None:
    """v0.2 Tier 2.1: stores smaller than any floor assumption (2 memories)
    behave sanely - both equally-matching memories are returned (ties are
    not floored)."""
    store = tmp_store()
    am.create_memory(store, "decision", "alpha auth", "about auth alpha",
                     tags=["auth"], project="p")
    am.create_memory(store, "decision", "beta auth", "about auth beta",
                     tags=["auth"], project="p")
    for r in am.list_memories(store):
        am.promote_trust(store, r["id"], "approved")
    results = am.recall_memories(store, "auth")
    check("tier2.1: tiny store returns both matching memories",
          len(results) == 2, f"(got {len(results)})")


def test_tier21_floor_ignores_superseded() -> None:
    """v0.2 Tier 2.1: the floor never resurrects superseded memories - the
    candidate set is active-only (floor applies after status filtering)."""
    store = tmp_store()
    a = am.create_memory(store, "decision", "config: yaml", "yaml wins",
                         tags=["config"], project="p")
    b = am.create_memory(store, "decision", "config: toml", "toml supersedes",
                         tags=["config"], project="p")
    for r in am.list_memories(store):
        am.promote_trust(store, r["id"], "approved")
    am.supersede(store, a["id"], b["id"])
    results = am.recall_memories(store, "config")
    check("tier2.1: superseded excluded (floor does not resurrect)",
          [r["title"] for r in results] == ["config: toml"],
          f"(got {[r['title'] for r in results]})")


def test_tier21_floor_path_match_exempt() -> None:
    """v0.2 Tier 2.1: a memory that matches --path survives the floor even
    when its text score is far below the ratio - path is explicit intent
    (always kept), though it may still rank below a stronger text match."""
    store = tmp_store()
    am.create_memory(store, "constraint", "auth must use service",
                     "always route through AuthService",
                     tags=["auth"], paths=["src/auth/*"], project="p")
    am.create_memory(store, "lesson", "auth middleware note",
                     "middleware wraps the handler", tags=["auth"],
                     paths=["src/auth/middleware.py"], project="p")
    for r in am.list_memories(store):
        am.promote_trust(store, r["id"], "approved")
    results = am.recall_memories(store, "auth must use service",
                                 path="src/auth/middleware.py")
    titles = [r["title"] for r in results]
    check("tier2.1: path-matched memory exempt from the floor",
          "auth middleware note" in titles, f"(got {titles})")


def test_tier21_floor_search_stays_inclusive() -> None:
    """v0.2 Tier 2.1: search is NOT floored - operators keep full visibility
    of weak matches (EVIDENCE-007); only recall (agent-facing) applies the
    floor."""
    store = tmp_store()
    am.create_memory(store, "decision", "config format: back to YAML",
                     "REASON: YAML is the canonical config format",
                     tags=["config"], project="p")
    am.create_memory(store, "decision", "filler unrelated",
                     "just one config mention nowhere else", tags=["misc"],
                     project="p")
    for r in am.list_memories(store):
        am.promote_trust(store, r["id"], "approved")
    s = am.search_memories(store, "config format")
    r = am.recall_memories(store, "config format")
    s_titles = [m["title"] for m in s]
    r_titles = [m["title"] for m in r]
    check("tier2.1: search returns weak matches (inclusive)",
          "filler unrelated" in s_titles, f"(got {s_titles})")
    check("tier2.1: recall drops the weak match (floored)",
          "filler unrelated" not in r_titles, f"(got {r_titles})")


# 1. Schema / validation
ERROR_LOG_FIXTURE = """================================================================================
ERROR LOG - HOW TO USE
================================================================================
Tooling (run from this folder):
  python check_errors.py   validate every entry

================================================================================
EXAMPLE ENTRIES (replace with your own; delete this section header)
================================================================================

[2026-08-05] AREA: payment webhook parser
  ERROR: KeyError: 'amount' on webhook payloads without an amount field
  CAUSE: the payload dict has no 'amount' key
  FIX: use payload.get('amount', 0)
  STATUS: FIXED.

================================================================================
REAL ENTRIES
================================================================================

[2026-08-09] AREA: CI commit-message gate missing
  ERROR: --no-verify commits slip past the local gate unnoticed
  CAUSE: the CI backstop never re-checked the commit message
  FIX: check_errors.py --check-commit + a commit-gate CI job
  STATUS: FIXED.

[2026-08-12] AREA: diff-gate --staged outside a git repo
  ERROR: dumps raw git usage and exits 2
  CAUSE: no git repo guard existed
  FIX: detect non-repo and exit cleanly
  STATUS: OPEN.
"""

DECISION_LOG_FIXTURE = """# DECISION LOG
# Append-only: corrections are new entries that point back.

================================================================================
1) EXAMPLE ENTRIES
================================================================================
[2026-08-09 14:32] DECISION: used regex instead of AST parser
  REASON: faster for simple case, file was small
  FILES: src/parser.py
  STATUS: LOCKED

================================================================================
4) COMPANION TOOL
================================================================================
[2026-08-09 05:45] DECISION: JWT for single-consumer API
  REASON: one API consumer, no third-party login needed
  FILES: auth.py
  SUPERSEDES: 2026-08-09 14:32
  STATUS: LOCKED.

[2026-08-11 10:00] DECISION: back to regex for the small parser
  REASON: parser file shrank to 40 lines
  FILES: src/parser.py
  SUPERSEDES: 2026-08-09 14:32
  STATUS: REVISED.

[2026-08-12 11:30] DECISION: AST after all - parser split across modules
  REASON: refactor split the parser; regex state scattered
  FILES: src/parser.py
  SUPERSEDES: 2026-08-11 10:00
  STATUS: REVISED.
"""


def test_tier23_import_all_entries_mirrors_sibling_parser() -> None:
    """v0.2 Tier 2.3: import parses what the sibling's own parse_entries
    considers an entry - including EXAMPLE-section entries (they are real
    chain roots: the decision-log's SUPERSEDES references point at them)."""
    store = tmp_store()
    report = am.import_source_log(store, "error-log", ERROR_LOG_FIXTURE, project="p")
    check("tier2.3: error-log parses all 3 real + example entries",
          report["new"] == 3, f"(got {report['entries']} entries, {report['new']} new)")
    store2 = tmp_store()
    report2 = am.import_source_log(store2, "decision-log", DECISION_LOG_FIXTURE, project="p")
    check("tier2.3: decision-log parses all 4 entries",
          report2["new"] == 4, f"(got {report2['new']})")


def test_tier23_invariant_same_entry_same_fingerprint() -> None:
    """INVARIANT 1: same source entry -> same fingerprint, independent of
    EOL/trailing whitespace (canonicalization contract, ROUND 2 #2)."""
    a = "[2026-08-09] AREA: CI gate\n  ERROR: x\n  STATUS: FIXED."
    b = "[2026-08-09] AREA: CI gate\r\n  ERROR: x \r\n  STATUS: FIXED.   \n\n"
    check("tier2.3: canonical bytes identical across EOL/whitespace",
          am._canonical_bytes(a) == am._canonical_bytes(b))
    check("tier2.3: same fingerprint", am.fingerprint_entry(a) == am.fingerprint_entry(b))
    check("tier2.3: fingerprint prefixed sha256:",
          am.fingerprint_entry(a).startswith("sha256:"))


def test_tier23_invariant_reimport_no_duplicates() -> None:
    """INVARIANT 2: re-import -> no duplicate memory (fingerprint dedupe)."""
    store = tmp_store()
    r1 = am.import_source_log(store, "error-log", ERROR_LOG_FIXTURE, project="p")
    r2 = am.import_source_log(store, "error-log", ERROR_LOG_FIXTURE, project="p")
    total = len(am.list_memories(store))
    check("tier2.3: first import created memories",
          r1["new"] == 3 and total == 3, f"(total {total})")
    check("tier2.3: re-import creates nothing",
          r2["new"] == 0 and r2["duplicates"] == 3,
          f"(new {r2['new']}, dup {r2['duplicates']})")
    check("tier2.3: re-import total unchanged", total == 3)


def test_tier23_invariant_untrusted_and_import_provenance() -> None:
    """INVARIANT 3+4: imported material starts untrusted with provenance
    import; import never manufactures verified/approved/system trust."""
    store = tmp_store()
    am.import_source_log(store, "error-log", ERROR_LOG_FIXTURE, project="p")
    for r in am.list_memories(store):
        check(f"tier2.3: {r['title']} born untrusted", r["trust"] == "untrusted",
              f"(got {r['trust']})")
        check(f"tier2.3: {r['title']} provenance import", r["provenance"] == "import",
              f"(got {r['provenance']})")
        src = r.get("source")
        check(f"tier2.3: {r['title']} source populated",
              isinstance(src, dict) and src.get("repository") == "agent-error-log"
              and src.get("fingerprint", "").startswith("sha256:"),
              f"(got {src})")


def test_tier23_invariant_secret_before_persist() -> None:
    """INVARIANT 5: secret detection runs BEFORE persistence; a source entry
    carrying a secret is rejected + audited, never stored."""
    secret_log = ERROR_LOG_FIXTURE.replace("use payload.get('amount', 0)",
                                           "use token ghp_1234567890abcdefghij")
    store = tmp_store()
    report = am.import_source_log(store, "error-log", secret_log, project="p")
    check("tier2.3: secret entry rejected", report["rejected"] == 1,
          f"(got {report['rejected']}, {report['rejected_details']})")
    check("tier2.3: secret entry not persisted", len(am.list_memories(store)) == 2,
          f"(got {len(am.list_memories(store))})")
    check("tier2.3: MEMORY_REJECTED audited",
          "MEMORY_REJECTED" in am.read_audit(store).__class__.__name__ or
          any(e["event"] == "MEMORY_REJECTED" for e in am.read_audit(store)),
          "no MEMORY_REJECTED event")


def test_tier23_invariant_provenance_survives() -> None:
    """INVARIANT 6: source provenance survives import (tag + fingerprint
    stored on the memory and readable back)."""
    store = tmp_store()
    am.import_source_log(store, "error-log", ERROR_LOG_FIXTURE, project="p")
    rec = next(r for r in am.list_memories(store) if "diff-gate" in r["title"])
    check("tier2.3: source.tag survives",
          rec["source"].get("tag") == "2026-08-12",
          f"(got {rec['source']})")
    check("tier2.3: source.type is memory type",
          rec["source"].get("type") == "error", f"(got {rec['source']})")


def test_tier23_invariant_supersession_wired() -> None:
    """INVARIANT 7: decision-log SUPERSEDES links are preserved where
    deterministically knowable (same run, active targets)."""
    store = tmp_store()
    report = am.import_source_log(store, "decision-log", DECISION_LOG_FIXTURE, project="p")
    check("tier2.3: supersession links wired", report["superseded"] == 2,
          f"(got {report['superseded']}, unresolved {report['unresolved_supersedes']})")
    # AST-after-all is active and supersedes back-to-regex; back-to-regex superseded.
    ast = next(r for r in am.list_memories(store) if "AST after all" in r["title"])
    back = next(r for r in am.list_memories(store) if "back to regex" in r["title"])
    check("tier2.3: AST-after-all active", ast["status"] == "active",
          f"(got {ast['status']})")
    check("tier2.3: back-to-regex superseded", back["status"] == "superseded",
          f"(got {back['status']})")
    check("tier2.3: AST supersedes back-to-regex",
          ast.get("supersedes") == back["id"],
          f"(got {ast.get('supersedes')} vs {back['id']})")


def test_tier23_invariant_no_overwrite_of_existing() -> None:
    """INVARIANT 8: existing manually-created memories are never silently
    overwritten - import only creates; a same-title manual memory stays."""
    store = tmp_store()
    am.create_memory(store, "error", "CI commit-message gate missing",
                     "manual entry - do not touch", project="p",
                     provenance="human")
    am.import_source_log(store, "error-log", ERROR_LOG_FIXTURE, project="p")
    manual = next(r for r in am.list_memories(store)
                  if r["provenance"] == "human")
    check("tier2.3: manual memory untouched",
          manual["content"] == "manual entry - do not touch" and
          manual["source"] is None, f"(got {manual})")


def test_tier23_invariant_failures_explicit_auditable() -> None:
    """INVARIANT 9: import failures are explicit and auditable - bad source
    is a usage error; IMPORT_RUN audit records counts."""
    store = tmp_store()
    try:
        am.import_source_log(store, "unknown-source", "x", project="p")
        check("tier2.3: unknown source rejected", False, "no error raised")
    except am.UsageError:
        check("tier2.3: unknown source rejected", True, "")
    am.import_source_log(store, "error-log", ERROR_LOG_FIXTURE, project="p")
    events = [e for e in am.read_audit(store) if e["event"] == "IMPORT_RUN"]
    check("tier2.3: IMPORT_RUN audited", len(events) == 1, f"(got {len(events)})")
    check("tier2.3: IMPORT_RUN detail",
          events[0]["detail"]["new"] == 3 and
          events[0]["detail"]["repository"] == "agent-error-log",
          f"(got {events[0]['detail']})")


def test_tier23_invariant_recall_excludes_untrusted_imports() -> None:
    """INVARIANT 10: recall continues excluding untrusted material by
    default - imported memories are invisible until human promotion."""
    store = tmp_store()
    am.import_source_log(store, "error-log", ERROR_LOG_FIXTURE, project="p")
    check("tier2.3: recall excludes untrusted imports",
          am.recall_memories(store, "commit message gate") == [],
          "recall leaked untrusted imports")
    # human promotion makes it visible
    rec = next(r for r in am.list_memories(store) if "CI commit" in r["title"])
    am.promote_trust(store, rec["id"], "approved")
    res = am.recall_memories(store, "commit message gate")
    check("tier2.3: promoted import recallable",
          any(r["id"] == rec["id"] for r in res), f"(got {len(res)})")


def test_tier23_invariant_honest_zeros_preserved() -> None:
    """INVARIANT 11: unknown/irrelevant queries continue producing honest
    zero results after import (the invariant survives import)."""
    store = tmp_store()
    am.import_source_log(store, "error-log", ERROR_LOG_FIXTURE, project="p")
    for r in am.list_memories(store):
        am.promote_trust(store, r["id"], "approved")
    check("tier2.3: honest zero after import",
          am.recall_memories(store, "deploy kubernetes cluster") == [],
          "import broke honest zeros")


def test_tier23_dry_run_persists_nothing() -> None:
    """v0.2 Tier 2.3: --dry-run computes fingerprints/report but persists
    nothing and writes no audit events."""
    store = tmp_store()
    report = am.import_source_log(store, "error-log", ERROR_LOG_FIXTURE,
                                  project="p", dry_run=True)
    check("tier2.3: dry-run reports would-create",
          report["dry_run"] and report["new"] == 3, f"(got {report})")
    check("tier2.3: dry-run persists nothing", len(am.list_memories(store)) == 0)
    check("tier2.3: dry-run writes no audit",
          am.read_audit(store) == [], f"(got {am.read_audit(store)})")



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


def test_path_tier() -> None:
    """v0.3 T2.1 (EVIDENCE-031): path_tier hierarchy - exact file (3) > bare
    directory (2) > glob/prefix (1) > unrelated/neutral (0)."""
    check("tier: exact file", am.path_tier("src/auth/login.py", "src/auth/login.py") == 3)
    check("tier: bare directory routes files under it",
          am.path_tier("src/auth", "src/auth/login.py") == 2)
    check("tier: deep directory", am.path_tier("src/auth", "src/auth/deep/x.py") == 2)
    check("tier: sibling prefix collision not directory",
          am.path_tier("src/auth", "src/auth2/x.py") == 0)
    check("tier: glob prefix match", am.path_tier("src/auth/**", "src/auth/login.py") == 1)
    check("tier: loose ** glob", am.path_tier("src/**", "src/auth/x.py") == 1)
    check("tier: bare ** is neutral", am.path_tier("**", "src/auth/x.py") == 0)
    check("tier: unrelated no match", am.path_tier("src/other/**", "src/auth/x.py") == 0)
    check("tier: unrelated bare dir", am.path_tier("src/other", "src/auth/x.py") == 0)
    check("tier: empty pattern neutral", am.path_tier("", "src/a.py") == 0)
    check("tier: empty path neutral", am.path_tier("src/a.py", "") == 0)


def test_tier21_path_bonus_hierarchy() -> None:
    """v0.3 T2.1 A4 (EVIDENCE-031): for equal text scores, exact file >
    directory > glob > unrelated ordering when --path is given."""
    store = tmp_store()
    ids = {}
    for kind, pats in [("exact", ["src/auth/login.py"]),
                       ("dir", ["src/auth"]),
                       ("glob", ["src/auth/**"]),
                       ("none", [])]:
        ids[kind] = am.create_memory(
            store, "decision", f"auth {kind}", "auth policy",
            project="p", paths=pats)["id"]
    for mid in ids.values():
        am.promote_trust(store, mid, "approved")
    results = am.recall_memories(store, "auth", path="src/auth/login.py")
    order = [r["title"] for r in results]
    check("t2.1 A4: exact ranks above directory",
          order.index("auth exact") < order.index("auth dir"), f"(got {order})")
    check("t2.1 A4: directory ranks above glob",
          order.index("auth dir") < order.index("auth glob"), f"(got {order})")
    check("t2.1 A4: glob ranks above unrelated",
          order.index("auth glob") < order.index("auth none"), f"(got {order})")


def test_t21_path_text_still_matters() -> None:
    """v0.3 T2.1 A5 (EVIDENCE-031): a strong text match without a path can
    still outrank a weak-text exact-path match - path is a signal, not an
    auto-win."""
    store = tmp_store()
    strong = am.create_memory(
        store, "lesson", "postgresql transactional consistency engine",
        "postgresql chosen for transactional consistency of financial data",
        project="p", tags=[])["id"]
    weak_exact = am.create_memory(
        store, "decision", "x", "y", project="p",
        paths=["src/db/main.py"])["id"]
    for mid in (strong, weak_exact):
        am.promote_trust(store, mid, "approved")
    results = am.recall_memories(store, "postgresql transactional",
                                 path="src/db/main.py")
    titles = [r["title"] for r in results]
    check("t2.1 A5: strong text outranks weak exact-path",
          titles[0] == "postgresql transactional consistency engine",
          f"(got {titles})")


def test_t21_directory_scoping_recall() -> None:
    """v0.3 T2.1 A6 (EVIDENCE-031): a bare directory pattern routes to files
    under it via the CLI - previously it matched nothing useful."""
    tmp = Path(tempfile.mkdtemp(prefix="agent-memory-t21-"))
    _run_cli(tmp, "init")
    m1 = _run_cli(tmp, "add", "--type", "constraint", "--title", "Auth dir rule",
                  "--content", "all auth via AuthService", "--paths", "src/auth")
    m2 = _run_cli(tmp, "add", "--type", "constraint", "--title", "Other area",
                  "--content", "unrelated area", "--paths", "src/other")
    check("t2.1 A6: two memories added", m1.returncode == 0 and m2.returncode == 0,
          f"({m1.stderr} {m2.stderr})")
    for line in (m1.stdout + m2.stdout).splitlines():
        if "created mem_" in line:
            mid = line.split()[-1]
            r = _run_cli(tmp, "promote", mid, "--trust", "approved")
            check("t2.1 A6: promote ok", r.returncode == 0, r.stderr)
    r = _run_cli(tmp, "recall", "auth", "--path", "src/auth/login.py", "--json")
    payload = json.loads(r.stdout)
    titles = [x["title"] for x in payload.get("results", [])]
    check("t2.1 A6: directory-routed memory first",
          titles and titles[0] == "Auth dir rule", f"(got {titles})")


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
# T3 suggestions - approve-to-persist loop (v0.3 Tier 3, EVIDENCE-034)
# A suggestion is a CANDIDATE, not a memory: no trust/status, no recall
# participation, human-only approval, terminal reject/approve, audited.
# --------------------------------------------------------------------------

def test_t3_propose_creates_pending_suggestion_not_memory() -> None:
    store = tmp_store()
    sug = am.propose_suggestion(store, "constraint", "auth via AuthService",
                                "All auth must use AuthService",
                                project="p", paths=["src/auth/**"])
    check("t3: id space is sug_", sug["id"].startswith("sug_"), f"(got {sug['id']})")
    check("t3: state pending", sug["state"] == "pending", f"(got {sug})")
    check("t3: provenance pinned to agent", sug["provenance"] == "agent",
          f"(got {sug.get('provenance')})")
    check("t3: no trust field (not a memory)", "trust" not in sug,
          f"(got {sorted(sug.keys())})")
    check("t3: no status field (not a memory)", "status" not in sug,
          f"(got {sorted(sug.keys())})")
    check("t3: no MEMORY persisted", am.list_memories(store) == [],
          f"(got {am.list_memories(store)})")
    check("t3: suggestion file written to suggestions dir",
          (store / am.SUGGESTION_SUBDIR / f"{sug['id']}.json").exists())
    events = am.read_audit(store)
    check("t3: SUGGESTION_CREATED audited",
          [e["event"] for e in events] == ["SUGGESTION_CREATED"], f"(got {events})")
    check("t3: audit actor agent", events[0]["actor"] == "agent", f"(got {events[0]})")


def test_t3_propose_secret_rejected_before_write() -> None:
    """The queue must not be a backdoor persistence mechanism: secrets are
    rejected BEFORE any write - nothing is persisted and it is audited."""
    store = tmp_store()
    try:
        am.propose_suggestion(store, "constraint", "token",
                              "here is the key: " + "ghp_" + "a" * 36,
                              project="p")
        raised = False
    except am.AgentMemoryError:
        raised = True
    check("t3: secret proposal rejected", raised)
    check("t3: no suggestion persisted", am.list_suggestions(store) == [],
          f"(got {am.list_suggestions(store)})")
    check("t3: no memory persisted", am.list_memories(store) == [],
          f"(got {am.list_memories(store)})")
    events = am.read_audit(store)
    check("t3: SUGGESTION_REJECTED audited",
          [e["event"] for e in events] == ["SUGGESTION_REJECTED"], f"(got {events})")
    check("t3: rejection detail is secret_detected",
          events[0]["detail"].get("reason") == "secret_detected", f"(got {events[0]})")


def test_t3_approve_converts_to_memory_with_chosen_trust() -> None:
    store = tmp_store()
    sug = am.propose_suggestion(store, "error", "token refresh bug",
                                "tokens were reusable", project="p")
    record = am.approve_suggestion(store, sug["id"], "approved")
    check("t3: approve creates a memory", record["id"].startswith("mem_"),
          f"(got {record['id']})")
    check("t3: trust is the human-chosen level", record["trust"] == "approved",
          f"(got {record['trust']})")
    check("t3: memory provenance stays agent", record["provenance"] == "agent",
          f"(got {record.get('provenance')})")
    check("t3: suggestion now approved",
          am.load_suggestion(store, sug["id"])["state"] == "approved")
    check("t3: approved memory is recalled",
          [r["title"] for r in am.recall_memories(store, "token refresh")]
          == ["token refresh bug"], f"(got {am.recall_memories(store, 'token refresh')})")
    events = am.read_audit(store)
    approves = [e for e in events if e["event"] == "SUGGESTION_APPROVED"]
    check("t3: SUGGESTION_APPROVED audited with memory_id + trust",
          len(approves) == 1 and approves[0]["memory_id"] == record["id"]
          and approves[0]["detail"]["trust"] == "approved", f"(got {approves})")


def test_t3_approve_trust_ladder_governs() -> None:
    """Approval != automatic max trust: the human picks verified or approved,
    never 'system'; invalid trust is a usage error."""
    store = tmp_store()
    sug = am.propose_suggestion(store, "lesson", "rule", "content", project="p")
    record = am.approve_suggestion(store, sug["id"], "verified")
    check("t3: verified chosen", record["trust"] == "verified", f"(got {record['trust']})")
    sug2 = am.propose_suggestion(store, "lesson", "rule2", "content2", project="p")
    try:
        am.approve_suggestion(store, sug2["id"], "system")
        raised = False
    except am.UsageError:
        raised = True
    check("t3: approve --trust system rejected", raised)
    check("t3: rejected-system suggestion still pending",
          am.load_suggestion(store, sug2["id"])["state"] == "pending")


def test_t3_approve_reject_terminal() -> None:
    store = tmp_store()
    sug = am.propose_suggestion(store, "constraint", "t", "c", project="p")
    am.approve_suggestion(store, sug["id"], "approved")
    try:
        am.approve_suggestion(store, sug["id"], "approved")
        twice = False
    except am.AgentMemoryError:
        twice = True
    check("t3: double-approve rejected (terminal)", twice)
    sug2 = am.propose_suggestion(store, "constraint", "t2", "c2", project="p")
    am.reject_suggestion(store, sug2["id"])
    check("t3: reject marks state rejected",
          am.load_suggestion(store, sug2["id"])["state"] == "rejected")
    try:
        am.approve_suggestion(store, sug2["id"], "approved")
        after = False
    except am.AgentMemoryError:
        after = True
    check("t3: approve after reject rejected (terminal)", after)
    try:
        am.reject_suggestion(store, sug2["id"])
        rej2 = False
    except am.AgentMemoryError:
        rej2 = True
    check("t3: double-reject rejected (terminal)", rej2)
    check("t3: rejected suggestion created no memory",
          am.list_memories(store) != [] and len(am.list_memories(store)) == 1,
          f"(got {am.list_memories(store)})")
    rejects = [e for e in am.read_audit(store) if e["event"] == "SUGGESTION_REJECTED"]
    check("t3: SUGGESTION_REJECTED audited (human_review)",
          len(rejects) == 1 and rejects[0]["detail"]["reason"] == "human_review",
          f"(got {rejects})")


def test_t3_approve_reruns_secret_scan() -> None:
    """Defense against disk tampering: a proposal edited to contain a secret
    after enqueue must never become memory."""
    store = tmp_store()
    sug = am.propose_suggestion(store, "constraint", "t", "clean content",
                                project="p")
    path = store / am.SUGGESTION_SUBDIR / f"{sug['id']}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["content"] = "leak: " + "sk-" + "c" * 20
    am._write_text_utf8(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    try:
        am.approve_suggestion(store, sug["id"], "approved")
        raised = False
    except am.AgentMemoryError:
        raised = True
    check("t3: tampered proposal refused at approval", raised)
    check("t3: no memory created from tampered proposal",
          am.list_memories(store) == [], f"(got {am.list_memories(store)})")


def test_t3_list_filters_and_status() -> None:
    store = tmp_store()
    s1 = am.propose_suggestion(store, "constraint", "one", "c1", project="p")
    s2 = am.propose_suggestion(store, "error", "two", "c2", project="p")
    am.reject_suggestion(store, s2["id"])
    all_ids = [s["id"] for s in am.list_suggestions(store)]
    check("t3: list all returns both (created_at desc, id desc tie-break)",
          len(all_ids) == 2 and set(all_ids) == {s1["id"], s2["id"]}
          and all_ids == sorted(all_ids, reverse=True), f"(got {all_ids})")
    check("t3: list state filter pending",
          [s["id"] for s in am.list_suggestions(store, state="pending")] == [s1["id"]])
    check("t3: list state filter rejected",
          [s["id"] for s in am.list_suggestions(store, state="rejected")] == [s2["id"]])
    try:
        am.load_suggestion(store, "sug_00000000-0000-0000-0000-000000000000")
        raised = False
    except am.AgentMemoryError:
        raised = True
    check("t3: missing suggestion id raises clean error", raised)
    summary = am.status_summary(store)
    check("t3: status reports pending suggestion count",
          summary["suggestions_pending"] == 1, f"(got {summary})")


def test_t3_cli_flow() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="agent-memory-t3-cli-"))
    _run_cli(tmp, "init", "--project", "demo")
    r = _run_cli(tmp, "add", "--type", "constraint", "--title", "Existing",
                 "--content", "already trusted context")
    mem_id = r.stdout.strip().split()[-1]
    _run_cli(tmp, "promote", mem_id, "--trust", "approved")
    r = _run_cli(tmp, "suggestions", "list")
    check("t3-cli: empty list prints 0 results exit 0",
          r.returncode == 0 and "0 results" in r.stdout, f"(got {r.stdout!r})")
    # Enqueue via the core (the agent surface is MCP; the CLI owns approval).
    store = am.find_store(tmp)
    sug = am.propose_suggestion(store, "error", "refresh tokens",
                                "tokens were reusable", project="demo")
    r = _run_cli(tmp, "suggestions", "list")
    check("t3-cli: pending listed",
          r.returncode == 0 and "refresh tokens" in r.stdout
          and "state=pending" in r.stdout, f"(got {r.stdout!r})")
    r = _run_cli(tmp, "suggestions", "approve", sug["id"])
    check("t3-cli: approve without --trust exit 2", r.returncode == 2,
          f"(rc={r.returncode}, err={r.stderr!r})")
    r = _run_cli(tmp, "suggestions", "approve", sug["id"], "--trust", "approved")
    check("t3-cli: approve exit 0 + trust printed",
          r.returncode == 0 and "trust=approved" in r.stdout,
          f"(rc={r.returncode}, out={r.stdout!r})")
    r = _run_cli(tmp, "recall", "refresh")
    check("t3-cli: approved suggestion recalled", "RELEVANT CONTEXT" in r.stdout,
          f"(got {r.stdout!r})")
    # Reject path with --json.
    sug2 = am.propose_suggestion(store, "error", "bad idea",
                                 "should not become memory", project="demo")
    r = _run_cli(tmp, "suggestions", "reject", sug2["id"], "--json")
    data = json.loads(r.stdout)
    check("t3-cli: reject --json returns rejected record",
          r.returncode == 0 and data["state"] == "rejected", f"(got {r.stdout!r})")
    r = _run_cli(tmp, "suggestions", "approve", sug2["id"], "--trust", "verified")
    check("t3-cli: approve after reject exit 1", r.returncode == 1,
          f"(rc={r.returncode}, err={r.stderr!r})")
    r = _run_cli(tmp, "status", "--json")
    data = json.loads(r.stdout)
    check("t3-cli: status pending count 0", data["suggestions_pending"] == 0,
          f"(got {data})")
    r = _run_cli(tmp, "suggestions", "bogus")
    check("t3-cli: bad action exit 2", r.returncode == 2, f"(rc={r.returncode})")

# --------------------------------------------------------------------------
# T4 possible-conflict detection (EVIDENCE-038, user-locked 2026-08-14)
# --------------------------------------------------------------------------

def _conflict_pair(store, title_a, content_a, title_b, content_b,
                   type_a="decision", type_b=None, paths_a=None, paths_b=None,
                   tags_a=None, tags_b=None):
    """Create two memories + scan; return (memories dict, conflict report)."""
    type_b = type_b or type_a
    a = am.create_memory(store, type_a, title_a, content_a, project="p",
                         paths=paths_a, tags=tags_a)
    b = am.create_memory(store, type_b, title_b, content_b, project="p",
                         paths=paths_b, tags=tags_b)
    return {"a": a, "b": b}, am.scan_conflicts(store)


def test_t4_narrow_candidate_detected() -> None:
    """AC2: same type + shared scope + meaningful topical overlap -> one
    possible-conflict record with an explanation (never a verdict)."""
    store = tmp_store()
    mems, rep = _conflict_pair(
        store,
        "auth refresh token timeout",
        "token refresh invalidation session secret policy",
        "auth session invalidation",
        "session token refresh invalidation policy",
        paths_a=["src/auth/**"], paths_b=["src/auth/session.py"],
        tags_a=["auth"], tags_b=["auth"],
    )
    check("t4: one candidate from an overlapping pair", rep["candidates"] == 1,
          f"(got {rep['candidates']})")
    rec = rep["results"][0]
    check("t4: conflict id space is cf_", rec["id"].startswith("cf_"),
          f"(got {rec['id']})")
    check("t4: state open", rec["state"] == "open")
    check("t4: record references both memory ids",
          {rec["memory_a"], rec["memory_b"]} == {mems["a"]["id"], mems["b"]["id"]})
    ex = rec["explanation"]
    check("t4: shared scope paths recorded", "src/auth/**" in ex["shared_scope_paths"]
          or "src/auth/session.py" in ex["shared_scope_paths"], f"(got {ex})")
    check("t4: shared tags recorded", "auth" in ex["shared_tags"])
    check("t4: shared high-weight terms recorded (>= 2)",
          len(ex["shared_high_weight_terms"]) >= am.CONFLICT_MIN_SHARED_TERMS,
          f"(got {ex['shared_high_weight_terms']})")
    check("t4: same_type recorded", ex["same_type"] == "decision")
    check("t4: trust explains, never decides (both recorded)",
          "trust_a" in ex and "trust_b" in ex)
    check("t4: no winner field (AC4)", "winner" not in ex
          and "relationship" not in ex and "verdict" not in ex)
    check("t4: no memory mutation from scan (AC5)",
          am.list_memories(store)[0]["status"] == "active"
          and am.list_memories(store)[0]["trust"] == "untrusted")
    events = [e["event"] for e in am.read_audit(store)]
    check("t4: CONFLICT_DETECTED audited", "CONFLICT_DETECTED" in events,
          f"(got {events})")


def test_t4_scan_deterministic_and_deduped() -> None:
    """AC1/AC3: identical store -> identical scan outcome; re-scan does not
    duplicate open records; unrelated pairs are never flagged."""
    store = tmp_store()
    _conflict_pair(
        store,
        "auth refresh token timeout", "token refresh invalidation session policy",
        "auth session invalidation", "session token refresh invalidation policy",
        paths_a=["src/auth/**"], paths_b=["src/auth/session.py"],
    )
    am.create_memory(store, "decision", "frontend button layout",
                     "padding margin css layout", project="p",
                     paths=["src/frontend/**"])
    r1 = am.scan_conflicts(store)
    r2 = am.scan_conflicts(store)
    check("t4: second scan adds zero candidates", r2["candidates"] == 0,
          f"(got {r2['candidates']})")
    check("t4: second scan reports already_open", r2["already_open"] == 1,
          f"(got {r2['already_open']})")
    ids = {r["memory_a"] + r["memory_b"] for r in r2["results"]}
    unrelated = [r for r in r2["results"]
                 if "frontend" in r["memory_a"] + r["memory_b"]]
    check("t4: unrelated pair never flagged", unrelated == [], f"(got {unrelated})")


def test_t4_same_type_required() -> None:
    """AC2: same type is mandatory - a lesson sharing scope + terms with a
    decision is not a candidate."""
    store = tmp_store()
    mems, rep = _conflict_pair(
        store,
        "auth refresh token timeout", "token refresh invalidation session policy",
        "auth session invalidation", "session token refresh invalidation policy",
        type_a="decision", type_b="lesson",
        paths_a=["src/auth/**"], paths_b=["src/auth/session.py"],
    )
    check("t4: different types -> no candidate", rep["candidates"] == 0,
          f"(got {rep['candidates']})")


def test_t4_shared_scope_required() -> None:
    """AC2: shared path OR shared tag is required; overlapping text alone is
    not enough."""
    store = tmp_store()
    mems, rep = _conflict_pair(
        store,
        "auth refresh token timeout", "token refresh invalidation session policy",
        "auth session invalidation", "session token refresh invalidation policy",
    )
    check("t4: no shared scope -> no candidate", rep["candidates"] == 0,
          f"(got {rep['candidates']})")
    # shared tag only, no path overlap
    store2 = tmp_store()
    mems2, rep2 = _conflict_pair(
        store2,
        "auth refresh token timeout", "token refresh invalidation session policy",
        "auth session invalidation", "session token refresh invalidation policy",
        tags_a=["auth"], tags_b=["auth"],
    )
    check("t4: shared tag IS shared scope", rep2["candidates"] == 1,
          f"(got {rep2['candidates']})")


def test_t4_meaningful_overlap_required() -> None:
    """AC2: topical overlap must be MEANINGFUL - shared corpus-common terms
    (present in > 2/3 of the active set) do not qualify. Build a corpus where
    the two candidates share only generic terms, and a control where a
    distinctive term is shared in the same corpus."""
    store = tmp_store()
    # 6 active memories; the weak pair shares only 'auth'/'policy', which
    # appear in all 6 -> above max_df (ceil(6*2/3) = 4) -> not distinctive.
    # Each filler carries a UNIQUE token so no two fillers share a
    # distinctive term (they only share the corpus-common auth/policy).
    unique = ["zebra", "umbrella", "quasar", "giraffe"]
    for i, tok in enumerate(unique):
        am.create_memory(store, "decision", f"auth policy filler {i}",
                         f"auth policy filler {i} {tok}", project="p",
                         paths=["src/auth/**"])
    mems, rep = _conflict_pair(
        store,
        "auth refresh", "auth policy one",
        "auth session", "auth policy two",
        paths_a=["src/auth/**"], paths_b=["src/auth/**"],
    )
    check("t4: weak (corpus-common) overlap -> no candidate",
          rep["candidates"] == 0, f"(got {rep['candidates']})")
    # Control: same corpus shape, but the pair shares a distinctive term
    # ('rotation'/'token' appear only in these two) -> flagged.
    store2 = tmp_store()
    for i, tok in enumerate(unique):
        am.create_memory(store2, "decision", f"auth policy filler {i}",
                         f"auth policy filler {i} {tok}", project="p",
                         paths=["src/auth/**"])
    mems2, rep2 = _conflict_pair(
        store2,
        "refresh token rotation", "token rotation refresh rotation policy",
        "session token rotation", "token rotation session rotation policy",
        paths_a=["src/auth/**"], paths_b=["src/auth/session.py"],
    )
    check("t4: distinctive overlap still flagged in same corpus",
          rep2["candidates"] == 1, f"(got {rep2['candidates']})")


def test_t4_honest_zero() -> None:
    """AC8: empty or unrelated store -> 0 possible conflicts, exit-able cleanly."""
    store = tmp_store()
    r = am.scan_conflicts(store)
    check("t4: empty store honest zero", r["candidates"] == 0 and r["scanned"] == 0,
          f"(got {r})")


def test_t4_dismiss_lifecycle() -> None:
    """AC6: dismiss is audited and terminal while memories are unchanged; a
    later scan re-establishes the pair (observation, not authority)."""
    store = tmp_store()
    mems, rep = _conflict_pair(
        store,
        "auth refresh token timeout", "token refresh invalidation session policy",
        "auth session invalidation", "session token refresh invalidation policy",
        paths_a=["src/auth/**"], paths_b=["src/auth/session.py"],
    )
    cf = rep["results"][0]["id"]
    rec = am.dismiss_conflict(store, cf)
    check("t4: dismissed state + snapshot", rec["state"] == "dismissed"
          and rec["dismissed_by"] == "human"
          and rec["dismissed_snapshot"]["memory_a_updated_at"],
          f"(got {rec})")
    events = [e["event"] for e in am.read_audit(store)]
    check("t4: CONFLICT_DISMISSED audited", "CONFLICT_DISMISSED" in events,
          f"(got {events})")
    r = am.scan_conflicts(store)
    check("t4: dismissed-unchanged pair skipped", r["candidates"] == 0
          and r["dismissed_unchanged"] == 1, f"(got {r})")
    # A state change re-establishes the current state: supersede b by a new
    # memory c. The old a<->b observation stays dismissed (not an authority),
    # and the pair a<->b is now supersession-linked so never re-flagged. The
    # genuinely new pair a<->c IS flagged (a new observation).
    c = am.create_memory(store, "decision", "auth v2 unified",
                         "token refresh invalidation session unified policy",
                         project="p", paths=["src/auth/**"], tags=["auth"])
    am.supersede(store, mems["b"]["id"], c["id"])
    r2 = am.scan_conflicts(store)
    records = {x["id"]: x for x in am.list_conflicts(store)}
    check("t4: old a<->b record stays dismissed (not resurrected)",
          records[cf]["state"] == "dismissed", f"(got {records[cf]})")
    new_ids = [r["id"] for r in r2["results"]]
    check("t4: old dismissed record not re-opened", cf not in new_ids,
          f"(got {new_ids})")
    new_pair = [r for r in r2["results"]
                if c["id"] in (r["memory_a"], r["memory_b"])]
    check("t4: new a<->c pair flagged as fresh observation",
          len(new_pair) == 1, f"(got {new_pair})")


def test_t4_dismiss_guard() -> None:
    """AC9: only open conflicts can be dismissed; missing id is a clean error."""
    store = tmp_store()
    mems, rep = _conflict_pair(
        store,
        "auth refresh token timeout", "token refresh invalidation session policy",
        "auth session invalidation", "session token refresh invalidation policy",
        paths_a=["src/auth/**"], paths_b=["src/auth/session.py"],
    )
    cf = rep["results"][0]["id"]
    am.dismiss_conflict(store, cf)
    try:
        am.dismiss_conflict(store, cf)
        raised = False
    except am.AgentMemoryError:
        raised = True
    check("t4: re-dismiss refused (terminal)", raised)
    try:
        am.load_conflict(store, "cf_00000000-0000-0000-0000-000000000000")
        raised = False
    except am.AgentMemoryError:
        raised = True
    check("t4: missing conflict id raises clean error", raised)


def test_t4_resolve_via_supersede() -> None:
    """AC7: resolve calls the existing supersede() machinery (bidirectional
    links, no chains, MEMORY_SUPERSEDED audit) and closes the observation."""
    store = tmp_store()
    mems, rep = _conflict_pair(
        store,
        "auth refresh token timeout", "token refresh invalidation session policy",
        "auth session invalidation", "session token refresh invalidation policy",
        paths_a=["src/auth/**"], paths_b=["src/auth/session.py"],
    )
    cf = rep["results"][0]["id"]
    rec = am.resolve_conflict(store, cf, mems["a"]["id"], mems["b"]["id"])
    check("t4: conflict closed with reason superseded", rec["state"] == "closed"
          and rec["closed_reason"] == "superseded", f"(got {rec})")
    old = am.load_memory(store, mems["a"]["id"])
    new = am.load_memory(store, mems["b"]["id"])
    check("t4: supersede wired old->new",
          old["status"] == "superseded" and old["superseded_by"] == new["id"]
          and new["supersedes"] == old["id"])
    events = [e["event"] for e in am.read_audit(store)]
    check("t4: MEMORY_SUPERSEDED + CONFLICT_RESOLVED audited",
          "MEMORY_SUPERSEDED" in events and "CONFLICT_RESOLVED" in events,
          f"(got {events})")


def test_t4_resolve_guards() -> None:
    """AC7/AC9: resolve only open records, only the referenced pair."""
    store = tmp_store()
    mems, rep = _conflict_pair(
        store,
        "auth refresh token timeout", "token refresh invalidation session policy",
        "auth session invalidation", "session token refresh invalidation policy",
        paths_a=["src/auth/**"], paths_b=["src/auth/session.py"],
    )
    cf = rep["results"][0]["id"]
    other = am.create_memory(store, "decision", "frontend layout",
                             "padding margin css", project="p")
    try:
        am.resolve_conflict(store, cf, mems["a"]["id"], other["id"])
        raised = False
    except am.AgentMemoryError:
        raised = True
    check("t4: resolving a different pair refused", raised)
    am.resolve_conflict(store, cf, mems["a"]["id"], mems["b"]["id"])
    try:
        am.resolve_conflict(store, cf, mems["a"]["id"], mems["b"]["id"])
        raised = False
    except am.AgentMemoryError:
        raised = True
    check("t4: re-resolve refused (terminal)", raised)


def test_t4_status_reports_open_conflicts() -> None:
    store = tmp_store()
    _conflict_pair(
        store,
        "auth refresh token timeout", "token refresh invalidation session policy",
        "auth session invalidation", "session token refresh invalidation policy",
        paths_a=["src/auth/**"], paths_b=["src/auth/session.py"],
    )
    summary = am.status_summary(store)
    check("t4: status reports open conflict count", summary["conflicts_open"] == 1,
          f"(got {summary})")


def test_t4_cli_flow() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="agent-memory-t4-cli-"))
    _run_cli(tmp, "init", "--project", "demo")
    _run_cli(tmp, "add", "--type", "decision", "--title", "auth refresh token timeout",
             "--content", "token refresh invalidation session policy",
             "--paths", "src/auth/**", "--tags", "auth")
    _run_cli(tmp, "add", "--type", "decision", "--title", "auth session invalidation",
             "--content", "session token refresh invalidation policy",
             "--paths", "src/auth/session.py", "--tags", "auth")
    r = _run_cli(tmp, "conflicts", "scan")
    check("t4-cli: scan reports 1 new possible conflict",
          r.returncode == 0 and "1 new possible conflict" in r.stdout,
          f"(got {r.stdout!r})")
    r = _run_cli(tmp, "conflicts", "list", "--json")
    data = json.loads(r.stdout)
    check("t4-cli: list --json has 1 open",
          data["count"] == 1 and data["results"][0]["state"] == "open",
          f"(got {r.stdout!r})")
    cf = data["results"][0]["id"]
    r = _run_cli(tmp, "conflicts", "list")
    check("t4-cli: human list shows pair + state",
          "<->" in r.stdout and "state=open" in r.stdout, f"(got {r.stdout!r})")
    r = _run_cli(tmp, "conflicts", "dismiss", cf)
    check("t4-cli: dismiss exit 0", r.returncode == 0 and "dismissed" in r.stdout,
          f"(rc={r.returncode}, out={r.stdout!r})")
    r = _run_cli(tmp, "conflicts", "dismiss", cf)
    check("t4-cli: dismiss twice exit 1", r.returncode == 1,
          f"(rc={r.returncode}, err={r.stderr!r})")
    r = _run_cli(tmp, "conflicts", "bogus")
    check("t4-cli: bad action exit 2", r.returncode == 2, f"(rc={r.returncode})")


def test_t4_cli_resolve_flow() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="agent-memory-t4-cli-"))
    _run_cli(tmp, "init", "--project", "demo")
    _run_cli(tmp, "add", "--type", "decision", "--title", "auth refresh token timeout",
             "--content", "token refresh invalidation session policy",
             "--paths", "src/auth/**", "--tags", "auth")
    _run_cli(tmp, "add", "--type", "decision", "--title", "auth session invalidation",
             "--content", "session token refresh invalidation policy",
             "--paths", "src/auth/session.py", "--tags", "auth")
    _run_cli(tmp, "conflicts", "scan")
    r = _run_cli(tmp, "conflicts", "list", "--json")
    cf = json.loads(r.stdout)["results"][0]["id"]
    r = _run_cli(tmp, "list", "--json")
    mems = json.loads(r.stdout)["results"]
    old = [m["id"] for m in mems if "timeout" in m["title"]][0]
    new = [m["id"] for m in mems if "invalidation" in m["title"]][0]
    r = _run_cli(tmp, "conflicts", "resolve", cf, "--old", old, "--new", new)
    check("t4-cli: resolve exit 0", r.returncode == 0 and "superseded by" in r.stdout,
          f"(rc={r.returncode}, out={r.stdout!r})")
    r = _run_cli(tmp, "conflicts", "list", "--state", "closed")
    check("t4-cli: closed record listed", "state=closed" in r.stdout,
          f"(got {r.stdout!r})")
    r = _run_cli(tmp, "conflicts", "resolve")
    check("t4-cli: resolve missing args exit 2", r.returncode == 2,
          f"(rc={r.returncode})")
    r = _run_cli(tmp, "conflicts", "resolve", cf, "--old", old, "--new", new)
    check("t4-cli: resolve twice exit 1", r.returncode == 1,
          f"(rc={r.returncode}, err={r.stderr!r})")


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



# === Tier 2.2 lesson-log import (EVIDENCE-017) ===
# The AI-derived lesson drafts that check_logs_ai.py --lessons --apply writes
# into rules.txt section 7 (real format: bold or plain CLUSTER blocks, '---'
# separators, RULE-less clusters possible when the LLM drops one).

LESSON_FIXTURE = '## 7) LESSONS LEARNED (proposed drafts)\n\nDistilled by check_logs_ai.py - human-confirm before promoting.\n\n**CLUSTER 1 - ROOT CAUSE:** Inconsistent handling of Unicode input/output\nstreams on Windows.\n\n**CLUSTER 1 - RULE:** *Always ensure that any change affecting stdin/stdout\nencoding is paired with docs.*\n\n---\n\n**CLUSTER 2 - ROOT CAUSE:** Lack of automated enforcement for README style.\n\n**CLUSTER 2 - RULE:** *Any commit message or README formatting rule must be\nenforced by an automated check.*\n\n---\n\n**CLUSTER 3 - ROOT CAUSE:** Memory-heavy operations not optimized.\nNo rule was produced for this cluster (LLM dropped it).\n'
LEAKY_LESSON = '## 7) LESSONS LEARNED (proposed drafts)\n\nDistilled by check_logs_ai.py - human-confirm before promoting.\n\n**CLUSTER 1 - ROOT CAUSE:** Inconsistent handling of Unicode input/output\nstreams on Windows.\n\n**CLUSTER 1 - RULE:** *Always ensure that any change affecting stdin/stdout, api key ghp_12345678901234567890ab\nencoding is paired with docs.*\n\n---\n\n**CLUSTER 2 - ROOT CAUSE:** Lack of automated enforcement for README style.\n\n**CLUSTER 2 - RULE:** *Any commit message or README formatting rule must be\nenforced by an automated check.*\n\n---\n\n**CLUSTER 3 - ROOT CAUSE:** Memory-heavy operations not optimized.\nNo rule was produced for this cluster (LLM dropped it).\n'


def test_tier22_lesson_parser_rule_first_order() -> None:
    """Tier 2.2 REVIEW-REG: the spec claims order-agnostic - a RULE seen
    before its ROOT CAUSE must be buffered and attached, not crash (the
    reviewer's critical finding)."""
    rule_first = ("**CLUSTER 1 - RULE:** *Always ensure stdin/stdout changes "
                  "are documented.*\n\n"
                  "**CLUSTER 1 - ROOT CAUSE:** Inconsistent handling of "
                  "unicode streams.\n\n"
                  "---\n\n"
                  "**CLUSTER 2 - ROOT CAUSE:** No automated README "
                  "enforcement.\n\n"
                  "**CLUSTER 2 - RULE:** *Automated checks must enforce "
                  "formatting rules.*\n")
    entries = am._parse_lesson_drafts(rule_first.splitlines())
    check("tier2.2: RULE-first order does not crash and parses both",
          len(entries) == 2, f"(got {len(entries)})")
    check("tier2.2: buffered RULE attached to its cluster",
          "stdin/stdout" in entries[0]["fields"]["RULE"],
          f"(got {entries[0]['fields']['RULE'][:50]!r})")
    check("tier2.2: CAUSE-first cluster unaffected",
          "automated checks" in entries[1]["fields"]["RULE"].lower(),
          f"(got {entries[1]['fields']['RULE'][:50]!r})")


def test_tier22_lesson_parser_real_format() -> None:
    """Tier 2.2: the draft parser handles the real rules.txt section 7
    format - bold CLUSTER headers, closing ** after the colon, '---'
    separators - and skips clusters without a RULE."""
    entries = am._parse_lesson_drafts(LESSON_FIXTURE.splitlines())
    check("tier2.2: 2 complete clusters parsed (RULE-less cluster 3 skipped)",
          len(entries) == 2, f"(got {len(entries)})")
    check("tier2.2: tags are CLUSTER 1/2 in order",
          [e["tag"] for e in entries] == ["CLUSTER 1", "CLUSTER 2"])
    check("tier2.2: values stripped of bold/markdown",
          entries[0]["fields"]["ROOT CAUSE"].startswith("Inconsistent")
          and not entries[0]["fields"]["RULE"].startswith("*"),
          f"(got {entries[0]['fields']['RULE'][:40]!r})")
    check("tier2.2: title is the RULE (actionable knowledge)",
          "stdin/stdout" in entries[0]["title"])
    check("tier2.2: deterministic - second parse identical",
          am._parse_lesson_drafts(LESSON_FIXTURE.splitlines()) == entries)


def test_tier22_import_born_untrusted_import_provenance() -> None:
    """Tier 2.2 INVARIANT: AI-derived lessons are born untrusted with
    provenance=import and source.repository=agent-log-ai; type=lesson with
    ai-draft/unconfirmed tags. Import cannot manufacture trust."""
    store = tmp_store()
    am.import_source_log(store, "lesson-log", LESSON_FIXTURE, project="p")
    recs = am.list_memories(store)
    check("tier2.2: both lessons imported", len(recs) == 2, f"(got {len(recs)})")
    for r in recs:
        check(f"tier2.2: {r['title'][:30]}... born untrusted",
              r["trust"] == "untrusted", f"(got {r['trust']})")
        check(f"tier2.2: {r['title'][:30]}... provenance import",
              r["provenance"] == "import", f"(got {r['provenance']})")
        check(f"tier2.2: {r['title'][:30]}... type lesson",
              r["type"] == "lesson", f"(got {r['type']})")
        check(f"tier2.2: {r['title'][:30]}... tags ai-draft",
              "ai-draft" in r.get("tags", []) and "unconfirmed" in r.get("tags", []),
              f"(got {r.get('tags')})")
        src = r.get("source")
        check(f"tier2.2: {r['title'][:30]}... source agent-log-ai",
              isinstance(src, dict) and src.get("repository") == "agent-log-ai"
              and src.get("fingerprint", "").startswith("sha256:"),
              f"(got {src})")


def test_tier22_reimport_no_duplicates() -> None:
    """Tier 2.2 INVARIANT: re-import of the same rules.txt -> no duplicates."""
    store = tmp_store()
    r1 = am.import_source_log(store, "lesson-log", LESSON_FIXTURE, project="p")
    r2 = am.import_source_log(store, "lesson-log", LESSON_FIXTURE, project="p")
    check("tier2.2: first import created 2", r1["new"] == 2, f"(got {r1['new']})")
    check("tier2.2: re-import all duplicates",
          r2["new"] == 0 and r2["duplicates"] == 2,
          f"(new {r2['new']}, dup {r2['duplicates']})")
    check("tier2.2: total unchanged", len(am.list_memories(store)) == 2)


def test_tier22_recall_excludes_untrusted_ai_lessons() -> None:
    """Tier 2.2 DANGEROUS CASE 1: untrusted AI-derived lessons are excluded
    from recall even on a direct match; search still shows them (operator
    visibility preserved)."""
    store = tmp_store()
    am.import_source_log(store, "lesson-log", LESSON_FIXTURE, project="p")
    rec = am.recall_memories(store, "stdin encoding windows")
    check("tier2.2: recall returns 0 on untrusted AI lessons",
          rec == [], f"(got {len(rec)})")
    hits = am.search_memories(store, "stdin encoding")
    check("tier2.2: search still surfaces them for the operator",
          len(hits) >= 1, f"(got {len(hits)})")


def test_tier22_promotion_grants_recall_access() -> None:
    """Tier 2.2 DANGEROUS CASE 2: after human approval the same lesson IS
    recalled; an unpromoted sibling lesson stays excluded."""
    store = tmp_store()
    am.import_source_log(store, "lesson-log", LESSON_FIXTURE, project="p")
    recs = am.list_memories(store)
    stdin_lesson = next(r for r in recs if "stdin" in r["title"])
    am.promote_trust(store, stdin_lesson["id"], "approved")
    hits = am.recall_memories(store, "stdin encoding windows")
    check("tier2.2: approved lesson now recalled",
          len(hits) == 1 and "stdin" in hits[0]["title"],
          f"(got {[h['title'][:40] for h in hits]})")
    hits2 = am.recall_memories(store, "commit message formatting")
    check("tier2.2: unpromoted lesson still excluded",
          all("commit message" not in h["title"] for h in hits2),
          f"(got {[h['title'][:40] for h in hits2]})")


def test_tier22_dangerous_wrong_lesson_never_reaches_agent() -> None:
    """Tier 2.2 DANGEROUS CASE 3: a WRONG AI-derived lesson can never reach
    agent context unless a human promotes it. The recall contract is 'relevant
    context or nothing' - silence is the safe outcome."""
    store = tmp_store()
    # The LLM's rule is a wrong/incomplete generalization (over-generalizing
    # from one case to ALL changes) - still untrusted, still excluded.
    wrong = LESSON_FIXTURE.replace(
        "Inconsistent handling of Unicode input/output",
        "A single user asked about unicode once - irrelevant",
    )
    am.import_source_log(store, "lesson-log", wrong, project="p")
    hits = am.recall_memories(store, "unicode encoding")
    check("tier2.2: wrong untrusted lesson excluded from agent context",
          hits == [], f"(got {len(hits)})")
    shits = am.search_memories(store, "unicode encoding")
    check("tier2.2: search shows it flagged untrusted only",
          all(h["trust"] == "untrusted" for h in shits),
          f"(got {[h['trust'] for h in shits]})")


def test_tier22_secret_in_lesson_rejected_before_persist() -> None:
    """Tier 2.2 INVARIANT: secret detection runs before persistence for
    lesson imports too - an AI draft containing a leaked key is rejected,
    counted, and never stored."""
    store = tmp_store()
    report = am.import_source_log(store, "lesson-log", LEAKY_LESSON, project="p")
    check("tier2.2: leaky draft rejected, not persisted",
          report["rejected"] == 1 and report["new"] == 1,
          f"(rejected {report['rejected']}, new {report['new']})")
    check("tier2.2: rejection detail names the entry",
          len(report["rejected_details"]) == 1
          and "secret" in report["rejected_details"][0]["reason"].lower(),
          f"(got {report['rejected_details']})")
    recs = am.list_memories(store)
    check("tier2.2: only the clean lesson persisted",
          len(recs) == 1 and all("ghp_" not in str(r.get("content", "")) for r in recs))


# === Tier 2.3 rule-log import (EVIDENCE-023) ===
# The numbered RULES OF ENGAGEMENT (sections 1-6) that sit ABOVE the
# AI-derived section 7 drafts in agent-log-ai/rules.txt. Found by the T9
# BB-cell forensics: lesson-log imports only section 7, so the authoritative
# numbered rules (the AREA-marker gate) never reached the store.

RULES_FIXTURE = (
    "RULES OF ENGAGEMENT\n"
    "\n"
    "1) READ FIRST\n"
    "   Read rules.txt, then the log, then notes.txt at session start (start.py).\n"
    "\n"
    "2) LOG BEFORE FIXING\n"
    "   If something breaks, log it in the error log first, then fix.\n"
    "\n"
    "3) DECIDE BEFORE YOU CODE\n"
    "   Log decisions in decisions.txt before coding.\n"
    "\n"
    "4) NEVER REWRITE HISTORY\n"
    "   Append-only. Corrections are new entries that point back with SUPERSEDES.\n"
    "\n"
    "5) THE AREA-MARKER GATE\n"
    "   Every commit / PR title carries (AREA: <logged decision>). CI enforces it.\n"
    "\n"
    "6) LLM DISCIPLINE\n"
    "   Dry-run before you send. Never commit an API key. Local-first by default.\n"
    "\n"
    "## 7) LESSONS LEARNED (proposed drafts)\n"
    "\n"
    "Distilled by check_logs_ai.py - human-confirm before promoting.\n"
    "\n"
    "**CLUSTER 1 - ROOT CAUSE:** Inconsistent handling of Unicode streams.\n"
    "\n"
    "**CLUSTER 1 - RULE:** *Always pair encoding changes with docs.*\n"
)

LEAKY_RULE = RULES_FIXTURE.replace(
    "   Dry-run before you send. Never commit an API key. Local-first by default.",
    "   Dry-run before you send. Hook token: ghp_12345678901234567890ab.",
)


def test_rulelog_parser_real_format() -> None:
    """Tier 2.3: the numbered-rules parser reads sections 1-6 (real
    rules.txt layout: 'N) TITLE' headers, indented bodies, '## 7)'
    LESSONS marker) and stops at section 7 - the AI draft clusters belong
    to the lesson-log source, never both."""
    entries = am._parse_rules_engagement(RULES_FIXTURE.splitlines())
    check("tier2.3: exactly 6 numbered sections parsed",
          len(entries) == 6, f"(got {len(entries)})")
    check("tier2.3: tags RULE 1..6 in file order",
          [e["tag"] for e in entries] == [f"RULE {n}" for n in range(1, 7)],
          f"(got {[e['tag'] for e in entries]})")
    check("tier2.3: title is the section title",
          entries[4]["title"] == "THE AREA-MARKER GATE",
          f"(got {entries[4]['title']!r})")
    check("tier2.3: body is the indented text",
          "AREA: <logged decision>" in entries[4]["fields"]["BODY"],
          f"(got {entries[4]['fields']['BODY'][:60]!r})")
    check("tier2.3: section 7 drafts excluded (lesson source owns them)",
          all("CLUSTER" not in e["tag"] for e in entries))
    check("tier2.3: deterministic - second parse identical",
          am._parse_rules_engagement(RULES_FIXTURE.splitlines()) == entries)


def test_rulelog_import_born_untrusted_import_provenance() -> None:
    """Tier 2.3 INVARIANT: numbered rules are born untrusted with
    provenance=import and source.repository=agent-log-ai; type=constraint
    with rule/numbered tags. Import cannot manufacture trust."""
    store = tmp_store()
    am.import_source_log(store, "rule-log", RULES_FIXTURE, project="p")
    recs = am.list_memories(store)
    check("tier2.3: all 6 rules imported", len(recs) == 6, f"(got {len(recs)})")
    gate = next(r for r in recs if "AREA-MARKER" in r["title"])
    check("tier2.3: AREA gate born untrusted",
          gate["trust"] == "untrusted", f"(got {gate['trust']})")
    check("tier2.3: provenance import", gate["provenance"] == "import",
          f"(got {gate['provenance']})")
    check("tier2.3: type constraint", gate["type"] == "constraint",
          f"(got {gate['type']})")
    check("tier2.3: tags rule/numbered",
          "rule" in gate.get("tags", []) and "numbered" in gate.get("tags", []),
          f"(got {gate.get('tags')})")
    src = gate.get("source")
    check("tier2.3: source agent-log-ai + sha256 fingerprint",
          isinstance(src, dict) and src.get("repository") == "agent-log-ai"
          and src.get("fingerprint", "").startswith("sha256:"),
          f"(got {src})")


def test_rulelog_reimport_no_duplicates() -> None:
    """Tier 2.3 INVARIANT: re-import of the same rules -> no duplicates."""
    store = tmp_store()
    r1 = am.import_source_log(store, "rule-log", RULES_FIXTURE, project="p")
    r2 = am.import_source_log(store, "rule-log", RULES_FIXTURE, project="p")
    check("tier2.3: first import created 6", r1["new"] == 6, f"(got {r1['new']})")
    check("tier2.3: re-import all duplicates",
          r2["new"] == 0 and r2["duplicates"] == 6,
          f"(new {r2['new']}, dup {r2['duplicates']})")
    check("tier2.3: total unchanged", len(am.list_memories(store)) == 6)


def test_rulelog_recall_excludes_untrusted_rules() -> None:
    """Tier 2.3 DANGEROUS CASE: untrusted numbered rules are excluded from
    recall even on a direct match; search still shows them (operator
    visibility preserved)."""
    store = tmp_store()
    am.import_source_log(store, "rule-log", RULES_FIXTURE, project="p")
    rec = am.recall_memories(store, "commit message area marker")
    check("tier2.3: recall returns 0 on untrusted rules",
          rec == [], f"(got {len(rec)})")
    hits = am.search_memories(store, "area marker gate")
    check("tier2.3: search still surfaces them for the operator",
          len(hits) >= 1, f"(got {len(hits)})")


def test_rulelog_promotion_grants_recall_access() -> None:
    """Tier 2.3 DANGEROUS CASE 2: after human approval the AREA-marker rule
    IS recalled (the T9 fix); unpromoted siblings stay excluded."""
    store = tmp_store()
    am.import_source_log(store, "rule-log", RULES_FIXTURE, project="p")
    gate = next(r for r in am.list_memories(store) if "AREA-MARKER" in r["title"])
    am.promote_trust(store, gate["id"], "approved")
    hits = am.recall_memories(store, "commit message area marker")
    check("tier2.3: approved AREA rule now recalled",
          any("AREA-MARKER" in h["title"] for h in hits),
          f"(got {[h['title'][:40] for h in hits]})")


def test_rulelog_secret_in_rule_rejected_before_persist() -> None:
    """Tier 2.3 INVARIANT: secret detection gates rules too - a rule body
    containing a leaked token is rejected, counted, and never stored."""
    store = tmp_store()
    report = am.import_source_log(store, "rule-log", LEAKY_RULE, project="p")
    check("tier2.3: leaky rule rejected, not persisted",
          report["rejected"] == 1 and report["new"] == 5,
          f"(rejected {report['rejected']}, new {report['new']})")
    check("tier2.3: 5 clean rules stored",
          len(am.list_memories(store)) == 5, f"(got {len(am.list_memories(store))})")


if __name__ == "__main__":
    sys.exit(main())

