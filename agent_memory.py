"""agent_memory.py - local-first, persistent knowledge and governance layer for AI coding agents.

agent-memory gives agents persistent project memory while keeping trust, history,
and authority under system and human control. The AI can USE memory, but the AI
must NOT have unrestricted authority over memory.

v0.1 is deliberately boring: a small, deterministic, secure, inspectable memory
engine. No LLM, no embeddings, no MCP, no cloud, no vector database.

Contract: V0.1_SPEC.md in this repo. Run from a project folder:

    python agent_memory.py init [--project NAME]
    python agent_memory.py add --type decision --title T --content C [flags]
    python agent_memory.py list [--type T] [--status S] [--json]
    python agent_memory.py show <mem_id> [--json]
    python agent_memory.py search QUERY [--type T] [--status S] [--limit N] [--json]
    python agent_memory.py recall QUERY [--path P] [--limit N] [--json]
    python agent_memory.py promote <mem_id> --trust verified|approved
    python agent_memory.py supersede <old_id> <new_id>
    python agent_memory.py delete <mem_id> [--purge]
    python agent_memory.py suggestions list|approve <sug_id> --trust verified|approved|reject <sug_id>
    python agent_memory.py conflicts scan|list|dismiss <conflict_id>|resolve <conflict_id> --old <id> --new <id>
    python agent_memory.py git context <mem_id>|git list
    python agent_memory.py status [--json]

Exit codes: 0 = success, 1 = runtime error, 2 = usage error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    tomllib = None

VERSION = "0.1.0"

# --------------------------------------------------------------------------
# Constants (from V0.1_SPEC.md)
# --------------------------------------------------------------------------

FORMAT_VERSION = 1
MEMORY_PREFIX = "mem_"
AGENT_DIR = ".agent"
MEMORY_SUBDIR = "memory"
CONFIG_FILE = "config.toml"
AUDIT_FILE = "audit.jsonl"

# T3 suggestions (v0.3 Tier 3, EVIDENCE-034): a suggestion is NOT a memory.
# It lives in .agent/suggestions/ (separate subdir, separate schema - no
# trust/status), never participates in recall/trust/supersession/lifecycle,
# and only a human can convert it into a real memory. No new memory status.
SUGGESTION_PREFIX = "sug_"
SUGGESTION_SUBDIR = "suggestions"
SUGGESTION_STATES = ("pending", "approved", "rejected")

# T4 possible-conflict detection (v0.3 Tier 4, EVIDENCE-038): a conflict
# record is an OBSERVATION, not a memory and not an authority. It lives in
# .agent/conflicts/ (separate subdir, separate id space cf_), carries an
# explanation (WHY the pair was surfaced - shared scope, shared high-weight
# terms, type, trust, age) and NEVER a winner/relationship field. Only a
# human can dismiss (audited, terminal while memories are unchanged) or
# resolve via the existing supersede() machinery. No MCP surface.
CONFLICT_PREFIX = "cf_"
CONFLICT_SUBDIR = "conflicts"
CONFLICT_STATES = ("open", "dismissed", "closed")

# T5 git awareness (v0.3 Tier 5, EVIDENCE-041): write-time contextual
# evidence + deterministic retrieval enrichment. When a memory is
# created/suggested, agent-memory MAY capture a git snapshot (repository
# identity, current branch, HEAD commit, relevant changed paths, commit
# author, commit timestamp) into .agent/gitcontext/<mem_id>.json - a
# versioned SIDECAR (git-context-v1), separate from the memory schema
# (memory records stay untouched). Retrieval reads ONLY the stored
# snapshot, never live git: same store + same query + same path -> same
# recall. Fail-soft: any git failure (not a repo, git missing, command
# error) yields NO snapshot and NO error - memory operations continue
# unchanged, absence of git context means no git-derived bonus.
GITCONTEXT_SUBDIR = "gitcontext"
GITCONTEXT_SCHEMA = "git-context-v1"
GIT_MAX_CHANGED_PATHS = 50  # deterministic cap on the changed-paths list
# Minimal deterministic ranking signals (FORK 4): a memory whose STORED
# git changed-paths overlap the recall --path argument gets a bounded
# boost reusing path_tier tiers; a memory whose STORED branch equals the
# recall --branch argument gets a small bounded boost. Author/timestamp/
# commit are surfaced in the snapshot but NEVER scored (git authorship is
# evidence, not trust).
GIT_PATH_TIER_BONUS = {3: 2.0, 2: 1.5, 1: 1.0}
GIT_BRANCH_BONUS = 1.0
# Narrow detection (FORK 1): same type + shared scope + meaningful topical
# overlap. "Meaningful" = at least this many shared terms, where a shared
# term is DISTINCTIVE: it appears in at most ~half the active corpus
# (df <= ceil(n / 2), floored at 2). Corpus-relative and deterministic -
# a term that appears in most memories carries no discrimination.
CONFLICT_MIN_SHARED_TERMS = 2
CONFLICT_MAX_DF_NUMER = 1
CONFLICT_MAX_DF_DENOM = 2
CONFLICT_TERM_MIN_LEN = 3

MEMORY_TYPES = ("decision", "error", "lesson", "constraint", "architecture", "pattern")
PROVENANCES = ("human", "agent", "import", "system", "external")
TRUST_LEVELS = ("untrusted", "verified", "approved", "system")
STATUSES = ("active", "superseded", "deleted")
SEVERITIES = ("low", "normal", "high", "critical")

DEFAULT_RECALL_LIMIT = 10
DEFAULT_SEARCH_LIMIT = 50
# T2.1 path-relevance tiers (EVIDENCE-031, locked 2026-08-13): --path is a
# ranking SIGNAL, not an auto-win. Hierarchy: exact file > directory >
# glob/prefix > unrelated. Text relevance still matters - a strong text
# match without a path can outrank a weak-text exact-path match (A5).
PATH_TIER_BONUS = {3: 10.0, 2: 6.0, 1: 3.0}

# Family import (V0.1_SPEC.md section 14, v0.2 Tier 2.3 - EVIDENCE-003/016).
IMPORT_SOURCES = ("error-log", "decision-log", "lesson-log", "rule-log")
IMPORT_REPOS = {"error-log": "agent-error-log", "decision-log": "agent-decision-log",
               "lesson-log": "agent-log-ai", "rule-log": "agent-log-ai"}
IMPORT_LOG_FILES = {"error-log": "errors.txt", "decision-log": "decisions.txt",
                   "lesson-log": "rules.txt", "rule-log": "rules.txt"}

# Secret detection (V0.1_SPEC.md section 5) - best-effort, documented imperfect.
SECRET_PATTERNS = [
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"gh[ousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"sk-[A-Za-z0-9]{16,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN (?:RSA|EC|OPENSSH|DSA|PGP) PRIVATE KEY-----"),
    re.compile(r"(?:postgres|mysql|redis|amqp|mongodb)(?:\+srv)?://[^\s:/]+:[^@\s]+@"),
    re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
    re.compile(
        r"(?:password|passwd|pwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token)"
        r"\s*[=:]\s*\S{8,}",
        re.IGNORECASE,
    ),
]
ENTROPY_TOKEN_RE = re.compile(r"[A-Za-z0-9_-]{20,}")
ENTROPY_MIN = 4.5

REQUIRED_FIELDS = (
    "format_version", "id", "type", "title", "content", "project",
    "provenance", "trust", "status", "severity", "created_at", "updated_at",
)
OPTIONAL_FIELDS = (
    "tags", "paths", "source", "supersedes", "superseded_by",
    "deleted_at", "deleted_by",
)


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------

class AgentMemoryError(Exception):
    """Runtime error -> exit 1."""


class UsageError(Exception):
    """Usage error -> exit 2."""


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def reconfigure_utf8() -> None:
    """Windows consoles can crash printing non-ASCII without this (workspace rule 9)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):  # pragma: no cover
            pass


def now_utc() -> str:
    """UTC ISO-8601 with Z suffix (ROUND 3 #10)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_id() -> str:
    """mem_<uuid4> (ROUND 3 #3)."""
    return MEMORY_PREFIX + str(uuid.uuid4())


def new_suggestion_id() -> str:
    """sug_<uuid4> (T3 - distinct id space from memories)."""
    return SUGGESTION_PREFIX + str(uuid.uuid4())


def new_conflict_id() -> str:
    """cf_<uuid4> (T4 - distinct id space from memories and suggestions)."""
    return CONFLICT_PREFIX + str(uuid.uuid4())


def shannon_entropy(text: str) -> float:
    """Shannon entropy per byte over an 8-bit alphabet."""
    if not text:
        return 0.0
    counts: dict[str, int] = {}
    for ch in text:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


# --------------------------------------------------------------------------
# Path-pattern matcher (V0.1_SPEC.md 3.9a - own matcher, deterministic on 3.11)
# --------------------------------------------------------------------------

def validate_path_pattern(pattern: str) -> None:
    """Reject unsupported constructs; patterns are simple globs with * ? **."""
    if not isinstance(pattern, str) or not pattern.strip():
        raise UsageError("paths entries must be non-empty strings")
    if "[" in pattern or "]" in pattern:
        raise UsageError(
            f"path pattern {pattern!r}: character classes are not supported in v0.1"
        )


def path_matches(pattern: str, path: str) -> bool:
    """Match a path pattern (supports *, ?, **) against a forward-slash path.

    * matches within one segment (not /), ? matches one char (not /),
    ** matches any number of segments including zero.
    """
    pattern = pattern.replace("\\", "/").strip("/")
    path = path.replace("\\", "/").strip("/")
    if not path:
        return False
    if pattern == "**":
        return True
    return _match_segments(pattern.split("/"), path.split("/"))


def path_tier(pattern: str, path: str) -> int:
    """Path-relevance tier of a memory pattern against a file path (T2.1).

    Returns 3 (exact file) / 2 (bare directory containing the path) /
    1 (glob/prefix match) / 0 (unrelated or neutral). A bare '**' pattern
    matches everything, so it carries no path specificity - treated as
    neutral (0), same as a memory with no paths (EVIDENCE-031 A6: a bare
    directory like 'src/auth' now routes to files under it; previously
    path_matches required the full segment list, so directories matched
    nothing useful).
    """
    pattern = pattern.replace("\\", "/").strip("/")
    path = path.replace("\\", "/").strip("/")
    if not path or not pattern or pattern == "**":
        return 0
    if pattern == path:
        return 3  # exact file
    if not any(ch in pattern for ch in "*?"):
        # bare literal pattern: directory containing the path (prefix + '/')
        # guards against sibling-prefix collisions (src/auth vs src/auth2).
        if path.startswith(pattern + "/"):
            return 2
        return 0
    if path_matches(pattern, path):
        return 1  # glob/prefix
    return 0


def _match_segments(pat: list[str], parts: list[str]) -> bool:
    """Recursive glob matching over path segments; '**' matches any depth."""
    if not pat:
        return not parts
    if pat[0] == "**":
        for skip in range(len(parts) + 1):
            if _match_segments(pat[1:], parts[skip:]):
                return True
        return False
    if not parts:
        return False
    return _match_segment(pat[0], parts[0]) and _match_segments(pat[1:], parts[1:])


def _match_segment(pattern: str, segment: str) -> bool:
    """Match a single path segment pattern (* and ? wildcards, no /)."""
    return re.fullmatch(_segment_regex(pattern), segment) is not None


def _segment_regex(pattern: str) -> str:
    out: list[str] = []
    for ch in pattern:
        if ch == "*":
            out.append("[^/]*")
        elif ch == "?":
            out.append("[^/]")
        else:
            out.append(re.escape(ch))
    return "".join(out)


# --------------------------------------------------------------------------
# Secret detection (V0.1_SPEC.md section 5)
# --------------------------------------------------------------------------

def detect_secret(title: str, content: str) -> str | None:
    """Return the matched pattern description if a secret is detected, else None."""
    haystack = f"{title}\n{content}"
    for pat in SECRET_PATTERNS:
        if pat.search(haystack):
            return f"matched pattern: {pat.pattern[:60]}"
    for tok in ENTROPY_TOKEN_RE.findall(haystack):
        if shannon_entropy(tok) > ENTROPY_MIN:
            return f"high-entropy token ({len(tok)} chars)"
    return None


# --------------------------------------------------------------------------
# Schema validation (V0.1_SPEC.md sections 3 + 6)
# --------------------------------------------------------------------------

def _validate_source(source: dict) -> None:
    """Source-provenance dict (spec 3.9): type/repository/fingerprint strings."""
    if not isinstance(source, dict):
        raise UsageError("source must be a dict or null")
    for key in ("type", "repository", "fingerprint"):
        if not isinstance(source.get(key), str) or not source[key]:
            raise UsageError(f"source.{key} must be a non-empty string")
    if not source["fingerprint"].startswith("sha256:"):
        raise UsageError("source.fingerprint must start with 'sha256:'")
    if "tag" in source and (not isinstance(source["tag"], str) or not source["tag"]):
        raise UsageError("source.tag must be a non-empty string when present")


def validate_memory(record: dict) -> None:
    """Validate a memory record. Raises UsageError with a clean message on failure."""
    if record.get("format_version") != FORMAT_VERSION:
        raise UsageError(
            f"invalid format_version {record.get('format_version')!r}: must be {FORMAT_VERSION}"
        )
    if not isinstance(record.get("id"), str) or not record["id"].startswith(MEMORY_PREFIX):
        raise UsageError(f"invalid id {record.get('id')!r}: must be mem_<uuid>")
    if record.get("type") not in MEMORY_TYPES:
        raise UsageError(
            f"invalid type {record.get('type')!r}: must be one of {', '.join(MEMORY_TYPES)}"
        )
    for field in ("title", "content", "project"):
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            raise UsageError(f"{field} must be a non-empty string")
    if record.get("provenance") not in PROVENANCES:
        raise UsageError(
            f"invalid provenance {record.get('provenance')!r}: "
            f"must be one of {', '.join(PROVENANCES)}"
        )
    if record.get("trust") not in TRUST_LEVELS:
        raise UsageError(
            f"invalid trust {record.get('trust')!r}: must be one of {', '.join(TRUST_LEVELS)}"
        )
    if record.get("status") not in STATUSES:
        raise UsageError(
            f"invalid status {record.get('status')!r}: must be one of {', '.join(STATUSES)}"
        )
    if record.get("severity") not in SEVERITIES:
        raise UsageError(
            f"invalid severity {record.get('severity')!r}: must be one of {', '.join(SEVERITIES)}"
        )
    for field in ("created_at", "updated_at"):
        value = record.get(field)
        if not isinstance(value, str) or not _is_utc_timestamp(value):
            raise UsageError(f"{field} must be a UTC ISO-8601 timestamp with Z")
    if "tags" in record:
        if not isinstance(record["tags"], list) or not all(
            isinstance(t, str) for t in record["tags"]
        ):
            raise UsageError("tags must be a list of strings")
    if "paths" in record:
        if not isinstance(record["paths"], list):
            raise UsageError("paths must be a list of strings")
        for p in record["paths"]:
            validate_path_pattern(p)
    for link in ("supersedes", "superseded_by"):
        if record.get(link) is not None and not isinstance(record[link], str):
            raise UsageError(f"{link} must be a string or null")


def _is_utc_timestamp(value: str) -> bool:
    return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value))


# --------------------------------------------------------------------------
# Storage (V0.1_SPEC.md section 2 + 12)
# --------------------------------------------------------------------------

def find_store(start: pathlib.Path | None = None) -> pathlib.Path:
    """Walk up from the given dir (default cwd) looking for .agent/. None found -> error."""
    cur = (start or pathlib.Path.cwd()).resolve()
    for folder in (cur, *cur.parents):
        candidate = folder / AGENT_DIR
        if candidate.is_dir():
            return candidate
    raise AgentMemoryError(
        "not in an agent-memory project (no .agent/ found in this directory or any parent); "
        "run: agent-memory init"
    )


def _write_text_utf8(path: pathlib.Path, text: str) -> None:
    """UTF-8 write with newline='\\n' - byte-identical on Windows and POSIX.

    Text mode without newline='\\n' would translate \\n to \\r\\n on Windows,
    breaking the byte-identical determinism contract (V0.1_SPEC.md 13/15.8).
    """
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def init_store(target: pathlib.Path | None = None, project: str | None = None, force: bool = False) -> pathlib.Path:
    """Create the .agent/ store in the given dir (default cwd)."""
    root = (target or pathlib.Path.cwd()).resolve()
    store = root / AGENT_DIR
    if store.exists() and not force:
        raise AgentMemoryError(
            f"an agent-memory store already exists at {store}; use --force only to re-init an empty store"
        )
    if store.exists() and force:
        # Only allow re-init if the existing store holds nothing but a valid layout.
        existing = [p for p in store.rglob("*") if p.is_file()]
        if existing:
            raise AgentMemoryError(
                f"refusing to re-init: existing store at {store} is not empty "
                f"({len(existing)} file(s))"
            )
        store.mkdir(parents=True, exist_ok=True)
    else:
        store.mkdir(parents=True, exist_ok=True)
    for t in MEMORY_TYPES:
        (store / MEMORY_SUBDIR / t).mkdir(parents=True, exist_ok=True)
    (store / SUGGESTION_SUBDIR).mkdir(parents=True, exist_ok=True)
    (store / CONFLICT_SUBDIR).mkdir(parents=True, exist_ok=True)
    project_name = project or root.name
    config = {"project": project_name, "recall_limit": DEFAULT_RECALL_LIMIT, "include_untrusted": False}
    _write_text_utf8(store / CONFIG_FILE, _dump_toml(config))
    if not (store / AUDIT_FILE).exists():
        _write_text_utf8(store / AUDIT_FILE, "")
    return store


def _dump_toml(config: dict) -> str:
    lines = [f'project = {json.dumps(config["project"], ensure_ascii=False)}']
    lines.append(f"recall_limit = {config['recall_limit']}")
    lines.append(f"include_untrusted = {str(config['include_untrusted']).lower()}")
    return "\n".join(lines) + "\n"


def load_config(store: pathlib.Path) -> dict:
    """Read config.toml; missing file -> defaults; malformed -> clean error."""
    defaults = {"project": store.parent.name, "recall_limit": DEFAULT_RECALL_LIMIT, "include_untrusted": False}
    path = store / CONFIG_FILE
    if not path.exists():
        return defaults
    if tomllib is None:  # pragma: no cover
        raise AgentMemoryError("tomllib unavailable (requires Python >= 3.11)")
    try:
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
    except (tomllib.TOMLDecodeError, OSError) as exc:
        raise AgentMemoryError(f"malformed config.toml: {exc}") from exc
    merged = dict(defaults)
    merged.update({k: v for k, v in data.items() if k in defaults})
    return merged


def memory_path(store: pathlib.Path, mem_type: str, mem_id: str) -> pathlib.Path:
    return store / MEMORY_SUBDIR / mem_type / f"{mem_id}.json"


def load_memory(store: pathlib.Path, mem_id: str) -> dict:
    """Load a memory record by id (searches all type dirs)."""
    for t in MEMORY_TYPES:
        path = memory_path(store, t, mem_id)
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                raise AgentMemoryError(f"corrupt memory file {path}: {exc}") from exc
    raise AgentMemoryError(f"no memory with id {mem_id}")


def find_memory_path(store: pathlib.Path, mem_id: str) -> pathlib.Path | None:
    for t in MEMORY_TYPES:
        path = memory_path(store, t, mem_id)
        if path.exists():
            return path
    return None


def save_memory(store: pathlib.Path, record: dict) -> pathlib.Path:
    path = memory_path(store, record["type"], record["id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_text_utf8(path, json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    return path


# --------------------------------------------------------------------------
# Audit log (V0.1_SPEC.md section 11)
# --------------------------------------------------------------------------

def append_audit(store: pathlib.Path, event: str, memory_id: str | None, actor: str, detail: dict | None = None) -> None:
    """Append one audit event. Append-only by contract: never rewrites lines."""
    record = {
        "at": now_utc(),
        "event": event,
        "memory_id": memory_id,
        "actor": actor,
        "detail": detail or {},
    }
    with open(store / AUDIT_FILE, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_audit(store: pathlib.Path) -> list[dict]:
    """Read all audit events (used by status/tests)."""
    path = store / AUDIT_FILE
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:  # pragma: no cover - corrupted audit is a defect
            raise AgentMemoryError("corrupt audit.jsonl: non-JSON line found")
    return events


# --------------------------------------------------------------------------
# Core operations
# --------------------------------------------------------------------------

def create_memory(
    store: pathlib.Path,
    mem_type: str,
    title: str,
    content: str,
    project: str,
    provenance: str = "human",
    severity: str = "normal",
    tags: list[str] | None = None,
    paths: list[str] | None = None,
    source: dict | None = None,
    allow_secret: bool = False,
    actor: str = "human",
    git_context: dict | None = None,
) -> dict:
    """Validate + secret-scan + persist a new memory. Raises on rejection.

    git_context (T5, EVIDENCE-041): an optional PRE-CAPTURED snapshot to
    persist instead of capturing fresh at write time - used by
    approve_suggestion so the approved memory keeps the git context of
    WHERE the proposal was made (the snapshot travels with the suggestion).
    """
    if mem_type not in MEMORY_TYPES:
        raise UsageError(f"invalid type {mem_type!r}: must be one of {', '.join(MEMORY_TYPES)}")
    if provenance not in PROVENANCES:
        raise UsageError(
            f"invalid provenance {provenance!r}: must be one of {', '.join(PROVENANCES)}"
        )
    if provenance == "system":
        raise UsageError("provenance 'system' is application-internal and cannot be set via the CLI")
    if severity not in SEVERITIES:
        raise UsageError(f"invalid severity {severity!r}: must be one of {', '.join(SEVERITIES)}")
    if source is not None:
        _validate_source(source)
    if not isinstance(title, str) or not title.strip():
        raise UsageError("title must be a non-empty string")
    if not isinstance(content, str) or not content.strip():
        raise UsageError("content must be a non-empty string")

    detected = detect_secret(title, content)
    if detected and not allow_secret:
        append_audit(store, "MEMORY_REJECTED", None, actor, {"reason": "secret_detected", "detail": detected})
        raise AgentMemoryError("Memory rejected: potential secret detected.")

    timestamp = now_utc()
    record = {
        "format_version": FORMAT_VERSION,
        "id": new_id(),
        "type": mem_type,
        "title": title.strip(),
        "content": content.strip(),
        "project": project,
        "provenance": provenance,
        "trust": "untrusted",
        "status": "active",
        "severity": severity,
        "tags": tags or [],
        "paths": paths or [],
        "created_at": timestamp,
        "updated_at": timestamp,
        "source": source,
        "supersedes": None,
        "superseded_by": None,
        "deleted_at": None,
        "deleted_by": None,
    }
    for p in record["paths"]:
        validate_path_pattern(p)
    validate_memory(record)
    save_memory(store, record)
    detail = {"secret_override": True} if (detected and allow_secret) else {}
    append_audit(store, "MEMORY_CREATED", record["id"], actor, detail)
    if detected and allow_secret:
        append_audit(store, "SECRET_OVERRIDE", record["id"], actor, {"detail": detected})
    # T5 (EVIDENCE-041): persist a git snapshot sidecar, fail-soft. Never
    # raises, never blocks the memory write (no git context is simply no
    # git-derived bonus). An explicit git_context (from suggestion approval)
    # wins over a fresh capture, so the approved memory keeps the proposal's
    # context.
    try:
        if git_context is not None:
            save_git_context(store, record["id"], git_context)
        else:
            snap = capture_git_context(store)
            if snap is not None:
                save_git_context(store, record["id"], snap)
    except Exception:  # noqa: BLE001 - fail-soft by contract
        pass
    return record


def promote_trust(store: pathlib.Path, mem_id: str, target: str, actor: str = "human") -> dict:
    """Human-only trust promotion. Raises on invalid transitions."""
    if target not in ("verified", "approved"):
        raise UsageError("promote --trust must be 'verified' or 'approved' (never 'system')")
    record = load_memory(store, mem_id)
    if record["status"] != "active":
        raise AgentMemoryError(
            f"cannot promote {mem_id}: status is {record['status']}; only active memories can be promoted"
        )
    current = record["trust"]
    if current == target:
        raise AgentMemoryError(f"memory {mem_id} is already trust={target}")
    if current not in ("untrusted", "verified"):
        raise AgentMemoryError(
            f"cannot promote trust from {current!r}: promotion is only from untrusted/verified"
        )
    if current == "verified" and target != "approved":
        raise AgentMemoryError(f"cannot promote from verified to {target!r}")
    record["trust"] = target
    record["updated_at"] = now_utc()
    save_memory(store, record)
    append_audit(store, "TRUST_PROMOTED", mem_id, actor, {"old_trust": current, "new_trust": target})
    return record


def supersede(store: pathlib.Path, old_id: str, new_id: str, actor: str = "human") -> dict:
    """Mark old superseded by new; bidirectional links; no chains (V0.1_SPEC.md 10)."""
    if old_id == new_id:
        raise UsageError("supersede requires two different memory ids")
    old = load_memory(store, old_id)
    new = load_memory(store, new_id)
    if old["status"] != "active":
        raise AgentMemoryError(f"old memory {old_id} is {old['status']}; only active can be superseded")
    if new["status"] != "active":
        raise AgentMemoryError(f"new memory {new_id} is {new['status']}; must be active")
    if new.get("supersedes"):
        raise AgentMemoryError(
            f"new memory {new_id} already supersedes {new['supersedes']}; no chains in v0.1"
        )
    timestamp = now_utc()
    old["status"] = "superseded"
    old["superseded_by"] = new_id
    old["updated_at"] = timestamp
    new["supersedes"] = old_id
    new["updated_at"] = timestamp
    save_memory(store, old)
    save_memory(store, new)
    append_audit(store, "MEMORY_SUPERSEDED", old_id, actor, {"new_id": new_id})
    return old


def delete_memory(store: pathlib.Path, mem_id: str, purge: bool = False, actor: str = "human") -> dict:
    """Tombstone delete; physical purge only for untrusted memories."""
    record = load_memory(store, mem_id)
    if record["status"] == "deleted":
        raise AgentMemoryError(f"memory {mem_id} is already deleted")
    if purge:
        if record["trust"] != "untrusted":
            raise AgentMemoryError(
                f"cannot purge {mem_id}: trust={record['trust']}; purge is only for untrusted memories"
            )
        path = find_memory_path(store, mem_id)
        if path is not None:
            path.unlink()
        append_audit(store, "MEMORY_DELETED", mem_id, actor, {"purged": True})
        return record
    timestamp = now_utc()
    record["status"] = "deleted"
    record["deleted_at"] = timestamp
    record["deleted_by"] = actor
    record["updated_at"] = timestamp
    save_memory(store, record)
    append_audit(store, "MEMORY_DELETED", mem_id, actor, {"purged": False})
    return record


# --------------------------------------------------------------------------
# T3 suggestions (v0.3 Tier 3, EVIDENCE-034 - approve-to-persist loop)
# --------------------------------------------------------------------------
# A suggestion is a CANDIDATE proposed by an agent, not a memory: it carries
# no trust/status, never participates in recall/trust ranking/supersession/
# lifecycle/history until a HUMAN converts it into a real memory. The queue
# must not become a backdoor persistence mechanism: propose validates +
# secret-screens BEFORE writing, pins provenance to agent, never assigns
# trust, and approval/rejection are human-only, audited, and TERMINAL.

def propose_suggestion(
    store: pathlib.Path,
    mem_type: str,
    title: str,
    content: str,
    project: str,
    severity: str = "normal",
    tags: list[str] | None = None,
    paths: list[str] | None = None,
    actor: str = "agent",
) -> dict:
    """Validate + secret-screen + enqueue a pending suggestion.

    A suggestion is NOT a memory: it is written to .agent/suggestions/
    sug_<uuid>.json with NO trust/status fields (the memory schema is
    untouched). provenance is hard-pinned to 'agent' - only agents propose
    suggestions. Secret material is rejected BEFORE any write (never
    persisted) and audited as SUGGESTION_REJECTED. Raises on rejection.
    """
    if mem_type not in MEMORY_TYPES:
        raise UsageError(f"invalid type {mem_type!r}: must be one of {', '.join(MEMORY_TYPES)}")
    if severity not in SEVERITIES:
        raise UsageError(f"invalid severity {severity!r}: must be one of {', '.join(SEVERITIES)}")
    if not isinstance(title, str) or not title.strip():
        raise UsageError("title must be a non-empty string")
    if not isinstance(content, str) or not content.strip():
        raise UsageError("content must be a non-empty string")
    for p in (paths or []):
        validate_path_pattern(p)

    detected = detect_secret(title, content)
    if detected:
        append_audit(store, "SUGGESTION_REJECTED", None, actor,
                     {"reason": "secret_detected", "detail": detected})
        raise AgentMemoryError("Suggestion rejected: potential secret detected.")

    timestamp = now_utc()
    # T5 (EVIDENCE-041): capture a write-time git snapshot and embed it in
    # the suggestion record (fail-soft: None when not a git repo). It
    # travels with the suggestion into approval conversion, so an approved
    # memory keeps the context of WHERE the proposal was made.
    try:
        git_context = capture_git_context(store)
    except Exception:  # noqa: BLE001 - fail-soft by contract
        git_context = None
    sug = {
        "format_version": FORMAT_VERSION,
        "id": new_suggestion_id(),
        "type": mem_type,
        "title": title.strip(),
        "content": content.strip(),
        "project": project,
        "provenance": "agent",  # pinned: only agents propose suggestions
        "severity": severity,
        "tags": tags or [],
        "paths": paths or [],
        "created_at": timestamp,
        "state": "pending",
        "approved_at": None,
        "approved_by": None,
        "rejected_at": None,
        "rejected_by": None,
        "git_context": git_context,
    }
    path = store / SUGGESTION_SUBDIR / f"{sug['id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_text_utf8(path, json.dumps(sug, indent=2, ensure_ascii=False) + "\n")
    append_audit(store, "SUGGESTION_CREATED", None, actor,
                 {"suggestion_id": sug["id"]})
    return sug


def load_suggestion(store: pathlib.Path, sug_id: str) -> dict:
    """Load one suggestion by id."""
    path = store / SUGGESTION_SUBDIR / f"{sug_id}.json"
    if not path.exists():
        raise AgentMemoryError(f"no suggestion with id {sug_id}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise AgentMemoryError(f"corrupt suggestion file {path}: {exc}") from exc


def list_suggestions(store: pathlib.Path, state: str | None = None) -> list[dict]:
    """List suggestions, newest first (id as final tie-break for determinism)."""
    folder = store / SUGGESTION_SUBDIR
    suggestions: list[dict] = []
    if folder.is_dir():
        for f in folder.glob("*.json"):
            try:
                suggestions.append(json.loads(f.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError):
                raise AgentMemoryError(f"corrupt suggestion file {f}")
    if state:
        suggestions = [s for s in suggestions if s.get("state") == state]
    suggestions.sort(key=lambda s: (s.get("created_at", ""), s.get("id", "")),
                     reverse=True)
    return suggestions


def approve_suggestion(store: pathlib.Path, sug_id: str, trust: str, actor: str = "human") -> dict:
    """Human-only conversion of a pending suggestion into a real memory.

    The human EXPLICITLY chooses the resulting trust level (verified or
    approved - never 'system'); approval is NOT automatically max trust.
    The created memory is born untrusted then promoted through the existing
    trust ladder (promote_trust), so the trust-transition rules govern. The
    stored proposal is re-validated and re-screened for secrets (defense
    against disk tampering). Terminal: only pending suggestions can be
    approved. Returns the created memory record.
    """
    if trust not in ("verified", "approved"):
        raise UsageError("approve --trust must be 'verified' or 'approved' (never 'system')")
    sug = load_suggestion(store, sug_id)
    if sug.get("state") != "pending":
        raise AgentMemoryError(
            f"cannot approve {sug_id}: state is {sug.get('state')}; only pending suggestions can be approved"
        )
    detected = detect_secret(sug.get("title", ""), sug.get("content", ""))
    if detected:
        raise AgentMemoryError(
            f"refusing to approve {sug_id}: secret detected in the stored proposal"
        )
    record = create_memory(
        store=store,
        mem_type=sug["type"],
        title=sug["title"],
        content=sug["content"],
        project=sug.get("project", store.parent.name),
        provenance="agent",  # the knowledge originated from the agent proposal
        severity=sug.get("severity", "normal"),
        tags=sug.get("tags", []),
        paths=sug.get("paths", []),
        actor=actor,
        git_context=sug.get("git_context"),  # T5: the proposal's git snapshot travels with it
    )
    if record["trust"] != trust:
        record = promote_trust(store, record["id"], trust, actor=actor)
    timestamp = now_utc()
    sug["state"] = "approved"
    sug["approved_at"] = timestamp
    sug["approved_by"] = actor
    _write_text_utf8(store / SUGGESTION_SUBDIR / f"{sug_id}.json",
                     json.dumps(sug, indent=2, ensure_ascii=False) + "\n")
    append_audit(store, "SUGGESTION_APPROVED", record["id"], actor,
                 {"suggestion_id": sug_id, "trust": trust})
    return record


def reject_suggestion(store: pathlib.Path, sug_id: str, actor: str = "human") -> dict:
    """Human-only terminal rejection: discards the proposal, audited."""
    sug = load_suggestion(store, sug_id)
    if sug.get("state") != "pending":
        raise AgentMemoryError(
            f"cannot reject {sug_id}: state is {sug.get('state')}; only pending suggestions can be rejected"
        )
    timestamp = now_utc()
    sug["state"] = "rejected"
    sug["rejected_at"] = timestamp
    sug["rejected_by"] = actor
    _write_text_utf8(store / SUGGESTION_SUBDIR / f"{sug_id}.json",
                     json.dumps(sug, indent=2, ensure_ascii=False) + "\n")
    append_audit(store, "SUGGESTION_REJECTED", None, actor,
                 {"suggestion_id": sug_id, "reason": "human_review"})
    return sug


# --------------------------------------------------------------------------
# T4 possible-conflict detection (v0.3 Tier 4, EVIDENCE-038)
# --------------------------------------------------------------------------
# A conflict record is an OBSERVATION, not an authority: it surfaces pairs
# that deserve human attention (same type + shared scope + meaningful
# topical overlap). It never decides a winner, never writes memory records,
# and only a human can dismiss it (audited, terminal while memories are
# unchanged) or resolve it via the existing supersede() machinery. Scan is
# on-demand and deterministic; there is NO write-time enforcement.


def _conflict_term_set(record: dict) -> set[str]:
    """Lowercased tokens (len >= 3) from title + tags + content, for IDF."""
    blob = " ".join((
        record.get("title", ""),
        " ".join(record.get("tags", [])),
        record.get("content", ""),
    )).lower()
    return {t for t in re.split(r"[^a-z0-9]+", blob) if len(t) >= CONFLICT_TERM_MIN_LEN}


def _paths_share_scope(a: list[str], b: list[str]) -> bool:
    """True if any pattern in a matches any pattern in b (either direction)."""
    for pa in a:
        for pb in b:
            if path_tier(pa, pb) > 0 or path_tier(pb, pa) > 0:
                return True
    return False


def _tags_share_scope(a: list[str], b: list[str]) -> bool:
    return bool(set(a) & set(b))


def _conflict_explanation(a: dict, b: dict,
                          shared_terms: list[str], idf: dict[str, float]) -> dict:
    """Deterministic WHY record - explanation only, never a verdict.

    EVIDENCE-038 FORK 2: NO winner field, NO relationship field - the
    detector cannot establish either. Trust and age explain the candidate;
    they never decide which memory (if either) is correct.
    """
    shared_paths = [p for p in a.get("paths", []) if any(
        path_tier(p, q) > 0 or path_tier(q, p) > 0 for q in b.get("paths", []))]
    shared_tags = sorted(set(a.get("tags", [])) & set(b.get("tags", [])))
    return {
        "shared_scope_paths": shared_paths,
        "shared_tags": shared_tags,
        "shared_high_weight_terms": sorted(shared_terms),
        "overlap_score": round(sum(idf[t] for t in shared_terms), 6),
        "same_type": a["type"],
        "trust_a": a["trust"],
        "trust_b": b["trust"],
        "age_a": a["created_at"],
        "age_b": b["created_at"],
        "reason": "overlapping scope + topical similarity",
    }


def scan_conflicts(store: pathlib.Path, actor: str = "human") -> dict:
    """On-demand deterministic scan: find pairs that deserve human review.

    Returns a report dict (never writes memory records). For each new
    candidate pair it writes one .agent/conflicts/cf_<uuid>.json record and
    audits CONFLICT_DETECTED. Narrow (EVIDENCE-038 FORK 1): same type +
    shared scope (path pattern overlap OR shared tag) + meaningful topical
    overlap (>= CONFLICT_MIN_SHARED_TERMS terms with idf >=
    CONFLICT_IDF_FLOOR). Ineligible: already supersession-linked, identical
    ids, or previously dismissed AND both memories unchanged. Dismissed
    pairs whose underlying memories have changed are re-eligible (a
    conflict is an observation, not an authority - lifecycle rule).
    Deterministic: pair iteration in (id) order, stable scoring, stable
    output ordering.
    """
    folder = store / CONFLICT_SUBDIR
    folder.mkdir(parents=True, exist_ok=True)
    active = [r for r in list_memories(store) if r.get("status") == "active"]
    active.sort(key=lambda r: r["id"])
    existing = _load_conflicts(store)
    open_pairs: set[tuple[str, str]] = set()
    dismissed_unchanged: set[tuple[str, str]] = set()
    state_closed: list[dict] = []
    for rec in existing:
        pair = tuple(sorted((rec["memory_a"], rec["memory_b"])))
        if rec.get("state") != "open":
            if rec.get("state") == "dismissed":
                try:
                    a = load_memory(store, rec["memory_a"])
                    b = load_memory(store, rec["memory_b"])
                except AgentMemoryError:
                    continue  # a memory was purged; observation is stale, re-eligible
                snap = rec.get("dismissed_snapshot", {})
                if (a.get("updated_at") == snap.get("memory_a_updated_at")
                        and b.get("updated_at") == snap.get("memory_b_updated_at")):
                    dismissed_unchanged.add(pair)
            continue
        # Open record: re-establish the CURRENT state - close the observation
        # if the pair is now supersession-linked or either memory is no longer
        # active (deleted/purged). A conflict is an observation, not authority.
        try:
            a = load_memory(store, rec["memory_a"])
            b = load_memory(store, rec["memory_b"])
        except AgentMemoryError:
            state_closed.append((rec, "memory_removed"))
            continue
        if a.get("status") != "active" or b.get("status") != "active":
            state_closed.append((rec, "memory_no_longer_active"))
            continue
        if _supersession_linked(a, b):
            state_closed.append((rec, "pair_superseded"))
            continue
        open_pairs.add(pair)
    for rec, reason in state_closed:
        rec["state"] = "closed"
        rec["closed_at"] = now_utc()
        rec["closed_by"] = actor
        rec["closed_reason"] = reason
        _write_text_utf8(store / CONFLICT_SUBDIR / f"{rec['id']}.json",
                         json.dumps(rec, indent=2, ensure_ascii=False) + "\n")
        append_audit(store, "CONFLICT_CLOSED", None, actor, {
            "conflict_id": rec["id"],
            "memory_a": rec["memory_a"],
            "memory_b": rec["memory_b"],
            "reason": reason,
        })

    all_terms = _all_terms(active)
    df = _term_doc_freqs(active, all_terms)
    idf = _idf_weights(len(active), df)
    max_df = max(2, (len(active) * CONFLICT_MAX_DF_NUMER + CONFLICT_MAX_DF_DENOM - 1)
                 // CONFLICT_MAX_DF_DENOM)
    created: list[dict] = []
    for i in range(len(active)):
        for j in range(i + 1, len(active)):
            a, b = active[i], active[j]
            pair = (a["id"], b["id"])
            if pair in open_pairs or pair in dismissed_unchanged:
                continue
            if _supersession_linked(a, b):
                continue
            if a["type"] != b["type"]:
                continue  # narrow: same type required (FORK 1)
            if not (_paths_share_scope(a.get("paths", []), b.get("paths", []))
                    or _tags_share_scope(a.get("tags", []), b.get("tags", []))):
                continue  # narrow: shared scope required
            terms_a = _conflict_term_set(a)
            terms_b = _conflict_term_set(b)
            shared = [t for t in all_terms if t in terms_a and t in terms_b
                      and df[t] <= max_df]
            if len(shared) < CONFLICT_MIN_SHARED_TERMS:
                continue  # narrow: meaningful topical overlap required
            record = {
                "format_version": FORMAT_VERSION,
                "id": new_conflict_id(),
                "state": "open",
                "memory_a": a["id"],
                "memory_b": b["id"],
                "created_at": now_utc(),
                "dismissed_at": None,
                "dismissed_by": None,
                "dismissed_snapshot": None,
                "closed_at": None,
                "closed_by": None,
                "closed_reason": None,
                "explanation": _conflict_explanation(a, b, shared, idf),
            }
            _write_text_utf8(folder / f"{record['id']}.json",
                             json.dumps(record, indent=2, ensure_ascii=False) + "\n")
            append_audit(store, "CONFLICT_DETECTED", None, actor, {
                "conflict_id": record["id"],
                "memory_a": a["id"],
                "memory_b": b["id"],
                "type": a["type"],
            })
            created.append(record)
    created.sort(key=lambda r: (r["created_at"], r["id"]))
    return {
        "scanned": len(active),
        "candidates": len(created),
        "already_open": len(open_pairs),
        "dismissed_unchanged": len(dismissed_unchanged),
        "closed_stale": len(state_closed),
        "results": created,
    }


def _all_terms(records: list[dict]) -> list[str]:
    """Deterministic sorted union of all memory term sets (df denominator)."""
    terms: set[str] = set()
    for r in records:
        terms |= _conflict_term_set(r)
    return sorted(terms)


def _supersession_linked(a: dict, b: dict) -> bool:
    return (a.get("supersedes") == b.get("id") or b.get("supersedes") == a.get("id")
            or a.get("superseded_by") == b.get("id") or b.get("superseded_by") == a.get("id"))


def _load_conflicts(store: pathlib.Path) -> list[dict]:
    """All conflict records, sorted by (created_at, id) for determinism."""
    folder = store / CONFLICT_SUBDIR
    records: list[dict] = []
    if folder.is_dir():
        for f in folder.glob("*.json"):
            try:
                records.append(json.loads(f.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError) as exc:
                raise AgentMemoryError(f"corrupt conflict file {f}: {exc}") from exc
    records.sort(key=lambda r: (r.get("created_at", ""), r.get("id", "")))
    return records


def list_conflicts(store: pathlib.Path, state: str | None = None) -> list[dict]:
    """List conflict records, newest first (state filter optional)."""
    records = _load_conflicts(store)
    if state:
        records = [r for r in records if r.get("state") == state]
    return list(reversed(records))


def load_conflict(store: pathlib.Path, conflict_id: str) -> dict:
    """Load one conflict record by id."""
    path = store / CONFLICT_SUBDIR / f"{conflict_id}.json"
    if not path.exists():
        raise AgentMemoryError(f"no conflict record with id {conflict_id}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise AgentMemoryError(f"corrupt conflict file {path}: {exc}") from exc


def dismiss_conflict(store: pathlib.Path, conflict_id: str, actor: str = "human") -> dict:
    """Human-only, audited dismissal of an open conflict observation.

    Terminal for the pair while the underlying memories are unchanged: the
    record snapshots both memories' updated_at, and a later scan will not
    re-flag a dismissed pair unless either memory changed (EVIDENCE-038
    lifecycle rule - a conflict is an observation, not an authority).
    """
    record = load_conflict(store, conflict_id)
    if record.get("state") != "open":
        raise AgentMemoryError(
            f"cannot dismiss {conflict_id}: state is {record.get('state')}; only open conflicts can be dismissed"
        )
    a = load_memory(store, record["memory_a"])
    b = load_memory(store, record["memory_b"])
    timestamp = now_utc()
    record["state"] = "dismissed"
    record["dismissed_at"] = timestamp
    record["dismissed_by"] = actor
    record["dismissed_snapshot"] = {
        "memory_a_updated_at": a.get("updated_at"),
        "memory_b_updated_at": b.get("updated_at"),
    }
    _write_text_utf8(store / CONFLICT_SUBDIR / f"{conflict_id}.json",
                     json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    append_audit(store, "CONFLICT_DISMISSED", None, actor, {
        "conflict_id": conflict_id,
        "memory_a": a["id"],
        "memory_b": b["id"],
    })
    return record


def resolve_conflict(store: pathlib.Path, conflict_id: str, old_id: str, new_id: str,
                     actor: str = "human") -> dict:
    """Human-only resolution: supersede old by new via the existing
    supersede() machinery, then close the observation (audited).

    EVIDENCE-038 AC7: resolution MUST call the existing supersede() - same
    guards (no chains, only active, bidirectional links). The conflict
    record is closed with closed_reason='superseded' and audited as
    CONFLICT_RESOLVED. Never invents a new relationship mechanism.
    """
    record = load_conflict(store, conflict_id)
    if record.get("state") != "open":
        raise AgentMemoryError(
            f"cannot resolve {conflict_id}: state is {record.get('state')}; only open conflicts can be resolved"
        )
    # The conflict must reference the pair being superseded.
    pair = {record.get("memory_a"), record.get("memory_b")}
    if pair != {old_id, new_id}:
        raise AgentMemoryError(
            f"conflict {conflict_id} references {record.get('memory_a')}/{record.get('memory_b')}, "
            f"not {old_id}/{new_id}; resolve the correct pair"
        )
    supersede(store, old_id, new_id, actor=actor)  # existing machinery + guards
    timestamp = now_utc()
    record["state"] = "closed"
    record["closed_at"] = timestamp
    record["closed_by"] = actor
    record["closed_reason"] = "superseded"
    _write_text_utf8(store / CONFLICT_SUBDIR / f"{conflict_id}.json",
                     json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    append_audit(store, "CONFLICT_RESOLVED", old_id, actor, {
        "conflict_id": conflict_id,
        "new_id": new_id,
        "reason": "superseded",
    })
    return record


# --------------------------------------------------------------------------
# T5 git awareness (v0.3 Tier 5, EVIDENCE-041 - write-time contextual
# evidence + deterministic retrieval enrichment)
# --------------------------------------------------------------------------
# A git snapshot is captured ONCE at write time (create/suggest) into
# .agent/gitcontext/<mem_id>.json - a versioned sidecar (git-context-v1),
# separate from the memory schema. Retrieval reads ONLY the stored
# snapshot; live git is NEVER consulted at recall (same store + same query
# + same path -> same recall). Fail-soft: any git failure yields None and
# never becomes a memory failure.


def _git_out(cwd: pathlib.Path, *args: str) -> str | None:
    """Run a git command; return trimmed stdout, or None on ANY failure.

    Fail-soft by contract (EVIDENCE-041): a git failure must never become
    a memory failure. LC_ALL=C + --no-pager for deterministic output.
    """
    try:
        env = dict(os.environ)
        env["LC_ALL"] = "C"
        r = subprocess.run(
            ["git", "--no-pager", *args],
            cwd=str(cwd), capture_output=True, text=True, encoding="utf-8",
            errors="replace", env=env, timeout=10,
            # EVIDENCE-045 (dogfood): when the parent is an MCP stdio server,
            # our own stdin is an open protocol pipe that never reaches EOF;
            # git inherits it and blocks reading it -> 10s timeout -> the
            # whole git snapshot silently fails (None). Redirect stdin to
            # DEVNULL so git never waits on our inherited stdio.
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    return r.stdout.strip()


def capture_git_context(store: pathlib.Path) -> dict | None:
    """Write-time git snapshot (T5, EVIDENCE-041): versioned sidecar data.

    Returns a git-context-v1 snapshot dict, or None on ANY failure (not a
    git repo, git missing, command error) - memory operations continue
    unchanged. Captures repository identity, current branch, HEAD commit
    (sha, author, timestamp) and relevant changed paths (files in the HEAD
    commit, capped deterministically). Captured ONCE at write time; recall
    never consults live git.
    """
    project = store.parent  # the .agent store sits inside the project root
    if not (project / ".git").exists():
        return None  # fast fail-soft: not a git repo - no subprocess spawned
    top = _git_out(project, "rev-parse", "--show-toplevel")
    if top is None:
        return None  # not a git repo (or git unavailable) - fail-soft
    branch = _git_out(project, "branch", "--show-current")
    head_sha = _git_out(project, "rev-parse", "HEAD")
    if head_sha is None:
        return None  # no commits yet - nothing meaningful to snapshot
    head_short = _git_out(project, "rev-parse", "--short", "HEAD")
    author = _git_out(project, "log", "-1", "--format=%an <%ae>")
    timestamp = _git_out(project, "log", "-1", "--format=%aI")
    # --root: the initial commit has no parent; without it diff-tree lists
    # nothing for the very first commit, losing its changed paths.
    changed_raw = _git_out(project, "diff-tree", "--no-commit-id", "--root",
                           "--name-only", "-r", "HEAD")
    changed_paths = (sorted(changed_raw.splitlines())[:GIT_MAX_CHANGED_PATHS]
                     if changed_raw else [])
    return {
        "schema": GITCONTEXT_SCHEMA,
        "captured_at": now_utc(),
        "repo": {"top_level": top, "name": pathlib.Path(top).name},
        "branch": branch or None,  # None on detached HEAD
        "head": {
            "sha": head_sha,
            "short": head_short or head_sha[:7],
            "author": author or None,
            "timestamp": timestamp or None,
        },
        "changed_paths": changed_paths,
    }


def _git_context_path(store: pathlib.Path, mem_id: str) -> pathlib.Path:
    return store / GITCONTEXT_SUBDIR / f"{mem_id}.json"


def save_git_context(store: pathlib.Path, mem_id: str, snapshot: dict) -> None:
    """Persist the stored git snapshot sidecar (never raises)."""
    try:
        path = _git_context_path(store, mem_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_text_utf8(path, json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n")
    except OSError:
        pass  # fail-soft: a git-context write must never break the memory


def load_git_context(store: pathlib.Path, mem_id: str) -> dict | None:
    """Stored git snapshot for one memory, or None (absent/corrupt/wrong schema)."""
    path = _git_context_path(store, mem_id)
    if not path.exists():
        return None
    try:
        snap = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return snap if snap.get("schema") == GITCONTEXT_SCHEMA else None


def load_git_contexts(store: pathlib.Path) -> dict[str, dict]:
    """All stored git snapshots keyed by mem_id (deterministic id order)."""
    folder = store / GITCONTEXT_SUBDIR
    out: dict[str, dict] = {}
    if not folder.is_dir():
        return out
    for f in sorted(folder.glob("mem_*.json")):
        try:
            snap = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if snap.get("schema") != GITCONTEXT_SCHEMA:
            continue
        out[f.stem] = snap
    return out


def git_bonuses(store: pathlib.Path, path: str | None, branch: str | None) -> dict[str, float]:
    """Deterministic per-memory git bonus from STORED snapshots (T5).

    (1) a memory whose STORED changed-paths overlap the recall --path
    argument gets a bounded boost reusing the path_tier tiers (max 2.0 -
    secondary to the memory's own declared path tier); (2) a memory whose
    STORED branch equals the recall --branch argument gets GIT_BRANCH_BONUS
    (1.0). Author/timestamp/commit are NEVER scored (git authorship is
    evidence, not trust). Returns {} when there is no git context - the
    absence of git context simply means no git-derived bonus.
    """
    if not path and not branch:
        return {}
    bonuses: dict[str, float] = {}
    for mem_id, snap in load_git_contexts(store).items():
        bonus = 0.0
        if path:
            best = 0
            for cp in snap.get("changed_paths", []):
                # Changed paths are concrete file paths; overlap can be
                # either direction (changed file vs a recalled directory).
                tier = max(path_tier(cp, path), path_tier(path, cp))
                best = max(best, tier)
            if best:
                bonus += GIT_PATH_TIER_BONUS[best]
        if branch and snap.get("branch") == branch:
            bonus += GIT_BRANCH_BONUS
        if bonus:
            bonuses[mem_id] = bonus
    return bonuses


def list_memories(store: pathlib.Path, mem_type: str | None = None, status: str | None = None) -> list[dict]:
    """List memory records, newest first (id as final tie-break for determinism)."""
    records: list[dict] = []
    for t in MEMORY_TYPES:
        if mem_type and t != mem_type:
            continue
        folder = store / MEMORY_SUBDIR / t
        if not folder.is_dir():
            continue
        for f in folder.glob("*.json"):
            try:
                records.append(json.loads(f.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError):
                raise AgentMemoryError(f"corrupt memory file {f}")
    if status:
        records = [r for r in records if r.get("status") == status]
    records.sort(key=lambda r: (r.get("created_at", ""), r.get("id", "")), reverse=True)
    return records


PHRASE_BONUS = 2.0  # per contiguous adjacent query pair found in a field
SCORE_FLOOR_RATIO = 0.25  # recall-only: drop text scores below ratio x top text score


def _term_doc_freqs(records: list[dict], terms: list[str]) -> dict[str, int]:
    """df(t): number of candidate records whose title/tags/content contain t."""
    df = {t: 0 for t in terms}
    for r in records:
        blob = " ".join((
            r.get("title", ""),
            " ".join(r.get("tags", [])),
            r.get("content", ""),
        )).lower()
        for t in terms:
            if t in blob:
                df[t] += 1
    return df


def _idf_weights(n: int, df: dict[str, int]) -> dict[str, float]:
    """Smooth IDF: idf(t) = ln(n / (df(t) + 1)) + 1; common tokens approach 1.

    Deterministic pure function of the candidate set (v0.2 Tier 1,
    EVIDENCE-012/013): a token present in many memories is downweighted, a
    distinctive token carries more signal.
    """
    if n < 1:
        return {t: 0.0 for t in df}
    return {t: math.log(n / (df[t] + 1)) + 1.0 for t in df}


def _field_score(text: str, terms: list[str], idf: dict[str, float]) -> float:
    """Field score: IDF-weighted term hits plus phrase (contiguous-pair) bonus."""
    lower = text.lower()
    score = sum(lower.count(t) * idf[t] for t in terms)
    for i in range(len(terms) - 1):
        if f"{terms[i]} {terms[i + 1]}" in lower:
            score += PHRASE_BONUS
    return score


def _rank_records(records: list[dict], terms: list[str], idf: dict[str, float],
                  path: str | None = None,
                  git_bonus: dict[str, float] | None = None,
                  floor_ratio: float | None = None) -> list[dict]:
    """Deterministic relevance ranking shared by search and recall.

    Score = 3 x title + 2 x tags + 1 x content, each field IDF-weighted with
    a phrase bonus; + a TIERED path bonus when --path matches (T2.1,
    EVIDENCE-031): exact file (10) > directory (6) > glob (3) > none (0);
    + a T5 git bonus from STORED snapshots (EVIDENCE-041) - a bounded,
    deterministic boost for memories whose captured git changed-paths
    overlap the recall --path argument and/or whose captured branch equals
    the recall --branch argument. The tier/branch signals are ranking
    signals, not auto-wins - text relevance still matters. Scores round to
    6 decimals for cross-platform-stable ordering; exact ties break on
    (created_at, title, id) desc. Honest zero results: score <= 0 (no term
    hits, no path match, no git bonus) is excluded - recall never invents
    context (EVIDENCE-003).

    floor_ratio (Tier 2.1, EVIDENCE-010/015): when set, a memory's TEXT score
    must be >= floor_ratio x the best text score in the candidate set, UNLESS
    any path tier matches (path = explicit operator/agent intent, always
    kept). Relative, not absolute: self-calibrates with corpus size (idf
    scales with ln N) and can never zero out a sparse-but-unique match (the
    top text score always passes). Applied by recall only; search stays
    inclusive so operators keep full visibility (EVIDENCE-007).
    """
    git_bonus = git_bonus or {}
    entries = []
    for r in records:
        tags_text = " ".join(r.get("tags", []))
        text = round(
            3.0 * _field_score(r.get("title", ""), terms, idf)
            + 2.0 * _field_score(tags_text, terms, idf)
            + 1.0 * _field_score(r.get("content", ""), terms, idf), 6)
        path_bonus = 0.0
        if path:
            for pat in r.get("paths", []):
                tier = path_tier(pat, path)
                if tier:
                    path_bonus = max(path_bonus, PATH_TIER_BONUS[tier])
        entries.append((text, path_bonus, r))
    best = max((e[0] for e in entries), default=0.0)
    scored = []
    for text, path_bonus, r in entries:
        # The honest-zero gate and the precision floor use ONLY the memory's
        # own declared path tier (explicit operator/agent intent). The T5 git
        # bonus is a pure RANKING signal within the already-qualified set - it
        # never expands recall (ambient repo state must not silently add
        # results) and never rescues weak text from the floor (EVIDENCE-041).
        if text <= 0 and path_bonus <= 0:
            continue  # honest zero: no term hits and no declared path match.
        if (floor_ratio is not None and path_bonus <= 0 and text > 0
                and text < floor_ratio * best):
            continue  # weak tail below the relevance floor.
        score = text + path_bonus + git_bonus.get(r.get("id", ""), 0.0)
        scored.append((round(score, 6), r))
    scored.sort(key=lambda pair: (pair[0], pair[1].get("created_at", ""),
                                  pair[1].get("title", ""),
                                  pair[1].get("id", "")), reverse=True)
    return [r for _, r in scored]


def _check_limit(limit: int) -> None:
    """V0.1_SPEC.md 7: N < 1 is a usage error (exit 2)."""
    if limit < 1:
        raise UsageError(f"--limit must be >= 1 (got {limit})")


def search_memories(
    store: pathlib.Path,
    query: str,
    mem_type: str | None = None,
    status: str | None = None,
    limit: int = DEFAULT_SEARCH_LIMIT,
) -> list[dict]:
    """Operator search: case-insensitive textual match on title/content/tags.

    Empty query or no matches -> [] (caller prints '0 results', exit 0).
    """
    _check_limit(limit)
    terms = [t for t in query.lower().split() if t]
    if not terms:
        return []  # V0.1_SPEC.md 7: empty query is a successful 0 results.
    records = list_memories(store, mem_type=mem_type, status=None)
    if status:
        records = [r for r in records if r.get("status") == status]
    else:
        records = [r for r in records if r.get("status") in ("active", "superseded")]
    idf = _idf_weights(len(records), _term_doc_freqs(records, terms))
    # Search is operator-facing: inclusive on purpose (EVIDENCE-007 - the
    # operator sees superseded/weak matches; the agent does not). No floor.
    return _rank_records(records, terms, idf)[:limit]


def recall_memories(
    store: pathlib.Path,
    query: str,
    path: str | None = None,
    branch: str | None = None,
    limit: int | None = None,
    include_untrusted: bool = False,
) -> list[dict]:
    """Agent recall: active + trusted memories, deterministic scoring, tiered
    path bonus, and T5 git bonus from STORED snapshots only (EVIDENCE-041).

    branch: the CURRENT branch name, supplied explicitly by the caller. It
    is compared against each memory's STORED git snapshot branch (captured
    at write time) - live git is NEVER consulted at recall, so same store +
    same query + same path + same branch -> same recall.
    """
    config = load_config(store)
    if limit is None:
        limit = config.get("recall_limit", DEFAULT_RECALL_LIMIT)
    _check_limit(limit)
    include_untrusted = include_untrusted or config.get("include_untrusted", False)
    terms = [t for t in query.lower().split() if t]
    records = list_memories(store, status="active")
    if not include_untrusted:
        records = [r for r in records if r.get("trust") != "untrusted"]
    idf = _idf_weights(len(records), _term_doc_freqs(records, terms))
    git_bonus = git_bonuses(store, path, branch)
    # Tier 2.1 floor (EVIDENCE-010/015): recall is agent-facing, precision
    # matters - drop the weak tail, keep path matches, keep honest zeros.
    return _rank_records(records, terms, idf, path=path, git_bonus=git_bonus,
                          floor_ratio=SCORE_FLOOR_RATIO)[:limit]


def status_summary(store: pathlib.Path) -> dict:
    """Store health + counts for the status command."""
    records = list_memories(store)
    by_type = {t: 0 for t in MEMORY_TYPES}
    by_status = {s: 0 for s in STATUSES}
    by_trust = {t: 0 for t in TRUST_LEVELS}
    for r in records:
        by_type[r.get("type", "?")] = by_type.get(r.get("type", "?"), 0) + 1
        by_status[r.get("status", "?")] = by_status.get(r.get("status", "?"), 0) + 1
        by_trust[r.get("trust", "?")] = by_trust.get(r.get("trust", "?"), 0) + 1
    config = load_config(store)
    suggestions = list_suggestions(store)
    conflicts = list_conflicts(store)
    return {
        "project": config.get("project", store.parent.name),
        "store": str(store),
        "total": len(records),
        "by_type": by_type,
        "by_status": by_status,
        "by_trust": by_trust,
        "audit_events": len(read_audit(store)),
        "suggestions_pending": sum(1 for s in suggestions if s.get("state") == "pending"),
        "conflicts_open": sum(1 for c in conflicts if c.get("state") == "open"),
        "git_contexts": len(load_git_contexts(store)),
    }


# --------------------------------------------------------------------------
# Output formatting
# --------------------------------------------------------------------------

def memory_summary(record: dict) -> str:
    return (
        f"{record['id']}  [{record['type']}] {record['title']} "
        f"(trust={record['trust']}, status={record['status']})"
    )


# --------------------------------------------------------------------------
# Family import (V0.1_SPEC.md section 14 - v0.2 Tier 2.3, EVIDENCE-003/016)
# --------------------------------------------------------------------------

_ERROR_ENTRY_RE = re.compile(r"^\[(?P<tag>[^\]]+)\] AREA: (?P<area>.+)$")
_DECISION_ENTRY_RE = re.compile(r"^\[(?P<tag>[^\]]+)\] DECISION: (?P<title>.+)$")
_IMPORT_FIELD_RE = re.compile(r"^  (?P<field>[A-Z]+):\s*(?P<value>.*)$")
_SECTION_SEP_RE = re.compile(r"^={4,}$")
_LESSON_CLUSTER_RE = re.compile(
    r"^\*{0,2}\s*CLUSTER\s+(?P<num>\d+)\s*[\u2014\u2013-]\s*"
    r"(?P<part>ROOT CAUSE|RULE):\*{0,2}\s*(?P<value>.*)$",
    re.IGNORECASE,
)
_LESSON_SEP_RE = re.compile(r"^\s*-{3,}$")

# rule-log source (v0.2 Tier 2.3, EVIDENCE-023): the numbered engagement
# sections 1-6 of rules.txt (section 7 drafts belong to lesson-log).
_RULE_SECTION_RE = re.compile(r"^(\d+)\)\s+(.+)$")
_LESSONS_MARKER_RE = re.compile(r"^#+\s*\d+\)\s*LESSONS", re.IGNORECASE)


def _canonical_bytes(block: str) -> bytes:
    """Canonical bytes of a source entry for fingerprinting (spec 14.2).

    UTF-8, LF line endings (CRLF normalized), trailing whitespace stripped
    per line, exactly one trailing newline. Same source entry -> same
    fingerprint, independent of the host OS or editor (ROUND 2 #2).
    """
    lines = [ln.rstrip() for ln in block.replace("\r\n", "\n").split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return ("\n".join(lines) + "\n").encode("utf-8")


def fingerprint_entry(block: str) -> str:
    """sha256:<64 hex> over the canonical bytes of a source entry block."""
    return "sha256:" + hashlib.sha256(_canonical_bytes(block)).hexdigest()


def _parse_source_log(source: str, text: str) -> list[dict]:
    """Parse a sibling log into entry dicts, in file order.

    Mirrors the siblings' OWN canonical parsers: an entry is a column-0
    '[tag] AREA:' / '[tag] DECISION:' header plus its indented fields,
    running until the next entry header or a '====' section separator. The
    template section is indented so it never matches (same contract as
    check_errors.parse_entries / check_decisions.parse_entries). No section
    sniffing: everything the sibling considers an entry is an entry.
    """
    entry_re = _ERROR_ENTRY_RE if source == "error-log" else _DECISION_ENTRY_RE
    if source == "lesson-log":
        return _parse_lesson_drafts(text.splitlines())
    if source == "rule-log":
        return _parse_rules_engagement(text.splitlines())
    lines = text.splitlines()
    entries: list[dict] = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        m = entry_re.match(line)
        if not m:
            i += 1
            continue
        j = i + 1
        body: list[str] = []
        while j < n and not (entry_re.match(lines[j]) or _SECTION_SEP_RE.match(lines[j])):
            body.append(lines[j])
            j += 1
        fields: dict[str, str] = {}
        for bl in body:
            fm = _IMPORT_FIELD_RE.match(bl)
            if fm:
                fields.setdefault(fm.group("field"), fm.group("value").strip())
        block = "\n".join([line] + body)
        entry = {
            "tag": m.group("tag"),
            "block": block,
            "fields": fields,
        }
        entry["title"] = m.group("area") if source == "error-log" else m.group("title")
        entries.append(entry)
        i = j
    return entries


def _parse_lesson_drafts(lines: list[str]) -> list[dict]:
    """Parse rules.txt AI-derived lesson drafts into entry dicts.

    A draft cluster is 'CLUSTER n - ROOT CAUSE: ...' plus 'CLUSTER n -'
    'RULE: ...' (bold or plain, order agnostic, one optional '---'
    separator between clusters). Clusters without a RULE are skipped
    (a lesson with no actionable rule is not importable). Deterministic.
    """
    entries: list[dict] = []
    cur: dict | None = None
    pending_rules: dict[str, dict] = {}  # num -> {value, lines}
    for line in lines:
        m = _LESSON_CLUSTER_RE.match(line)
        if m:
            num, part = m.group("num"), m.group("part")
            value = m.group("value").lstrip(" *").strip()
            if part == "ROOT CAUSE":
                if cur is not None and cur.get("rule"):
                    entries.append(cur)
                pr = pending_rules.pop(num, None)
                cur = {"tag": f"CLUSTER {num}", "cause": value,
                       "rule": pr["value"] if pr else "",
                       "block": list(pr["lines"]) if pr else []}
            elif part == "RULE":
                if cur is not None and cur.get("tag") == f"CLUSTER {num}" \
                        and not cur.get("rule"):
                    cur["rule"] = value
                    cur["block"].append(line)
                else:
                    # RULE seen before its cluster header (order-agnostic,
                    # spec 14.5): buffer it until the ROOT CAUSE arrives.
                    pending_rules.setdefault(num, {"value": "", "lines": []})
                    pending_rules[num]["value"] = value
                    pending_rules[num]["lines"].append(line)
                continue
            cur["block"].append(line)
            continue
        if _LESSON_SEP_RE.match(line):
            continue  # separator between clusters; not part of any block
        if cur is not None:
            cur["block"].append(line)
    if cur is not None and cur.get("rule"):
        entries.append(cur)
    out: list[dict] = []
    for e in entries:
        block = "\n".join(b for b in e["block"] if b.strip())
        out.append({
            "tag": e["tag"],
            "block": block,
            "fields": {"ROOT CAUSE": e["cause"], "RULE": e["rule"]},
            "title": e["rule"][:120] or e["cause"][:120],
        })
    return out


def _parse_rules_engagement(lines: list[str]) -> list[dict]:
    """Parse rules.txt numbered RULES OF ENGAGEMENT (sections 1-6) into
    entry dicts, in file order.

    A numbered section is 'N) TITLE' at column 0 plus its indented body,
    running until the next 'N)' header. Parsing stops at the '## 7)'
    LESSONS LEARNED marker: section 7 draft clusters belong to the
    lesson-log source, never both. Leading prose and blank lines are
    ignored. Deterministic.
    """
    entries: list[dict] = []
    cur: dict | None = None
    for line in lines:
        if _LESSONS_MARKER_RE.match(line) or _LESSON_CLUSTER_RE.match(line):
            break  # section 7 drafts can never become constraint rules
        m = _RULE_SECTION_RE.match(line)
        if m:
            if m.group(2).lstrip().upper().startswith("LESSONS"):
                break  # bare "N) LESSONS..." header without '#'
            if cur is not None:
                entries.append(cur)
            cur = {"tag": f"RULE {m.group(1)}", "title": m.group(2).strip(),
                   "body": [], "block": [line]}
            continue
        if cur is not None and line.strip():
            cur["body"].append(line.strip())
            cur["block"].append(line)
    if cur is not None:
        entries.append(cur)
    out: list[dict] = []
    for e in entries:
        block = "\n".join(b for b in e["block"] if b.strip())
        out.append({
            "tag": e["tag"],
            "block": block,
            "fields": {"RULE": e["title"], "BODY": " ".join(e["body"])},
            "title": e["title"],
        })
    return out


def _entry_to_memory(source: str, entry: dict) -> dict:
    """Map a parsed source entry to create_memory kwargs (boring mapping)."""
    fields = entry["fields"]
    if source == "error-log":
        parts = [f"{k}: {fields[k]}" for k in ("ERROR", "CAUSE", "FIX") if fields.get(k)]
        mem_type = "error"
        tags = [fields.get("STATUS", "open").split(".")[0].strip().lower()]
        paths: list[str] = []
    elif source == "lesson-log":
        parts = [f"ROOT CAUSE: {fields.get('ROOT CAUSE', '')}",
                 f"RULE: {fields.get('RULE', '')}"]
        mem_type = "lesson"
        tags = ["ai-draft", "unconfirmed"]
        paths: list[str] = []
    elif source == "rule-log":
        parts = [fields.get("BODY", "")]
        mem_type = "constraint"
        tags = ["rule", "numbered"]
        paths: list[str] = []
    else:
        parts = [f"REASON: {fields['REASON']}" if fields.get("REASON") else "",
                 f"FILES: {fields['FILES']}" if fields.get("FILES") else ""]
        mem_type = "decision"
        tags = [fields.get("STATUS", "open").split(".")[0].strip().lower()]
        paths = [p.strip() for p in fields.get("FILES", "").split(",") if p.strip()]
    content = "\n".join(p for p in parts if p)
    if not content:
        content = "(no fields)"  # title-only entry; still importable
    return {"type": mem_type, "title": entry["title"].strip(),
            "content": content, "tags": tags, "paths": paths}


def import_source_log(
    store: pathlib.Path,
    source: str,
    text: str,
    project: str,
    actor: str = "human",
    dry_run: bool = False,
) -> dict:
    """Import a sibling log into the store. Returns a report dict.

    Invariants (spec 14.3, EVIDENCE-016):
      - Same source entry -> same fingerprint; re-import -> no duplicates.
      - Imported memories are born untrusted, provenance='import'; import
        never promotes (cannot manufacture verified/approved/system).
      - Secret detection runs BEFORE persistence (create_memory rejects;
        rejections are counted + audited, not silently dropped).
      - Source provenance survives; SUPERSEDES links are wired where the
        target is deterministically knowable (same run, target active).
      - Existing memories are never overwritten.
    """
    if source not in IMPORT_SOURCES:
        raise UsageError(f"--source must be one of {', '.join(IMPORT_SOURCES)} (got {source!r})")
    entries = _parse_source_log(source, text)
    report = {
        "source": source,
        "repository": IMPORT_REPOS[source],
        "entries": len(entries),
        "new": 0,
        "duplicates": 0,
        "rejected": 0,
        "rejected_details": [],
        "superseded": 0,
        "already_wired": 0,
        "unresolved_supersedes": 0,
        "dry_run": dry_run,
        "results": [],
    }

    if dry_run:
        for e in entries:
            report["results"].append({
                "title": e["title"],
                "fingerprint": fingerprint_entry(e["block"]),
                "action": "create",
            })
        report["new"] = len(entries)
        return report

    # Existing fingerprints -> dedupe (same source entry, same fingerprint).
    existing: dict[str, str] = {}  # fingerprint -> id (all memories)
    tag_to_id: dict[str, str] = {}  # source tag -> id (all memories)
    for r in list_memories(store):
        src = r.get("source")
        if isinstance(src, dict):
            if src.get("fingerprint"):
                existing.setdefault(src["fingerprint"], r["id"])
            if src.get("tag"):
                tag_to_id.setdefault(src["tag"], r["id"])

    for e in entries:
        fp = fingerprint_entry(e["block"])
        if fp in existing:
            report["duplicates"] += 1
            tag_to_id[e["tag"]] = existing[fp]
            continue
        mapping = _entry_to_memory(source, e)
        try:
            record = create_memory(
                store=store,
                mem_type=mapping["type"],
                title=mapping["title"],
                content=mapping["content"],
                project=project,
                provenance="import",
                tags=mapping["tags"],
                paths=mapping["paths"],
                source={
                    "type": mapping["type"],
                    "repository": IMPORT_REPOS[source],
                    "fingerprint": fp,
                    "tag": e["tag"],
                },
                actor=actor,
            )
        except AgentMemoryError as exc:  # secret detection / validation
            report["rejected"] += 1
            report["rejected_details"].append({"title": e["title"], "reason": str(exc)})
            continue
        report["new"] += 1
        tag_to_id[e["tag"]] = record["id"]
        report["results"].append({"id": record["id"], "title": e["title"],
                                  "fingerprint": fp})

    # Supersession wiring: entry with SUPERSEDES <tag> -> supersede old->new,
    # oldest first (file order), only when both are active and knowable.
    if source == "decision-log":
        for e in entries:
            target = e["fields"].get("SUPERSEDES", "").strip()
            if not target:
                continue
            new_id = tag_to_id.get(e["tag"])
            old_id = tag_to_id.get(target)
            if not new_id or not old_id:
                report["unresolved_supersedes"] += 1
                continue
            old = load_memory(store, old_id)
            new = load_memory(store, new_id)
            if old["status"] != "active" or new.get("supersedes"):
                report["already_wired"] += 1
                continue
            try:
                supersede(store, old_id, new_id, actor=actor)
                report["superseded"] += 1
            except AgentMemoryError:
                report["unresolved_supersedes"] += 1

    append_audit(store, "IMPORT_RUN", None, actor, {
        "source": source,
        "repository": IMPORT_REPOS[source],
        "entries": report["entries"],
        "new": report["new"],
        "duplicates": report["duplicates"],
        "rejected": report["rejected"],
        "superseded": report["superseded"],
        "already_wired": report["already_wired"],
        "unresolved_supersedes": report["unresolved_supersedes"],
    })
    return report


# --------------------------------------------------------------------------
# CLI (V0.1_SPEC.md section 6)
# --------------------------------------------------------------------------

class JsonFriendlyParser(argparse.ArgumentParser):
    """In --json mode, usage errors emit {"error": ...} to stdout (spec 6.4)."""

    def error(self, message: str) -> None:
        if "--json" in sys.argv:
            _emit_json({"error": message})
            raise SystemExit(2)
        super().error(message)


def build_parser() -> argparse.ArgumentParser:
    parser = JsonFriendlyParser(
        prog="agent-memory",
        description="local-first, persistent knowledge and governance layer for AI coding agents",
    )
    parser.add_argument("--version", action="version", version=f"agent-memory {VERSION}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="create the .agent/ store in the current project")
    p_init.add_argument("--project", help="project name (default: basename of cwd)")
    p_init.add_argument("--force", action="store_true", help="re-init an empty existing store")
    p_init.add_argument("--dir", help="target directory (default: cwd)")

    p_add = sub.add_parser("add", help="create a memory (interactive when no flags are given)")
    p_add.add_argument("--type", choices=MEMORY_TYPES, help="memory type")
    p_add.add_argument("--title", help="memory title")
    p_add.add_argument("--content", help="memory content")
    p_add.add_argument("--tags", help="comma-separated tags")
    p_add.add_argument("--paths", help="comma-separated path globs")
    p_add.add_argument("--severity", choices=SEVERITIES, default="normal", help="severity (default normal)")
    p_add.add_argument("--provenance", choices=[p for p in PROVENANCES if p != "system"], default="human")
    p_add.add_argument("--allow-secret", action="store_true", help="explicitly allow content that trips the secret detector (audited)")
    p_add.add_argument("--json", action="store_true")

    p_list = sub.add_parser("list", help="list memories")
    p_list.add_argument("--type", choices=MEMORY_TYPES)
    p_list.add_argument("--status", choices=STATUSES)
    p_list.add_argument("--json", action="store_true")

    p_show = sub.add_parser("show", help="show one memory")
    p_show.add_argument("mem_id")
    p_show.add_argument("--json", action="store_true")

    p_search = sub.add_parser("search", help="operator textual search")
    p_search.add_argument("query", nargs="?", default="")
    p_search.add_argument("--type", choices=MEMORY_TYPES)
    p_search.add_argument("--status", choices=STATUSES)
    p_search.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT)
    p_search.add_argument("--json", action="store_true")

    p_recall = sub.add_parser("recall", help="agent context assembly")
    p_recall.add_argument("query", nargs="?", default="")
    p_recall.add_argument("--path", help="file path for the tiered path bonus")
    p_recall.add_argument("--branch", help="current branch for the T5 stored-snapshot bonus")
    p_recall.add_argument("--limit", type=int, default=None)
    p_recall.add_argument("--include-untrusted", action="store_true")
    p_recall.add_argument("--json", action="store_true")

    p_promote = sub.add_parser("promote", help="human trust promotion")
    p_promote.add_argument("mem_id")
    p_promote.add_argument("--trust", choices=("verified", "approved"), required=True)
    p_promote.add_argument("--json", action="store_true")

    p_import = sub.add_parser("import", help="import a family log (error-log / decision-log / lesson-log / rule-log)")
    p_import.add_argument("path", help="sibling log file (errors.txt / decisions.txt) or its directory")
    p_import.add_argument("--source", choices=IMPORT_SOURCES, required=True,
                          help="which family log format to parse")
    p_import.add_argument("--dry-run", action="store_true",
                          help="parse + fingerprint only; persist nothing")
    p_import.add_argument("--json", action="store_true")

    p_super = sub.add_parser("supersede", help="mark old memory superseded by new")
    p_super.add_argument("old_id")
    p_super.add_argument("new_id")
    p_super.add_argument("--json", action="store_true")

    p_del = sub.add_parser("delete", help="lifecycle delete (tombstone) or purge")
    p_del.add_argument("mem_id")
    p_del.add_argument("--purge", action="store_true", help="physically remove (untrusted only)")
    p_del.add_argument("--json", action="store_true")

    p_status = sub.add_parser("status", help="store health + counts")
    p_status.add_argument("--json", action="store_true")

    p_sug = sub.add_parser("suggestions",
                           help="human review of agent suggestions (T3 approve-to-persist loop)")
    p_sug.add_argument("action", choices=("list", "approve", "reject"))
    p_sug.add_argument("sug_id", nargs="?", help="suggestion id (approve/reject)")
    p_sug.add_argument("--trust", choices=("verified", "approved"),
                       help="resulting trust level (approve only - never 'system')")
    p_sug.add_argument("--state", choices=SUGGESTION_STATES, help="filter (list only)")
    p_sug.add_argument("--json", action="store_true")

    p_conf = sub.add_parser("conflicts",
                            help="human review of possible-conflict observations (T4)")
    p_conf.add_argument("action", choices=("scan", "list", "dismiss", "resolve"))
    p_conf.add_argument("conflict_id", nargs="?", help="conflict record id (dismiss/resolve)")
    p_conf.add_argument("--old", help="memory id to mark superseded (resolve only)")
    p_conf.add_argument("--new", help="memory id that supersedes (resolve only)")
    p_conf.add_argument("--state", choices=CONFLICT_STATES, help="filter (list only)")
    p_conf.add_argument("--json", action="store_true")

    p_git = sub.add_parser("git",
                           help="review stored T5 git context (write-time snapshots)")
    p_git.add_argument("action", choices=("context", "list"))
    p_git.add_argument("mem_id", nargs="?", help="memory id (context only)")
    p_git.add_argument("--json", action="store_true")

    return parser


def _emit_json(payload: dict) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _prompt(prompt: str) -> str:
    """Interactive prompt; EOFError (no tty) becomes a usage error."""
    try:
        return input(prompt).strip()
    except EOFError:
        raise UsageError("interactive add requires a terminal; pass --type/--title/--content flags instead")


def _run_add(args: argparse.Namespace, store: pathlib.Path, config: dict) -> int:
    """Spec 6.3: flags -> deterministic; no flags -> interactive guided mode."""
    if args.type is None or args.title is None or args.content is None:
        if args.type is None and args.title is None and args.content is None:
            # Interactive guided mode (human friction fix - ROUND 5 risk #1).
            mem_type = _prompt("type (decision/error/lesson/constraint/architecture/pattern): ")
            if mem_type not in MEMORY_TYPES:
                raise UsageError(f"invalid type {mem_type!r}")
            title = _prompt("title: ")
            content = _prompt("content: ")
            tags_input = _prompt("tags (comma-separated, optional): ")
            paths_input = _prompt("paths (comma-separated globs, optional): ")
            severity = args.severity or "normal"
            record = create_memory(
                store=store,
                mem_type=mem_type,
                title=title,
                content=content,
                project=config["project"],
                provenance=args.provenance,
                severity=severity,
                tags=[t.strip() for t in tags_input.split(",") if t.strip()] if tags_input else None,
                paths=[p.strip() for p in paths_input.split(",") if p.strip()] if paths_input else None,
                allow_secret=args.allow_secret,
            )
            if args.json:
                _emit_json(record)
            else:
                print(f"created {record['id']}")
            return 0
        raise UsageError("add requires --type, --title and --content (or no flags for interactive mode)")
    record = create_memory(
        store=store,
        mem_type=args.type,
        title=args.title,
        content=args.content,
        project=config["project"],
        provenance=args.provenance,
        severity=args.severity,
        tags=[t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else None,
        paths=[p.strip() for p in args.paths.split(",") if p.strip()] if args.paths else None,
        allow_secret=args.allow_secret,
    )
    if args.json:
        _emit_json(record)
    else:
        print(f"created {record['id']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    reconfigure_utf8()
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "init":
            store = init_store(
                target=pathlib.Path(args.dir) if args.dir else None,
                project=args.project,
                force=args.force,
            )
            print(f"initialized agent-memory store at {store}")
            return 0

        store = find_store()
        config = load_config(store)

        if args.command == "add":
            return _run_add(args, store, config)
        if args.command == "list":
            records = list_memories(store, mem_type=args.type, status=args.status)
            if args.json:
                _emit_json({"results": records, "count": len(records)})
            else:
                for r in records:
                    print(memory_summary(r))
                print(f"{len(records)} result(s)" if records else "0 results")
            return 0
        if args.command == "show":
            record = load_memory(store, args.mem_id)
            append_audit(store, "MEMORY_ACCESSED", record["id"], "human", {})
            if args.json:
                _emit_json(record)
            else:
                print(json.dumps(record, indent=2, ensure_ascii=False))
            return 0
        if args.command == "search":
            results = search_memories(store, args.query, mem_type=args.type, status=args.status, limit=args.limit)
            append_audit(store, "MEMORY_ACCESSED", None, "human", {"command": "search", "count": len(results)})
            if args.json:
                _emit_json({"results": results, "count": len(results)})
            else:
                for r in results:
                    print(memory_summary(r))
                print(f"{len(results)} result(s)" if results else "0 results")
            return 0
        if args.command == "recall":
            results = recall_memories(
                store, args.query, path=args.path, branch=args.branch,
                limit=args.limit, include_untrusted=args.include_untrusted,
            )
            append_audit(store, "MEMORY_ACCESSED", None, "human", {"command": "recall", "count": len(results)})
            if args.json:
                _emit_json({"results": results, "count": len(results)})
            else:
                if results:
                    print("RELEVANT CONTEXT")
                    for r in results:
                        print(f"[{r['type'].upper()}] {r['title']} (trust: {r['trust']})")
                else:
                    print("0 results")
            return 0
        if args.command == "promote":
            record = promote_trust(store, args.mem_id, args.trust)
            if args.json:
                _emit_json(record)
            else:
                print(f"promoted {args.mem_id} to trust={args.trust}")
            return 0
        if args.command == "import":
            path = pathlib.Path(args.path)
            if path.is_dir():
                path = path / IMPORT_LOG_FILES[args.source]
            if not path.is_file():
                raise UsageError(f"cannot read {path}: not a file")
            text = path.read_text(encoding="utf-8-sig")
            report = import_source_log(
                store, args.source, text, config["project"],
                dry_run=args.dry_run,
            )
            if args.json:
                _emit_json(report)
            else:
                verb = "would import" if report["dry_run"] else "imported"
                print(f"{verb} {report['entries']} entr(y/ies) from "
                      f"{report['repository']}: {report['new']} new, "
                      f"{report['duplicates']} duplicate(s) skipped, "
                      f"{report['rejected']} rejected")
                if report["superseded"]:
                    print(f"  {report['superseded']} supersession link(s) wired")
                if report["unresolved_supersedes"]:
                    print(f"  {report['unresolved_supersedes']} SUPERSEDES "
                          f"reference(s) unresolved (target not imported)")
                for r in report["rejected_details"]:
                    print(f"  rejected: {r['title']} - {r['reason']}")
            return 0

        if args.command == "supersede":
            old = supersede(store, args.old_id, args.new_id)
            if args.json:
                _emit_json(old)
            else:
                print(f"{args.old_id} superseded by {args.new_id}")
            return 0
        if args.command == "delete":
            record = delete_memory(store, args.mem_id, purge=args.purge)
            if args.json:
                _emit_json(record)
            else:
                if args.purge:
                    print(f"purged {args.mem_id}")
                else:
                    print(f"deleted {args.mem_id} (tombstone)")
            return 0
        if args.command == "suggestions":
            if args.action == "list":
                sugs = list_suggestions(store, state=args.state)
                if args.json:
                    _emit_json({"results": sugs, "count": len(sugs)})
                else:
                    for s in sugs:
                        print(f"{s['id']}  [{s['type']}] {s['title']} (state={s['state']})")
                    print(f"{len(sugs)} result(s)" if sugs else "0 results")
                return 0
            if not args.sug_id:
                raise UsageError(f"suggestions {args.action} requires a suggestion id")
            if args.action == "approve":
                if not args.trust:
                    raise UsageError("suggestions approve requires --trust verified|approved")
                record = approve_suggestion(store, args.sug_id, args.trust)
                if args.json:
                    _emit_json(record)
                else:
                    print(f"approved {args.sug_id} -> memory {record['id']} (trust={record['trust']})")
            else:
                sug = reject_suggestion(store, args.sug_id)
                if args.json:
                    _emit_json(sug)
                else:
                    print(f"rejected {args.sug_id}")
            return 0

        if args.command == "conflicts":
            if args.action == "scan":
                report = scan_conflicts(store)
                if args.json:
                    _emit_json(report)
                else:
                    print(f"scanned {report['scanned']} active memorie(s): "
                          f"{report['candidates']} new possible conflict(s), "
                          f"{report['already_open']} already open, "
                          f"{report['dismissed_unchanged']} dismissed-unchanged skipped")
                    if report["closed_stale"]:
                        print(f"  {report['closed_stale']} stale observation(s) closed (state changed)")
            elif args.action == "list":
                conflicts = list_conflicts(store, state=args.state)
                if args.json:
                    _emit_json({"results": conflicts, "count": len(conflicts)})
                else:
                    for c in conflicts:
                        ex = c.get("explanation", {})
                        print(f"{c['id']}  [{ex.get('same_type', '?')}] "
                              f"{c['memory_a']} <-> {c['memory_b']} (state={c['state']})")
                    print(f"{len(conflicts)} result(s)" if conflicts else "0 results")
            elif args.action == "dismiss":
                if not args.conflict_id:
                    raise UsageError("conflicts dismiss requires a conflict id")
                rec = dismiss_conflict(store, args.conflict_id)
                if args.json:
                    _emit_json(rec)
                else:
                    print(f"dismissed {args.conflict_id}")
            else:  # resolve
                if not args.conflict_id or not args.old or not args.new:
                    raise UsageError(
                        "conflicts resolve requires <conflict_id> --old <id> --new <id>")
                rec = resolve_conflict(store, args.conflict_id, args.old, args.new)
                if args.json:
                    _emit_json(rec)
                else:
                    print(f"resolved {args.conflict_id}: {args.old} superseded by {args.new}")
            return 0

        if args.command == "git":
            if args.action == "list":
                snaps = load_git_contexts(store)
                if args.json:
                    _emit_json({"results": [
                        {"mem_id": mid, **snap} for mid, snap in snaps.items()],
                        "count": len(snaps)})
                else:
                    for mid, snap in snaps.items():
                        branch = snap.get("branch") or "detached"
                        head = (snap.get("head") or {}).get("short", "?")
                        print(f"{mid}  branch={branch} head={head}")
                    print(f"{len(snaps)} result(s)" if snaps else "0 results")
            else:  # context
                if not args.mem_id:
                    raise UsageError("git context requires a memory id")
                snap = load_git_context(store, args.mem_id)
                if snap is None:
                    raise AgentMemoryError(
                        f"no git context for {args.mem_id} (not captured or not a git repo)")
                if args.json:
                    _emit_json(snap)
                else:
                    print(json.dumps(snap, indent=2, ensure_ascii=False))
            return 0

        if args.command == "status":
            summary = status_summary(store)
            if args.json:
                _emit_json(summary)
            else:
                print(f"project       : {summary['project']}")
                print(f"store         : {summary['store']}")
                print(f"total memories: {summary['total']}")
                print(f"audit events  : {summary['audit_events']}")
                print(f"pending sugg  : {summary['suggestions_pending']}")
                print(f"open conflicts: {summary['conflicts_open']}")
                print(f"git contexts  : {summary['git_contexts']}")
                print("by type      : " + ", ".join(f"{k}={v}" for k, v in summary['by_type'].items()))
                print("by status    : " + ", ".join(f"{k}={v}" for k, v in summary['by_status'].items()))
                print("by trust     : " + ", ".join(f"{k}={v}" for k, v in summary['by_trust'].items()))
            return 0
    except UsageError as exc:
        if getattr(args, "json", False):
            _emit_json({"error": str(exc)})
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 2
    except AgentMemoryError as exc:
        if getattr(args, "json", False):
            _emit_json({"error": str(exc)})
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
