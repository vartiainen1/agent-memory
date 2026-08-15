"""_audit_cli.py - external-API audit of the agent-memory CLI (v0.1).

Treats the CLI as an external API: every check runs the REAL binary via
subprocess in an isolated scratch project and asserts on stdout, stderr,
exit codes and output VALUES (workspace rule 12: verify values, not just
rc). Covers every command across: normal behavior, malformed input, missing
IDs, nonexistent project, corrupt files, invalid enums, secret detection,
permission/trust boundaries, --json contract, empty results, Unicode, and
cross-platform determinism.

Contract: V0.1_SPEC.md. Exit codes: 0 success, 1 runtime, 2 usage.

Run: python _audit_cli.py
"""

from __future__ import annotations

import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent
BIN = [sys.executable, str(ROOT / "agent_memory.py")]

COUNT = 0
FAILURES: list[str] = []
PROJECTS: list["Project"] = []


def run(args: list, cwd: pathlib.Path, stdin: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        BIN + [str(a) for a in args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        input=stdin,
        timeout=60,
    )


def ok(name: str, cond: bool, detail: str = "") -> None:
    global COUNT
    COUNT += 1
    if not cond:
        FAILURES.append(f"{name}: {detail}")
        print(f"FAIL {name} :: {detail}")


class Project:
    """An isolated scratch project with its own agent-memory store."""

    def __init__(self, name: str = "demo"):
        self.dir = pathlib.Path(tempfile.mkdtemp(prefix="am-audit-"))
        self.name = name
        PROJECTS.append(self)

    def init(self, *flags):
        """init with the project name pinned so assertions on 'demo' hold."""
        return self.run("init", "--project", self.name, *flags)

    def run(self, *args, stdin: str | None = None, sub: str | None = None):
        cwd = self.dir if sub is None else self.dir / sub
        return run(list(args), cwd, stdin=stdin)

    def store(self) -> pathlib.Path:
        return self.dir / ".agent"

    def config(self) -> pathlib.Path:
        return self.dir / ".agent" / "config.toml"

    def audit_lines(self) -> list[str]:
        p = self.store() / "audit.jsonl"
        if not p.exists():
            return []
        return [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]

    def audit_events(self) -> list[str]:
        return [json.loads(ln)["event"] for ln in self.audit_lines()]

    def add(self, type_, title, content, *flags):
        return self.run("add", "--type", type_, "--title", title, "--content", content, *flags)

    def add_ids(self, type_, title, content, *flags) -> list[str]:
        """Add, returning the created mem ids."""
        out = self.add(type_, title, content, *flags).stdout
        return re.findall(r"created (mem_[0-9a-f-]+)", out)

    def all_ids(self) -> list[str]:
        """All memory ids via list --json (used to drive promote loops)."""
        return [r["id"] for r in json.loads(self.run("list", "--json").stdout)["results"]]

    def cleanup(self):
        shutil.rmtree(self.dir, ignore_errors=True)


def cleanup_all():
    for p in PROJECTS:
        p.cleanup()


# ==========================================================================
# init
# ==========================================================================

def audit_init():
    p = Project()
    r = p.init()
    ok("init exit 0", r.returncode == 0 and "initialized" in r.stdout, f"rc={r.returncode} out={r.stdout!r}")
    ok("init creates layout",
       p.config().exists() and (p.store() / "audit.jsonl").exists()
       and all((p.store() / "memory" / t).is_dir() for t in
               ("decision", "error", "lesson", "constraint", "architecture", "pattern")),
       "missing files/dirs")

    r2 = p.run("init")
    ok("re-init refuses (exit 1)", r2.returncode == 1 and "already exists" in r2.stderr,
       f"rc={r2.returncode} err={r2.stderr!r}")

    # --force on a populated store must refuse (no silent clobber)
    p.add("decision", "keep me", "content here")
    r3 = p.run("init", "--force")
    ok("init --force refuses populated store", r3.returncode == 1 and "not empty" in r3.stderr,
       f"rc={r3.returncode} err={r3.stderr!r}")

    # --project
    q = Project()
    q.run("init", "--project", "custom-proj")
    q.add("decision", "t", "c")
    st = q.run("status", "--json")
    ok("init --project stored in config", '"custom-proj"' in st.stdout, st.stdout)

    # --dir
    tgt = pathlib.Path(tempfile.mkdtemp(prefix="am-target-")) / "sub"
    rr = q.run("init", "--dir", str(tgt))
    ok("init --dir creates store at target", rr.returncode == 0 and (tgt / ".agent" / "config.toml").exists(),
       f"rc={rr.returncode}")


# ==========================================================================
# add
# ==========================================================================

def audit_add():
    p = Project()
    p.init()

    r = p.add("decision", "Auth via AuthService", "All auth must use AuthService", "--paths", "src/auth/**")
    ok("add valid exit 0 + created", r.returncode == 0 and re.search(r"created mem_", r.stdout),
       f"rc={r.returncode} out={r.stdout!r} err={r.stderr!r}")

    rj = p.add("error", "Refresh token reuse", "Tokens were reusable", "--json")
    rec = json.loads(rj.stdout)
    ok("add --json record shape",
       rec["format_version"] == 1 and rec["type"] == "error" and rec["title"] == "Refresh token reuse"
       and rec["content"] == "Tokens were reusable" and rec["project"] == "demo"
       and rec["provenance"] == "human" and rec["trust"] == "untrusted"
       and rec["status"] == "active" and rec["severity"] == "normal"
       and rec["source"] is None and rec["tags"] == [] and rec["paths"] == [],
       json.dumps(rec)[:200])
    ok("add --json UTC timestamps",
       bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", rec["created_at"]))
       and rec["created_at"] == rec["updated_at"],
       f"{rec['created_at']!r} {rec['updated_at']!r}")

    for label, argv in [
        ("add missing --title", ["add", "--type", "decision", "--content", "x"]),
        ("add missing --content", ["add", "--type", "decision", "--title", "x"]),
        ("add missing --type", ["add", "--title", "x", "--content", "y"]),
        ("add empty --title", ["add", "--type", "decision", "--title", "", "--content", "y"]),
        ("add empty --content", ["add", "--type", "decision", "--title", "x", "--content", "  "]),
    ]:
        rr = p.run(*argv)
        ok(f"{label} -> exit 2", rr.returncode == 2, f"rc={rr.returncode} err={rr.stderr!r}")

    rr = p.run("add", "--type", "bogus", "--title", "x", "--content", "y")
    ok("add invalid --type -> exit 2", rr.returncode == 2, f"rc={rr.returncode}")
    rr = p.run("add", "--type", "decision", "--title", "x", "--content", "y", "--severity", "extreme")
    ok("add invalid --severity -> exit 2", rr.returncode == 2, f"rc={rr.returncode}")
    rr = p.run("add", "--type", "decision", "--title", "x", "--content", "y", "--provenance", "system")
    ok("add --provenance system -> exit 2", rr.returncode == 2, f"rc={rr.returncode}")
    rr = p.add("decision", "x", "y", "--paths", "src/[ab]/**")
    ok("add char-class path rejected -> exit 2", rr.returncode == 2 and "character classes" in rr.stderr,
       f"rc={rr.returncode} err={rr.stderr!r}")

    # tags + paths roundtrip
    rt = p.add("decision", "Tagged mem", "tagged content", "--tags", "alpha, beta", "--paths", "src/**")
    ok("add --tags/--paths exit 0", rt.returncode == 0, rt.stderr)
    mid = re.search(r"created (mem_[0-9a-f-]+)", rt.stdout).group(1)
    shown = json.loads(p.run("show", mid, "--json").stdout)
    ok("tags/paths stored as lists", shown["tags"] == ["alpha", "beta"] and shown["paths"] == ["src/**"],
       f"{shown['tags']} {shown['paths']}")

    # secrets
    for pat, sample in [
        ("github PAT", "my token is ghp_" + "A" * 36),
        ("openai-style", "sk-" + "A" * 24),
        ("aws key", "AKIAIOSFODNN7EXAMPLE"),
        ("private key", "-----BEGIN RSA PRIVATE KEY-----"),
        ("conn string", "postgres://user:pass123@localhost:5432/db"),
        ("JWT", "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"),
        ("inline secret", "password = s3cr3t-value-9"),
    ]:
        rr = p.add("lesson", "secret check", f"content with {sample}")
        ok(f"secret {pat} rejected exit 1", rr.returncode == 1 and "potential secret" in rr.stderr,
           f"rc={rr.returncode} err={rr.stderr!r}")

    # entropy heuristic
    rr = p.add("lesson", "entropy check", "token Kx9fT2mQ8vL4nR7sW1cZ5yA3")
    ok("high-entropy token rejected", rr.returncode == 1 and "potential secret" in rr.stderr, rr.stderr)

    # MEMORY_REJECTED audit recorded
    ok("MEMORY_REJECTED audited", p.audit_events().count("MEMORY_REJECTED") >= 8,
       str(p.audit_events()))

    # --allow-secret stores + audits
    rr = p.add("lesson", "test fixture", "ghp_" + "B" * 36, "--allow-secret")
    ok("--allow-secret stores", rr.returncode == 0, f"rc={rr.returncode} err={rr.stderr!r}")
    ok("SECRET_OVERRIDE audited", "SECRET_OVERRIDE" in p.audit_events(), str(p.audit_events()))

    # Unicode
    uni_title = "Zażółć gęślą jaźń キャッシュ 🚀"
    uni_content = "内容 – emoji 🎯 and 中文"
    ru = p.add("decision", uni_title, uni_content)
    ok("add unicode exit 0", ru.returncode == 0, f"rc={ru.returncode} err={ru.stderr!r}")
    uid = re.search(r"created (mem_[0-9a-f-]+)", ru.stdout).group(1)
    shown = p.run("show", uid, "--json").stdout
    stored = json.loads(shown)
    ok("unicode roundtrips", stored["title"] == uni_title and stored["content"] == uni_content,
       f"{stored['title']!r}")

    # interactive mode
    ri = p.run("add", stdin="constraint\nInteractive title\nInteractive body\n\n\n")
    ok("interactive add (piped) exit 0", ri.returncode == 0 and "created mem_" in ri.stdout,
       f"rc={ri.returncode} out={ri.stdout!r}")
    re_ = p.run("add", stdin="")
    ok("interactive add EOF -> exit 2 usage", re_.returncode == 2 and "terminal" in re_.stderr,
       f"rc={re_.returncode} err={re_.stderr!r}")


# ==========================================================================
# list / show
# ==========================================================================

def audit_list_show():
    p = Project()
    p.init()

    r = p.run("list")
    ok("list empty -> 0 results exit 0", r.returncode == 0 and "0 results" in r.stdout, f"{r.stdout!r}")

    p.add("decision", "Alpha", "first")
    p.add("error", "Beta", "second")
    r = p.run("list")
    ok("list shows 2 + count", "2 result(s)" in r.stdout and "Alpha" in r.stdout and "Beta" in r.stdout,
       r.stdout)

    r = p.run("list", "--type", "error")
    ok("list --type filters", "Beta" in r.stdout and "Alpha" not in r.stdout, r.stdout)

    r = p.run("list", "--json")
    payload = json.loads(r.stdout)
    ok("list --json shape", set(payload) == {"results", "count"} and payload["count"] == 2
       and len(payload["results"]) == 2, r.stdout[:200])

    r = p.run("list", "--type", "bogus")
    ok("list invalid --type -> exit 2", r.returncode == 2, f"rc={r.returncode}")

    # show
    mid = p.add_ids("decision", "ShowMe", "shown content")[0]
    r = p.run("show", mid)
    ok("show exit 0 + content", r.returncode == 0 and "ShowMe" in r.stdout, f"rc={r.returncode} {r.stdout!r}")
    rec = json.loads(p.run("show", mid, "--json").stdout)
    ok("show --json parses + id matches", rec["id"] == mid and rec["title"] == "ShowMe", rec.get("id"))

    r = p.run("show", "mem_00000000-0000-0000-0000-000000000000")
    ok("show missing id -> exit 1 clean", r.returncode == 1 and "no memory with id" in r.stderr,
       f"rc={r.returncode} err={r.stderr!r}")

    # corrupt memory file
    f = p.store() / "memory" / "decision" / "mem_c0ffee00-0000-0000-0000-000000000000.json"
    f.write_text("{ this is not json", encoding="utf-8")
    r = p.run("list")
    ok("corrupt memory -> list exit 1 clean", r.returncode == 1 and "corrupt memory file" in r.stderr,
       f"rc={r.returncode} err={r.stderr!r}")
    r = p.run("show", "mem_c0ffee00-0000-0000-0000-000000000000")
    ok("corrupt memory -> show exit 1 clean", r.returncode == 1 and "corrupt memory file" in r.stderr,
       f"rc={r.returncode} err={r.stderr!r}")

    # MEMORY_ACCESSED audit on show
    events = p.audit_events()
    ok("show emits MEMORY_ACCESSED", events.count("MEMORY_ACCESSED") >= 2, str(events))


# ==========================================================================
# search
# ==========================================================================

def audit_search():
    p = Project()
    p.init()
    p.add("constraint", "Authentication must use AuthService", "use AuthService everywhere", "--tags", "auth,security")
    p.add("decision", "Use Redis cache", "Redis chosen for caching", "--tags", "cache")
    p.add("error", "Old bug", "this was fixed and superseded")

    r = p.run("search", "auth")
    ok("search match exit 0 + count", r.returncode == 0 and "1 result(s)" in r.stdout
       and "Authentication" in r.stdout, r.stdout)
    r = p.run("search", "AUTH")
    ok("search case-insensitive", "1 result(s)" in r.stdout, r.stdout)
    r = p.run("search", "no-such-term-xyz")
    ok("search no match -> 0 results exit 0", r.returncode == 0 and "0 results" in r.stdout, r.stdout)
    r = p.run("search", "")
    ok("search empty query -> 0 results exit 0", r.returncode == 0 and "0 results" in r.stdout, r.stdout)
    r = p.run("search", "--limit", "0")
    ok("search --limit 0 -> exit 2", r.returncode == 2, f"rc={r.returncode} err={r.stderr!r}")
    r = p.run("search", "--limit", "-2")
    ok("search --limit negative -> exit 2", r.returncode == 2, f"rc={r.returncode}")

    # 3 matches, limit 1
    for i in range(3):
        p.add("lesson", f"widget memory {i}", "widget related content")
    r = p.run("search", "widget", "--limit", "1", "--json")
    payload = json.loads(r.stdout)
    ok("search --limit clamps", payload["count"] == 1, r.stdout[:120])

    r = p.run("search", "widget", "--json")
    payload = json.loads(r.stdout)
    titles = [x["title"] for x in payload["results"]]
    ok("search default limit 50 + all matches", payload["count"] == 3 and
       set(titles) == {"widget memory 0", "widget memory 1", "widget memory 2"}, str(titles))

    # deleted excluded by default; --status deleted includes
    p2 = Project()
    p2.init()
    mid = p2.add_ids("decision", "Ghost", "deleted soon")[0]
    p2.run("delete", mid)
    r = p2.run("search", "Ghost")
    ok("search default excludes deleted", "0 results" in r.stdout, r.stdout)
    r = p2.run("search", "Ghost", "--status", "deleted")
    ok("search --status deleted includes", "1 result(s)" in r.stdout, r.stdout)

    # unicode search
    p.add("decision", "UnicodeMatch 検索", "searchable unicode")
    r = p.run("search", "検索")
    ok("search unicode query", "1 result(s)" in r.stdout, r.stdout)

    # MEMORY_ACCESSED batched audit
    before = len(p.audit_events())
    p.run("search", "widget")
    ok("search emits batched MEMORY_ACCESSED", len(p.audit_events()) == before + 1,
       f"{before} -> {len(p.audit_events())}")

    # --json error contract
    r = p.run("search", "--limit", "0", "--json")
    parsed = json.loads(r.stdout)
    ok("search --json usage error -> stdout JSON, stderr empty, rc 2",
       r.returncode == 2 and "error" in parsed and r.stderr == "", f"rc={r.returncode} out={r.stdout!r} err={r.stderr!r}")


# ==========================================================================
# recall
# ==========================================================================

def audit_recall():
    p = Project()
    p.init()
    p.add("constraint", "Auth constraint", "use AuthService for all auth", "--paths", "src/auth/**")
    p.add("decision", "Auth note", "auth-related decision", "--paths", "other/**")

    r = p.run("recall", "auth")
    ok("recall excludes untrusted -> 0 results", r.returncode == 0 and "0 results" in r.stdout, r.stdout)
    r = p.run("recall", "auth", "--include-untrusted", "--json")
    payload = json.loads(r.stdout)
    ok("recall --include-untrusted includes", payload["count"] == 2, r.stdout[:120])

    # promote both, then recall with path bonus
    ids = p.all_ids()
    for mid in ids:
        p.run("promote", mid, "--trust", "verified")
    r = p.run("recall", "auth", "--path", "src/auth/login.py", "--json")
    payload = json.loads(r.stdout)
    ok("recall trusted + path bonus ranks path match first",
       payload["count"] == 2 and payload["results"][0]["title"] == "Auth constraint",
       [x["title"] for x in payload["results"]])

    base = json.loads(p.run("recall", "auth", "--json").stdout)
    r = p.run("recall", "auth", "--path", "src/other/thing.py", "--json")
    payload = json.loads(r.stdout)
    ok("recall non-matching path == no-path baseline",
       [x["id"] for x in payload["results"]] == [x["id"] for x in base["results"]],
       [x["title"] for x in payload["results"]])

    r = p.run("recall", "auth", "--limit", "0")
    ok("recall --limit 0 -> exit 2", r.returncode == 2, f"rc={r.returncode}")

    r = p.run("recall", "")
    ok("recall empty query -> 0 results exit 0", r.returncode == 0 and "0 results" in r.stdout, r.stdout)

    # default limit 10
    p2 = Project()
    p2.init()
    mids = []
    for i in range(12):
        mids += p2.add_ids("decision", f"widget {i}", f"widget content {i}")
    for mid in mids:
        p2.run("promote", mid, "--trust", "approved")
    r = p2.run("recall", "widget", "--json")
    payload = json.loads(r.stdout)
    ok("recall default limit 10", payload["count"] == 10, f"count={payload['count']}")

    # config recall_limit override
    cfg = p2.config().read_text(encoding="utf-8").replace("recall_limit = 10", "recall_limit = 2")
    p2.config().write_text(cfg, encoding="utf-8", newline="\n")
    r = p2.run("recall", "widget", "--json")
    payload = json.loads(r.stdout)
    ok("recall config recall_limit honored", payload["count"] == 2, f"count={payload['count']}")


# ==========================================================================
# promote
# ==========================================================================

def audit_promote():
    p = Project()
    p.init()
    mid = p.add_ids("decision", "Promotable", "trust me")[0]

    r = p.run("promote", mid, "--trust", "verified")
    ok("promote untrusted->verified exit 0", r.returncode == 0 and "promoted" in r.stdout,
       f"rc={r.returncode} err={r.stderr!r}")
    rec = json.loads(p.run("show", mid, "--json").stdout)
    ok("promote persisted", rec["trust"] == "verified", rec.get("trust"))

    r = p.run("promote", mid, "--trust", "approved")
    ok("promote verified->approved exit 0", r.returncode == 0, f"rc={r.returncode}")
    r = p.run("promote", mid, "--trust", "approved")
    ok("promote to same level refused", r.returncode == 1 and "already trust" in r.stderr,
       f"rc={r.returncode} err={r.stderr!r}")

    r = p.run("promote", mid, "--trust", "system")
    ok("promote --trust system -> exit 2", r.returncode == 2, f"rc={r.returncode}")

    r = p.run("promote", "mem_00000000-0000-0000-0000-000000000000", "--trust", "verified")
    ok("promote missing id -> exit 1", r.returncode == 1 and "no memory with id" in r.stderr,
       f"rc={r.returncode}")

    # single jump untrusted -> approved
    mid2 = p.add_ids("decision", "DirectApproved", "jump")[0]
    r = p.run("promote", mid2, "--trust", "approved")
    ok("promote untrusted->approved single jump", r.returncode == 0, f"rc={r.returncode}")

    # superseded / deleted cannot promote
    old = p.add_ids("decision", "OldOne", "will be superseded")[0]
    new = p.add_ids("decision", "NewOne", "replacement")[0]
    p.run("supersede", old, new)
    r = p.run("promote", old, "--trust", "verified")
    ok("promote superseded refused", r.returncode == 1 and "only active" in r.stderr,
       f"rc={r.returncode} err={r.stderr!r}")
    del_id = p.add_ids("decision", "Doomed", "deleted")[0]
    p.run("delete", del_id)
    r = p.run("promote", del_id, "--trust", "verified")
    ok("promote deleted refused", r.returncode == 1 and "only active" in r.stderr,
       f"rc={r.returncode} err={r.stderr!r}")

    ok("TRUST_PROMOTED audited", p.audit_events().count("TRUST_PROMOTED") >= 3, str(p.audit_events()))


# ==========================================================================
# supersede
# ==========================================================================

def audit_supersede():
    p = Project()
    p.init()
    old = p.add_ids("decision", "OldDecision", "old reasoning")[0]
    new = p.add_ids("decision", "NewDecision", "new reasoning")[0]

    r = p.run("supersede", old, new)
    ok("supersede exit 0", r.returncode == 0 and "superseded by" in r.stdout, f"{r.stdout!r}")
    o = json.loads(p.run("show", old, "--json").stdout)
    n = json.loads(p.run("show", new, "--json").stdout)
    ok("bidirectional links set",
       o["status"] == "superseded" and o["superseded_by"] == new
       and n["status"] == "active" and n["supersedes"] == old,
       f"old={o['status']}/{o['superseded_by']} new={n['status']}/{n['supersedes']}")

    r = p.run("supersede", old, new)
    ok("supersede old already superseded refused", r.returncode == 1 and "only active" in r.stderr,
       f"rc={r.returncode} err={r.stderr!r}")

    # no chains: the NEW memory must not already supersede something
    p3 = Project()
    p3.init()
    a = p3.add_ids("decision", "ChainA", "first link")[0]
    b = p3.add_ids("decision", "ChainB", "second link")[0]
    c = p3.add_ids("decision", "ChainC", "would be third")[0]
    p3.run("supersede", a, b)   # b now supersedes a
    r = p3.run("supersede", c, b)  # b already supersedes -> no chains
    ok("supersede new already supersedes (no chains) refused", r.returncode == 1 and "no chains" in r.stderr,
       f"rc={r.returncode} err={r.stderr!r}")

    r = p.run("supersede", old, old)
    ok("supersede same id -> exit 2", r.returncode == 2, f"rc={r.returncode}")

    r = p.run("supersede", "mem_00000000-0000-0000-0000-000000000000", new)
    ok("supersede missing old -> exit 1", r.returncode == 1, f"rc={r.returncode}")
    r = p.run("supersede", old, "mem_00000000-0000-0000-0000-000000000000")
    ok("supersede missing new -> exit 1", r.returncode == 1, f"rc={r.returncode}")

    # deleted old
    p2 = Project()
    p2.init()
    d = p2.add_ids("decision", "DeletedOld", "gone")[0]
    n2 = p2.add_ids("decision", "FreshNew", "fresh")[0]
    p2.run("delete", d)
    r = p2.run("supersede", d, n2)
    ok("supersede deleted old refused", r.returncode == 1, f"rc={r.returncode} err={r.stderr!r}")

    ok("MEMORY_SUPERSEDED audited", "MEMORY_SUPERSEDED" in p.audit_events(), str(p.audit_events()))


# ==========================================================================
# delete
# ==========================================================================

def audit_delete():
    p = Project()
    p.init()
    mid = p.add_ids("decision", "TombstoneMe", "deletable")[0]

    r = p.run("delete", mid)
    ok("delete tombstone exit 0", r.returncode == 0 and "tombstone" in r.stdout, r.stdout)
    ok("tombstone file retained", (p.store() / "memory" / "decision" / f"{mid}.json").exists())
    rec = json.loads(p.run("show", mid, "--json").stdout)
    ok("tombstone fields set",
       rec["status"] == "deleted" and rec["deleted_at"] and rec["deleted_by"] == "human",
       f"{rec['status']} {rec['deleted_at']} {rec['deleted_by']}")
    r = p.run("delete", mid)
    ok("delete twice refused", r.returncode == 1 and "already deleted" in r.stderr, f"rc={r.returncode}")

    # purge untrusted only
    garbage = p.add_ids("lesson", "Garbage", "never approved")[0]
    r = p.run("delete", garbage, "--purge")
    ok("purge untrusted exit 0", r.returncode == 0 and "purged" in r.stdout, r.stdout)
    ok("purge removes file", not (p.store() / "memory" / "lesson" / f"{garbage}.json").exists())
    r = p.run("show", garbage)
    ok("purged memory gone", r.returncode == 1, f"rc={r.returncode}")

    approved = p.add_ids("constraint", "ApprovedConst", "must keep")[0]
    p.run("promote", approved, "--trust", "approved")
    r = p.run("delete", approved, "--purge")
    ok("purge approved refused", r.returncode == 1 and "untrusted" in r.stderr,
       f"rc={r.returncode} err={r.stderr!r}")

    r = p.run("delete", "mem_00000000-0000-0000-0000-000000000000")
    ok("delete missing id -> exit 1", r.returncode == 1, f"rc={r.returncode}")

    ok("MEMORY_DELETED audited", p.audit_events().count("MEMORY_DELETED") >= 2, str(p.audit_events()))


# ==========================================================================
# status
# ==========================================================================

def audit_status():
    p = Project()
    p.init()
    p.add("decision", "One", "content")
    p.add("error", "Two", "content")
    r = p.run("status")
    ok("status exit 0 + counts", r.returncode == 0 and "total memories: 2" in r.stdout, r.stdout)
    payload = json.loads(p.run("status", "--json").stdout)
    ok("status --json shape",
       payload["project"] == "demo" and payload["total"] == 2
       and payload["by_type"]["decision"] == 1 and payload["by_type"]["error"] == 1
       and payload["by_status"]["active"] == 2 and payload["by_trust"]["untrusted"] == 2,
       json.dumps(payload)[:300])

    q = Project()
    r = q.run("status")
    ok("status not-initialized -> exit 1 + hint", r.returncode == 1 and "run: agent-memory init" in r.stderr,
       f"rc={r.returncode} err={r.stderr!r}")


# ==========================================================================
# suggestions - T3 approve-to-persist review loop (human CLI side)
# ==========================================================================

def audit_suggestions():
    p = Project()
    p.init()
    r = p.run("suggestions", "list")
    ok("suggestions: empty list -> 0 results exit 0",
       r.returncode == 0 and "0 results" in r.stdout,
       f"rc={r.returncode} out={r.stdout!r}")

    # Seed two pending suggestions exactly as the core writes them (the
    # agent surface is MCP; the CLI owns the human review side).
    sug_dir = p.store() / "suggestions"
    sug_dir.mkdir(parents=True, exist_ok=True)
    base = {
        "format_version": 1, "type": "error", "title": "t",
        "content": "c", "project": p.name, "provenance": "agent",
        "severity": "normal", "tags": [], "paths": [],
        "created_at": "2026-08-13T00:00:00Z", "state": "pending",
        "approved_at": None, "approved_by": None,
        "rejected_at": None, "rejected_by": None,
    }
    s1 = dict(base, id="sug_11111111-0000-0000-0000-000000000001",
              title="refresh tokens", content="tokens were reusable")
    s2 = dict(base, id="sug_11111111-0000-0000-0000-000000000002",
              title="bad idea", content="should never become memory")
    for s in (s1, s2):
        (sug_dir / f"{s['id']}.json").write_text(
            json.dumps(s, indent=2) + "\n", encoding="utf-8", newline="\n")

    r = p.run("suggestions", "list")
    ok("suggestions: pending listed with state",
       r.returncode == 0 and "refresh tokens" in r.stdout
       and "state=pending" in r.stdout, r.stdout)

    r = p.run("suggestions", "approve", "sug_11111111-0000-0000-0000-000000000001")
    ok("suggestions: approve without --trust -> exit 2", r.returncode == 2,
       f"rc={r.returncode}")
    r = p.run("suggestions", "approve", "sug_11111111-0000-0000-0000-000000000001",
              "--trust", "system")
    ok("suggestions: approve --trust system -> exit 2", r.returncode == 2,
       f"rc={r.returncode}")
    r = p.run("suggestions", "approve", "sug_11111111-0000-0000-0000-000000000001",
              "--trust", "approved")
    ok("suggestions: approve exit 0 + trust + memory id",
       r.returncode == 0 and "trust=approved" in r.stdout and "mem_" in r.stdout,
       f"rc={r.returncode} out={r.stdout!r}")
    m = re.search(r"memory (mem_[0-9a-f-]+)", r.stdout)
    mem_id = m.group(1) if m else ""
    r = p.run("recall", "refresh")
    ok("suggestions: approved suggestion recalled", "RELEVANT CONTEXT" in r.stdout,
       r.stdout)
    r = p.run("suggestions", "approve", "sug_11111111-0000-0000-0000-000000000001",
              "--trust", "verified")
    ok("suggestions: double-approve -> exit 1 (terminal)", r.returncode == 1,
       f"rc={r.returncode} err={r.stderr!r}")

    r = p.run("suggestions", "reject", "sug_11111111-0000-0000-0000-000000000002", "--json")
    parsed = json.loads(r.stdout)
    ok("suggestions: reject --json -> rejected record",
       r.returncode == 0 and parsed["state"] == "rejected", f"out={r.stdout!r}")
    r = p.run("suggestions", "approve", "sug_11111111-0000-0000-0000-000000000002",
              "--trust", "verified")
    ok("suggestions: approve after reject -> exit 1 (terminal)", r.returncode == 1,
       f"rc={r.returncode}")
    r = p.run("suggestions", "approve", "sug_none", "--trust", "verified")
    ok("suggestions: missing id -> exit 1 clean message",
       r.returncode == 1 and "no suggestion" in (r.stderr or ""),
       f"rc={r.returncode} err={r.stderr!r}")

    events = p.audit_events()
    ok("suggestions: SUGGESTION_APPROVED + REJECTED audited",
       events.count("SUGGESTION_APPROVED") == 1
       and events.count("SUGGESTION_REJECTED") == 1, str(events))
    ids = p.all_ids()
    ok("suggestions: only the approved suggestion became memory",
       len(ids) == 1 and mem_id in ids, str(ids))
    st = json.loads(p.run("status", "--json").stdout)
    ok("suggestions: status pending count 0 after review",
       st["suggestions_pending"] == 0, json.dumps(st)[:200])


def audit_conflicts():
    """T4 (EVIDENCE-038): narrow possible-conflict detection, human-only
    review (dismiss / resolve via existing supersede), audited lifecycle."""
    p = Project()
    p.init()
    r = p.run("conflicts", "scan")
    ok("conflicts: empty store scan -> 0 candidates exit 0",
       r.returncode == 0 and "0 new possible conflict" in r.stdout,
       f"rc={r.returncode} out={r.stdout!r}")
    r = p.run("conflicts", "list")
    ok("conflicts: empty list -> 0 results exit 0",
       r.returncode == 0 and "0 results" in r.stdout,
       f"rc={r.returncode} out={r.stdout!r}")

    # Narrow candidate: same type + shared path + meaningful overlap.
    p.add("decision", "auth refresh token timeout",
          "token refresh invalidation session policy",
          "--paths", "src/auth/**", "--tags", "auth")
    p.add("decision", "auth session invalidation",
          "session token refresh invalidation policy",
          "--paths", "src/auth/session.py", "--tags", "auth")
    p.add("decision", "frontend button layout", "padding margin css layout",
          "--paths", "src/frontend/**")  # unrelated scope, never flagged
    r = p.run("conflicts", "scan")
    ok("conflicts: scan finds exactly the overlapping pair",
       r.returncode == 0 and "1 new possible conflict" in r.stdout,
       f"rc={r.returncode} out={r.stdout!r}")
    r = p.run("conflicts", "scan")
    ok("conflicts: re-scan dedupes (0 new, 1 already open)",
       r.returncode == 0 and "0 new possible conflict" in r.stdout
       and "1 already open" in r.stdout, f"out={r.stdout!r}")
    r = p.run("conflicts", "list", "--json")
    data = json.loads(r.stdout)
    ok("conflicts: list --json 1 open record with explanation",
       data["count"] == 1 and data["results"][0]["state"] == "open"
       and "shared_high_weight_terms" in data["results"][0]["explanation"]
       and "winner" not in data["results"][0]["explanation"],
       f"out={r.stdout[:300]!r}")
    cf = data["results"][0]["id"]
    ok("conflicts: id space is cf_", cf.startswith("cf_"), cf)

    # Dismiss path: terminal while memories unchanged.
    r = p.run("conflicts", "dismiss", cf)
    ok("conflicts: dismiss exit 0", r.returncode == 0 and "dismissed" in r.stdout,
       f"rc={r.returncode} out={r.stdout!r}")
    r = p.run("conflicts", "dismiss", cf)
    ok("conflicts: re-dismiss -> exit 1 (terminal)", r.returncode == 1,
       f"rc={r.returncode} err={r.stderr!r}")
    r = p.run("conflicts", "scan")
    ok("conflicts: dismissed-unchanged pair skipped by scan",
       r.returncode == 0 and "1 dismissed-unchanged skipped" in r.stdout,
       f"out={r.stdout!r}")
    r = p.run("conflicts", "list", "--state", "dismissed")
    ok("conflicts: dismissed record listed by state filter",
       "state=dismissed" in r.stdout, f"out={r.stdout!r}")

    # Resolve path: reuses the existing supersede() machinery.
    p2 = Project()
    p2.init()
    p2.add("decision", "auth refresh token timeout",
           "token refresh invalidation session policy",
           "--paths", "src/auth/**", "--tags", "auth")
    p2.add("decision", "auth session invalidation",
           "session token refresh invalidation policy",
           "--paths", "src/auth/session.py", "--tags", "auth")
    p2.run("conflicts", "scan")
    r = p2.run("conflicts", "list", "--json")
    cf2 = json.loads(r.stdout)["results"][0]["id"]
    ids = p2.all_ids()
    old = ids[0]
    new = ids[1]
    r = p2.run("conflicts", "resolve", cf2, "--old", old, "--new", new)
    ok("conflicts: resolve exit 0 + supersede message",
       r.returncode == 0 and "superseded by" in r.stdout,
       f"rc={r.returncode} out={r.stdout!r}")
    o = json.loads(p2.run("show", old, "--json").stdout)
    n = json.loads(p2.run("show", new, "--json").stdout)
    ok("conflicts: resolve wired supersession links",
       o["status"] == "superseded" and o["superseded_by"] == new
       and n["supersedes"] == old, f"old={o['status']} new={n['supersedes']}")
    r = p2.run("conflicts", "resolve", cf2, "--old", old, "--new", new)
    ok("conflicts: re-resolve -> exit 1 (terminal)", r.returncode == 1,
       f"rc={r.returncode}")

    # Guards and errors.
    p3 = Project()
    p3.init()
    p3.add("decision", "auth refresh token timeout",
           "token refresh invalidation session policy",
           "--paths", "src/auth/**", "--tags", "auth")
    p3.add("decision", "auth session invalidation",
           "session token refresh invalidation policy",
           "--paths", "src/auth/session.py", "--tags", "auth")
    p3.run("conflicts", "scan")
    r = p3.run("conflicts", "list", "--json")
    cf3 = json.loads(r.stdout)["results"][0]["id"]
    r = p3.run("conflicts", "resolve")
    ok("conflicts: resolve missing args -> exit 2", r.returncode == 2,
       f"rc={r.returncode}")
    r = p3.run("conflicts", "resolve", cf3, "--old", "mem_x", "--new", "mem_y")
    ok("conflicts: resolve missing memories -> exit 1", r.returncode == 1,
       f"rc={r.returncode}")
    r = p3.run("conflicts", "dismiss", "cf_00000000-0000-0000-0000-000000000000")
    ok("conflicts: dismiss missing id -> exit 1 clean message",
       r.returncode == 1 and "no conflict record" in (r.stderr or ""),
       f"rc={r.returncode} err={r.stderr!r}")
    r = p3.run("conflicts", "bogus")
    ok("conflicts: bad action -> exit 2", r.returncode == 2, f"rc={r.returncode}")

    events = p.audit_events()
    ok("conflicts: CONFLICT_DETECTED + CONFLICT_DISMISSED audited",
       "CONFLICT_DETECTED" in events and "CONFLICT_DISMISSED" in events,
       str(events))
    events2 = p2.audit_events()
    ok("conflicts: MEMORY_SUPERSEDED + CONFLICT_RESOLVED audited on resolve",
       "MEMORY_SUPERSEDED" in events2 and "CONFLICT_RESOLVED" in events2,
       str(events2))
    st = json.loads(p2.run("status", "--json").stdout)
    ok("conflicts: status reports open conflict count 0 after resolve",
       st["conflicts_open"] == 0, json.dumps(st)[:200])
    st1 = json.loads(p.run("status", "--json").stdout)
    ok("conflicts: status reports open count on dismissed project",
       st1["conflicts_open"] == 0, json.dumps(st1)[:200])


# ==========================================================================
# not-initialized across commands + project discovery + corrupt config
# ==========================================================================

def audit_discovery_errors():
    p = Project()
    for cmd in (["list"], ["show", "mem_x"], ["search", "q"], ["recall", "q"],
                ["add", "--type", "decision", "--title", "t", "--content", "c"],
                ["promote", "mem_x", "--trust", "verified"],
                ["supersede", "a", "b"], ["delete", "mem_x"], ["status"]):
        r = p.run(*cmd)
        ok(f"not-initialized {cmd[0]} -> exit 1 + hint", r.returncode == 1 and "run: agent-memory init" in r.stderr,
           f"rc={r.returncode} err={r.stderr!r}")

    # parent-store discovery (git-style walk-up)
    parent = Project()
    parent.init()
    parent.add("decision", "FromParent", "found via walk-up")
    (parent.dir / "nested" / "deep").mkdir(parents=True)
    r = parent.run("list", sub="nested/deep")
    ok("store found via parent walk-up", r.returncode == 0 and "FromParent" in r.stdout,
       f"rc={r.returncode} err={r.stderr!r}")

    # corrupt config.toml
    c = Project()
    c.init()
    c.config().write_text("project = [broken", encoding="utf-8")
    r = c.run("status")
    ok("corrupt config -> exit 1 clean", r.returncode == 1 and "malformed config.toml" in r.stderr,
       f"rc={r.returncode} err={r.stderr!r}")


# ==========================================================================
# json error contract, stdout/stderr discipline, unicode, determinism, meta
# ==========================================================================

def audit_json_errors_and_meta():
    p = Project()
    p.init()

    # json error contract (our errors)
    r = p.run("show", "mem_00000000-0000-0000-0000-000000000000", "--json")
    parsed = json.loads(r.stdout)
    ok("show --json runtime error -> stdout JSON rc 1 stderr empty",
       r.returncode == 1 and "error" in parsed and r.stderr == "", f"rc={r.returncode} out={r.stdout!r} err={r.stderr!r}")

    r = p.run("add", "--type", "decision", "--title", "x", "--json")
    parsed = json.loads(r.stdout)
    ok("add --json usage error -> stdout JSON rc 2 stderr empty",
       r.returncode == 2 and "error" in parsed and r.stderr == "", f"rc={r.returncode} out={r.stdout!r} err={r.stderr!r}")

    # json error contract (argparse-level errors)
    r = p.run("add", "--type", "bogus", "--json")
    parsed = json.loads(r.stdout) if r.stdout.strip() else {}
    ok("argparse error in --json mode -> stdout JSON rc 2, stderr empty",
       r.returncode == 2 and "error" in parsed and r.stderr == "",
       f"rc={r.returncode} out={r.stdout!r} err={r.stderr!r}")

    # stdout/stderr discipline (text mode)
    r = p.add("decision", "Clean", "clean output")
    ok("success: stdout only, stderr empty", r.returncode == 0 and r.stderr == "", f"err={r.stderr!r}")
    r = p.run("show", "mem_00000000-0000-0000-0000-000000000000")
    ok("text error: stderr only, stdout empty", r.returncode == 1 and r.stdout == "" and "error:" in r.stderr,
       f"out={r.stdout!r} err={r.stderr!r}")

    # unicode search
    p.add("decision", "검색 테스트", "unicode target")
    r = p.run("search", "검색")
    ok("unicode search finds record", "1 result(s)" in r.stdout, r.stdout)

    # determinism: identical search runs -> byte-identical stdout
    p.add("decision", "DetCheck", "deterministic ordering")
    a = p.run("search", "deterministic")
    b = p.run("search", "deterministic")
    ok("identical search runs byte-identical", a.returncode == b.returncode == 0 and a.stdout == b.stdout,
       f"{a.stdout!r} != {b.stdout!r}")

    # stored memory files are LF-only (no Windows CRLF translation)
    files = list((p.store() / "memory").rglob("*.json"))
    ok("memory files LF-only + valid JSON",
       len(files) > 0 and all(b"\r\n" not in f.read_bytes() for f in files) and all(
           json.loads(f.read_text(encoding="utf-8"))["format_version"] == 1 for f in files),
       f"{len(files)} files")

    # audit file append-only integrity: every line valid JSON
    ok("audit lines all valid JSON", all(json.loads(ln) for ln in p.audit_lines()),
       str(p.audit_lines()[:3]))

    # meta
    r = p.run("--version")
    ok("--version exit 0 + value", r.returncode == 0 and "agent-memory 0.2.0" in r.stdout, r.stdout)
    r = p.run("--help")
    ok("--help exit 0", r.returncode == 0 and "usage" in r.stdout.lower(), f"rc={r.returncode}")
    r = p.run("frobnicate")
    ok("unknown command -> exit 2", r.returncode == 2, f"rc={r.returncode}")
    r = p.run("list", "--bogus-flag")
    ok("unknown flag -> exit 2", r.returncode == 2, f"rc={r.returncode}")
    r = p.run("export")
    ok("export not in v0.1 contract -> exit 2 (documented)", r.returncode == 2, f"rc={r.returncode}")

    # project name with quote must roundtrip (TOML escaping)
    q = Project()
    rr = q.run("init", "--project", 'we"ird')
    ok("init --project with quote succeeds", rr.returncode == 0, f"rc={rr.returncode} err={rr.stderr!r}")
    st = json.loads(q.run("status", "--json").stdout)
    ok("quoted project roundtrips via config", st["project"] == 'we"ird', st["project"])

    # project name with backslash must roundtrip
    q2 = Project()
    q2.run("init", "--project", "a\\b")
    st = json.loads(q2.run("status", "--json").stdout)
    ok("backslash project roundtrips via config", st["project"] == "a\\b", st["project"])


# ==========================================================================
# git - T5 write-time context + stored-snapshot recall enrichment (EVIDENCE-041)
# ==========================================================================

def audit_git():
    """T5 (EVIDENCE-041): versioned sidecar capture in a real git repo,
    CLI review surface, non-git fail-soft fallback, recall --branch."""
    # -- non-git project: fail-soft, no gitcontext dir, no git commands ----
    p = Project()
    p.init()
    p.add("decision", "plain memory", "no git here")
    r = p.run("git", "list")
    ok("git: non-git project git list -> 0 results exit 0",
       r.returncode == 0 and "0 results" in r.stdout,
       f"rc={r.returncode} out={r.stdout!r}")
    ok("git: non-git project no gitcontext dir",
       not (p.store() / "gitcontext").exists())
    r = p.run("status", "--json")
    ok("git: non-git status git_contexts 0",
       json.loads(r.stdout)["git_contexts"] == 0,
       f"out={r.stdout!r}")

    # -- real git repo: capture + review + recall -------------------------
    q = Project()
    # make q.dir a real git repo with one commit touching src/auth/login.py
    try:
        subprocess.run(["git", "init", "-q"], cwd=str(q.dir), check=True,
                       capture_output=True)
        subprocess.run(["git", "config", "user.email", "t@t.t"], cwd=str(q.dir),
                       check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "T"], cwd=str(q.dir),
                       check=True, capture_output=True)
        d = q.dir / "src" / "auth"
        d.mkdir(parents=True, exist_ok=True)
        (d / "login.py").write_text("x = 1\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=str(q.dir), check=True,
                       capture_output=True)
        subprocess.run(["git", "commit", "-q", "-m", "auth init"], cwd=str(q.dir),
                       check=True, capture_output=True)
        git_ok = True
    except (subprocess.SubprocessError, OSError):
        git_ok = False
    if not git_ok:
        ok("git: real-repo checks skipped (git unavailable)", True)
        return

    q.init()
    q.add("decision", "auth gate", "all auth through the gate", "--paths", "src/auth/**")
    r = q.run("list", "--json")
    mem_id = json.loads(r.stdout)["results"][0]["id"]
    q.run("promote", mem_id, "--trust", "verified")  # recall excludes untrusted
    r = q.run("git", "context", mem_id, "--json")
    ok("git: context exit 0 + versioned schema",
       r.returncode == 0 and "git-context-v1" in r.stdout,
       f"rc={r.returncode} out={r.stdout!r}")
    snap = json.loads(r.stdout)
    ok("git: changed paths include the committed file",
       "src/auth/login.py" in snap.get("changed_paths", []),
       f"out={snap.get('changed_paths')}")
    ok("git: branch captured", snap.get("branch") in ("master", "main"),
       f"out={snap.get('branch')}")
    r = q.run("git", "list")
    ok("git: list shows the memory", mem_id in r.stdout, f"out={r.stdout!r}")
    r = q.run("status")
    ok("git: status reports 1 git context", "git contexts  : 1" in r.stdout,
       f"out={r.stdout!r}")
    # recall --branch: matches recall; other branch no exclusion (bonus only)
    r = q.run("recall", "gate", "--branch", snap["branch"], "--json")
    ok("git: recall --branch returns the memory",
       r.returncode == 0 and json.loads(r.stdout)["count"] == 1,
       f"out={r.stdout!r}")
    r = q.run("recall", "gate", "--branch", "other", "--json")
    ok("git: other branch no exclusion",
       json.loads(r.stdout)["count"] == 1, f"out={r.stdout!r}")
    # error paths
    r = q.run("git", "context", "mem_nope")
    ok("git: missing context exit 1", r.returncode == 1, f"rc={r.returncode}")
    r = q.run("git", "context")
    ok("git: context without id exit 2", r.returncode == 2, f"rc={r.returncode}")


# ==========================================================================

def main() -> int:
    print("=== agent-memory external-API audit (v0.1) ===")
    audit_init()
    audit_add()
    audit_list_show()
    audit_search()
    audit_recall()
    audit_promote()
    audit_supersede()
    audit_delete()
    audit_status()
    audit_suggestions()
    audit_conflicts()
    audit_git()
    audit_discovery_errors()
    audit_json_errors_and_meta()
    cleanup_all()
    print(f"{COUNT} checks, {len(FAILURES)} failure(s)")
    if FAILURES:
        print("FAILURES:")
        for f in FAILURES:
            print("  - " + f)
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
