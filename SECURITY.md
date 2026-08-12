# Security Policy

## Reporting a vulnerability

agent-memory stores and serves project knowledge for AI coding agents, so
its security posture matters beyond a typical CLI. Please report suspected
vulnerabilities privately — do **not** open a public issue for security
problems:

- Open a **private advisory**: GitHub → *Security* → *Report a vulnerability*
- Or email the maintainer via the GitHub profile contact info.

You should get an acknowledgement within a few days. Please do not disclose
the issue publicly until it has been addressed.

## Scope & known limitations

- **Secret detection is best-effort, not a guarantee.** The deterministic
  pattern set + entropy heuristic (V0.1_SPEC.md §5) catches common secret
  shapes but is documented as imperfect. `--allow-secret` is an explicit,
  audited override — the override is recorded in the audit log.
- **Trust is human-controlled.** Memories are born `untrusted`; only a human
  can promote trust. Agents can never promote themselves or reach `system`.
  Do not disable this boundary in configuration.
- **Memory poisoning.** Because agents consume recall output, a poisoned or
  misleading memory can steer agent behavior. The defense is the trust
  ladder, immutable history (supersede, never rewrite), and the append-only
  audit log — not a filter on content.
- **The audit log is append-only by contract**, but it is not
  cryptographically tamper-proof; treat it as strong evidence, not proof.
- **Secrets or credentials must never be written into memory content** —
  treat memory files, `audit.jsonl`, and commit messages as public once the
  repo is pushed.

## Supported versions

Security fixes land on `master` and are released per
[SemVer](https://semver.org/). Always use the latest release:
https://github.com/vartiainen1/agent-memory/releases
