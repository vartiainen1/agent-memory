# Contributing

Thanks for considering a contribution. This is a small, dependency-free
project — please keep it that way.

## Ground rules

- **Stdlib only** — no new runtime dependencies unless truly unavoidable.
- **Python >= 3.11** — the stdlib `tomllib` requirement is deliberate.
- **Plain and portable** — must work on Windows, macOS, and Linux; memory
  files and CLI output must stay byte-identical across platforms.
- **Small changes** — prefer minimal diffs; no gold-plating.
- **Spec is the contract** — `V0.1_SPEC.md` pins the schema, enums, CLI,
  exit codes, and semantics. If implementation reveals a genuine
  contradiction, change the spec deliberately and record the decision in
  the changelog — don't let code quietly drift.

## Reporting bugs

Open an issue with:

- what you ran (exact commands),
- what happened (output),
- what you expected,
- your OS and Python version.

Security issues: see `SECURITY.md` — report privately, never in a public issue.

## Submitting changes

1. Fork the repo and create a feature branch.
2. Make your change. Add or update tests in `_test_agent_memory.py` for any
   behavior change.
3. Run the checks — all three must pass:
   ```sh
   python _test_agent_memory.py   # unit + CLI integration tests
   python _audit_cli.py           # external-API audit (real binary)
   python _check_drift.py         # README/SPEC/pyproject drift guards
   ```
4. If the README test count changes, update it in the SAME commit — the
   drift guard will block the push otherwise.
5. Commit and open a pull request.

## Design notes

- `_audit_cli.py` treats the CLI as an external API (subprocess only) —
  a change that passes unit tests but breaks the CLI contract will fail
  there.
- Trust transitions are human-only by design: `promote` never reaches
  `system`, and there are no downgrades in v0.1. Don't loosen this without
  a spec change.
