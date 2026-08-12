---
name: Bug report
about: Report a problem with agent-memory
title: ""
labels: bug
assignees: ""
---

**Describe the bug**
A clear and concise description of what the bug is.

**Steps to reproduce**
```bash
# exact commands
agent-memory init
agent-memory add --type decision --title "T" --content "C"
...
```

**Expected behavior**
What you expected to happen.

**Actual behavior**
What happened, including any output and the exit code. If the output is
JSON, paste it verbatim.

**Environment**
- OS: (Windows / macOS / Linux)
- Python version: (`python --version`)
- agent-memory version: (`agent-memory --version`)

**Additional context**
Anything else that might help — especially whether the issue involves
memory content that was rejected or accepted by secret detection, or a
trust/lifecycle transition.
