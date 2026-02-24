# Hardening PR Plan

Last updated: 2026-02-24

## Baseline findings

- Quality checks:
  - `ruff check neural tests scripts utils` passes.
  - `mypy neural` passes.
  - `pytest -q` passes (`33 passed, 6 skipped`) after making deployment import optional when Docker SDK is absent.
- Security checks:
  - `pip-audit -r requirements-dev.txt` reports `nltk 3.9.2` with `CVE-2025-14009` (no fixed version currently reported by the tool).
  - `bandit -r neural scripts utils` reports one medium issue (`exec` in docs test helper) plus low-severity findings.
- Repository hygiene:
  - Local `secrets/` directory exists with key-like files (ignored by `.gitignore`, not tracked).

## Proposed PR sequence

1. PR 1: Stabilize baseline checks
- Goal: Make local/CI checks reliable and remove immediate blockers.
- Scope:
  - Keep package importable without optional Docker dependency.
  - Fix current lint failures in deployment code.
  - Add repeatable audit commands to `Makefile`.
- Exit criteria:
  - `ruff check neural tests scripts utils` passes.
  - `mypy neural` passes.
  - `pytest -q` passes in an environment without Docker SDK installed.

2. PR 2: Introduce security gates (non-blocking first)
- Goal: Automate vulnerability and static security reporting.
- Scope:
  - Add a GitHub Actions workflow for `pip-audit` and `bandit`.
  - Publish machine-readable artifacts and PR summaries.
  - Keep advisory mode first (non-blocking) for one sprint.
- Exit criteria:
  - Security report is generated on each PR.
  - Team has clear baseline trend and ownership.

3. PR 3: Dependency risk remediation
- Goal: Reduce known vulnerable dependency surface.
- Scope:
  - Investigate `nltk` usage and replace/remove if not required.
  - If required, pin to a patched version once available and add compensating controls until then.
  - Tighten dependency bounds where practical.
- Exit criteria:
  - `pip-audit` shows zero critical/high issues; documented exception for unresolved upstream CVEs.

4. PR 4: Runtime safety + exception hygiene
- Goal: Remove hidden-failure patterns and improve observability.
- Scope:
  - Replace broad `except Exception: pass` in runtime paths with specific exception handling.
  - Add structured warnings/logging where errors are intentionally suppressed.
  - Keep permissive handling in non-runtime tooling only where justified.
- Exit criteria:
  - No silent runtime exception swallowing in `neural/` modules.

5. PR 5: CI workflow cleanup and reliability
- Goal: Eliminate brittle CI and dead references.
- Scope:
  - Fix YAML indentation and invalid/obsolete script references in docs workflows.
  - Align formatter/linter commands between local Makefile and CI.
  - Add workflow-level dependency caching consistency.
- Exit criteria:
  - CI workflows parse and execute reliably.
  - No failing jobs due to missing scripts or syntax.

6. PR 6: Codebase cleanup and maintainability
- Goal: Reduce complexity and improve long-term velocity.
- Scope:
  - Remove dead code/imports and reduce oversized modules.
  - Standardize module boundaries (especially deployment and docs tooling).
  - Add targeted tests for refactored surfaces.
- Exit criteria:
  - Reduced lint warnings and smaller hot-spot files with equal or better test coverage.

## Commands for each PR branch

```bash
# quality baseline
make audit

# security baseline
make audit-security
make audit-deps
```

## Branch naming convention

Use `codex/<area>-<intent>` (example: `codex/security-baseline-gates`).
