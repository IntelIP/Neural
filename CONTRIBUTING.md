# Contributing to Neural SDK

This document provides the default contributor workflow for Neural SDK.

## Prerequisites

- Python 3.10 or higher
- Git
- `uv`
- A Kalshi API account when you need to exercise live-backed integrations

## Development Setup

```bash
git clone https://github.com/YOUR_USERNAME/Neural.git
cd Neural
git remote add upstream https://github.com/IntelIP/Neural.git
python -m pip install uv
uv sync --extra dev
```

Optional local tooling:

```bash
uv tool install pre-commit
pre-commit install
```

## Branching

Use descriptive branch names:

- `feature/<name>` for new features
- `fix/<name>` for bug fixes
- `docs/<name>` for docs-only changes
- `refactor/<name>` for internal cleanup
- `test/<name>` for test updates

Example:

```bash
git checkout -b feature/add-paper-order-cli
```

## Before Opening a Pull Request

Run the local quality gate from the repo root:

```bash
uv run ruff check .
uv run black --check .
uv run mypy neural
uv run pytest
```

If your change touches docs or examples, validate those paths too:

```bash
uv sync --extra dev --extra docs
uv run python scripts/validate_docs.py
uv run python scripts/validate_examples.py
```

## Pull Request Expectations

- Rebase or merge from `main` before requesting review.
- Update documentation when public behavior changes.
- Add or update tests for functional changes.
- Link the relevant GitHub issue or Linear ticket in the PR body.

## Commit Messages

We use Conventional Commits:

- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation change
- `refactor:` internal refactor
- `test:` test change
- `chore:` maintenance work

Example:

```text
feat(cli): add paper order command
```

## Testing Notes

Useful test commands:

```bash
uv run pytest
uv run pytest tests/test_cli.py
uv run pytest tests/test_cli.py -k doctor
uv run pytest --cov=neural --cov-report=term-missing
```

## Documentation Layout

The SDK documentation lives in `docs/` and uses Mintlify. When you update docs automation or local docs checks, prefer Bun-based tooling for JavaScript CLI dependencies and `uv` for Python scripts.

## Questions

- Documentation: https://neural-sdk.mintlify.app
- Discussions: https://github.com/IntelIP/Neural/discussions
- Email: contributors@neural-sdk.dev
