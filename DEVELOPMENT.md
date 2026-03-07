# Neural SDK Development Workflow

**Repository:** https://github.com/IntelIP/Neural

## Development Setup

```bash
git clone https://github.com/IntelIP/Neural.git
cd Neural
python -m pip install uv
uv sync --extra dev
```

Optional tooling:

```bash
uv tool install pre-commit
pre-commit install
```

## Daily Workflow

```bash
git checkout main
git pull origin main
git checkout -b feature/descriptive-name
```

Implement your changes, then run the local checks:

```bash
uv run ruff check .
uv run ruff check . --fix
uv run black .
uv run mypy neural/
uv run pytest -v
uv run pytest --cov=neural --cov-report=term-missing
```

## Pull Requests

Create a pull request against `main` after:

1. Syncing with the latest `main`
2. Running lint, typecheck, and tests locally
3. Updating docs for public behavior changes
4. Adding tests for new logic or bug fixes

Example:

```bash
gh pr create --title "feat: descriptive title" --body "Summary of changes"
```

## Release Preparation

```bash
git checkout -b release/vX.Y.Z
# update version metadata
git commit -m "chore(release): prepare vX.Y.Z release"
gh pr create --title "release: vX.Y.Z" --body "Release preparation"
```

After merge:

```bash
git checkout main
git pull origin main
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin vX.Y.Z
```

To verify a published package in a clean environment:

```bash
uv venv .release-venv
.release-venv\Scripts\python -m pip install neural-sdk==X.Y.Z
.release-venv\Scripts\python -c "import neural; print(neural.__version__)"
```

## Quality Standards

- Ruff for linting
- Black for formatting
- MyPy for typing
- Pytest for tests
- `uv` as the default Python workflow manager

## Quick Reference

```bash
python -m pip install uv
uv sync --extra dev
uv run ruff check . --fix
uv run black .
uv run mypy neural
uv run pytest
```
