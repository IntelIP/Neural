# Neural SDK Development Workflow

**Version:** 0.3.0 (Beta)  
**Repository:** https://github.com/IntelIP/Neural  
**Maintainer:** Hudson Aikins, Neural Contributors

---

## 📋 Table of Contents

1. [Branch Strategy](#branch-strategy)
2. [Development Setup](#development-setup)
3. [Creating Features](#creating-features)
4. [Submitting Changes](#submitting-changes)
5. [Release Process](#release-process)
6. [Code Quality Standards](#code-quality-standards)
7. [Troubleshooting](#troubleshooting)

---

## 🌳 Branch Strategy

### **Main Production Branch**

```
main (protected)
  │
  ├─ Always production-ready
  ├─ Tagged with versions (v0.3.0, v0.4.0, etc.)
  ├─ Requires PR review before merge
  └─ CI/CD pipeline runs on all commits
```

### **Development Branches**

#### Feature Branches
```
feature/short-description (from main)
  │
  ├─ Used for: New features, enhancements
  ├─ Naming: feature/nba-markets, feature/historical-data
  ├─ Create: git checkout -b feature/xxx
  ├─ Review: Create PR against main
  └─ Merge: After approval + tests pass
```

#### Bugfix Branches
```
bugfix/issue-number-description (from main)
  │
  ├─ Used for: Bug fixes
  ├─ Naming: bugfix/123-type-errors, bugfix/456-import-fail
  ├─ Create: git checkout -b bugfix/xxx
  ├─ Review: Create PR against main
  └─ Merge: After approval + tests pass
```

#### Release Branches (Optional)
```
release/vX.Y.Z (from main)
  │
  ├─ Used for: Release preparation, hot fixes
  ├─ Naming: release/v0.4.0
  ├─ Create: git checkout -b release/v0.4.0
  ├─ Allowed commits: Version bumps, critical hot fixes only
  └─ Merge: Back to main after release
```

---

## 🚀 Development Setup

### **Local Setup**

```bash
# Clone the repository
git clone https://github.com/IntelIP/Neural.git
cd Neural

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (optional but recommended)
pip install pre-commit
pre-commit install
```

### **Branch Tracking**

```bash
# Create local tracking of remote branches
git fetch origin
git branch -r  # View all remote branches

# Checkout a feature branch from remote
git checkout -b feature/xxx origin/feature/xxx
```

---

## 💻 Creating Features

### **1. Create Feature Branch**

```bash
# Ensure main is up to date
git checkout main
git pull origin main

# Create new feature branch
git checkout -b feature/descriptive-name
```

### **2. Implement Feature**

```bash
# Make your changes
edit file1.py
edit file2.py

# Check status
git status

# Stage changes
git add file1.py file2.py

# Or stage all changes
git add -A
```

### **3. Commit Changes**

```bash
# Commit with descriptive message
git commit -m "feat(module): add descriptive feature title

- Detailed description of what was added
- Why it was added
- Any important notes"
```

**Commit Message Format:**
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation update
- `refactor:` - Code refactoring
- `test:` - Add/update tests
- `chore:` - Maintenance tasks

### **4. Code Quality Checks**

```bash
# Run linting
python -m ruff check .

# Auto-fix linting issues
python -m ruff check . --fix

# Format code
python -m black .

# Type checking
python -m mypy neural/

# Run tests
python -m pytest -v

# Check coverage
python -m pytest --cov=neural --cov-report=term-missing
```

### **5. Push to Remote**

```bash
# Push feature branch
git push origin feature/descriptive-name

# Set upstream tracking (first time)
git push -u origin feature/descriptive-name
```

---

## 📤 Submitting Changes

### **Create Pull Request**

```bash
# From GitHub.com or using gh CLI:
gh pr create --title "feat: descriptive title" \
             --body "Description of changes..."
```

### **PR Template**

```markdown
## Summary
Brief description of what this PR does.

## Changes
- Change 1
- Change 2
- Change 3

## Testing
How was this tested?
- [ ] Unit tests added
- [ ] Integration tests added
- [ ] Manual testing (describe)

## Related Issues
Fixes #123
Relates to #456

## Checklist
- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] No breaking changes
```

### **Code Review Process**

1. **Create PR** against `main`
2. **Wait for CI/CD** - All checks must pass
3. **Request review** from maintainers
4. **Address feedback** - Make requested changes
5. **Approve & Merge** - Squash or rebase as needed

### **Branch Protection Rules**

On `main` branch:
- ✅ Require PR review before merge
- ✅ Dismiss stale PR approvals
- ✅ Require status checks to pass
- ✅ Require branches to be up to date

---

## 🏷️ Release Process

### **Preparing a Release**

```bash
# 1. Create release branch from main
git checkout -b release/v0.4.0

# 2. Update version numbers
# - Update pyproject.toml version = "0.4.0"
# - Update neural/__init__.py __version__ = "0.4.0"
# - Update .bumpversion.cfg current_version = 0.4.0

# 3. Update CHANGELOG.md
# - Add new version section
# - Document all changes

# 4. Commit changes
git commit -m "chore(release): prepare v0.4.0 release"

# 5. Create PR for review
gh pr create --title "release: v0.4.0" \
             --body "Release preparation PR"
```

### **Publishing Release**

```bash
# After PR merged to main, create tag
git checkout main
git pull origin main

# Create annotated tag
git tag -a v0.4.0 -m "Release v0.4.0"

# Push tag (triggers PyPI publish workflow)
git push origin v0.4.0
```

### **Verify Release**

```bash
# Check PyPI
pip install neural-sdk==0.4.0

# Verify version
python -c "import neural; print(neural.__version__)"

# Should output: 0.4.0
```

---

## 📊 Code Quality Standards

### **Linting (Ruff)**

```bash
# Check
python -m ruff check neural/

# Fix automatically
python -m ruff check neural/ --fix
```

**Rules:**
- Line length: 100 characters
- Ignore: E501 (long lines handled by black)

### **Formatting (Black)**

```bash
# Format all code
python -m black .
```

**Standards:**
- Line length: 100 characters
- Target Python: 3.10+

### **Type Checking (MyPy)**

```bash
# Run type checker
python -m mypy neural/

# Configurations in pyproject.toml
```

**Standards:**
- Warn on missing types: true
- Check untyped defs: true
- No implicit optional: true

### **Testing (Pytest)**

```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_v030_features.py

# Run with coverage
python -m pytest --cov=neural

# Run specific test
python -m pytest tests/test_v030_features.py::TestHistoricalCandlesticks::test_fetch_historical_candlesticks_basic
```

**Standards:**
- All tests must pass
- Coverage should be ≥40%
- Use pytest fixtures for setup/teardown
- Mock external dependencies

### **Minimum Quality Gate**

Before submitting PR:

```bash
# Run all checks
ruff check . --fix
black .
mypy neural/
pytest -v --cov=neural

# All must pass before PR submission
```

---

## 🔗 Git Workflow Examples

### **Adding a Feature**

```bash
# 1. Start from clean main
git checkout main
git pull origin main

# 2. Create feature branch
git checkout -b feature/add-backtesting-viz

# 3. Make changes and test
edit neural/analysis/backtesting/engine.py
python -m pytest tests/

# 4. Commit with descriptive message
git commit -m "feat(backtesting): add plotly visualization

- Add plot_results() method to Backtester
- Support multiple metrics visualization
- Include confidence intervals"

# 5. Push and create PR
git push -u origin feature/add-backtesting-viz
gh pr create
```

### **Fixing a Bug**

```bash
# 1. Create bugfix branch
git checkout -b bugfix/123-signal-type-error

# 2. Make fix
edit neural/analysis/strategies/base.py

# 3. Add test for fix
edit tests/test_v030_features.py

# 4. Verify fix
python -m pytest tests/test_v030_features.py -v

# 5. Commit and push
git commit -m "fix(strategies): resolve Signal type constructor issue

Fixes #123

- Use signal_type, market_id, recommended_size params
- Maintain backward compatibility with properties
- Add tests for all constructor variants"

git push -u origin bugfix/123-signal-type-error
```

### **Syncing with Main**

```bash
# If main has new commits while you're working
git fetch origin
git rebase origin/main  # or merge
git push origin feature/xxx --force-with-lease  # only if rebased
```

---

## 🛠️ Troubleshooting

### **Can't Push - Branch Not Updated**

```bash
# Solution: Fetch and merge latest main
git fetch origin
git merge origin/main

# Then retry push
git push origin feature/xxx
```

### **Accidentally Committed to Main**

```bash
# Move last commit to new branch
git branch feature/oops
git reset --hard HEAD~1
git checkout feature/oops

# Or just create PR from main if accidental commit is good
```

### **Need to Undo Changes**

```bash
# Undo uncommitted changes
git checkout file.py

# Undo last commit (keep changes staged)
git reset --soft HEAD~1

# Undo last commit (discard changes)
git reset --hard HEAD~1
```

### **Merge Conflicts**

```bash
# When pulling or merging
git status  # See conflicts

# Edit conflicted files, then:
git add file.py
git commit -m "resolve: merge conflict"
git push origin feature/xxx
```

---

## 📝 Commit Message Guidelines

### **Good Examples**

```
feat(data_collection): add NBA market discovery with team parsing

- Implement get_nba_games() with automatic team extraction
- Add date parameter for filtering games
- Handle playoff/regular season markets
- Includes comprehensive tests

Closes #234
```

```
fix(order_manager): correct float to int conversion in order placement

- Convert signal.size from fraction to contract count
- Add null safety check for entry_price
- Prevents type errors in order execution

Fixes #567
```

### **Poor Examples**

```
updated code
fixed stuff
work in progress
todo
```

---

## 🚀 Quick Reference

```bash
# Clone and setup
git clone https://github.com/IntelIP/Neural.git && cd Neural
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"

# Create feature
git checkout -b feature/new-feature && git pull origin main

# Before PR
ruff check . --fix && black . && mypy neural && pytest

# Push and PR
git push -u origin feature/new-feature && gh pr create

# Sync with main
git fetch origin && git rebase origin/main

# Update version for release
# 1. Update pyproject.toml, neural/__init__.py, .bumpversion.cfg
# 2. Update CHANGELOG.md
# 3. git commit -m "chore(release): vX.Y.Z"
# 4. git tag -a vX.Y.Z -m "Release vX.Y.Z"
# 5. git push origin vX.Y.Z
```

---

## 📞 Questions?

- Check [BRANCH_ANALYSIS.md](BRANCH_ANALYSIS.md) for branch history
- Review [CHANGELOG.md](CHANGELOG.md) for version history
- Open an issue on [GitHub](https://github.com/IntelIP/Neural/issues)

Happy coding! 🚀

