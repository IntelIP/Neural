# Contributing to Neural SDK

Thank you for your interest in contributing to Neural SDK! This document provides guidelines and instructions for contributing.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Pull Request Process](#pull-request-process)
- [Code Standards](#code-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)

## 🤝 Code of Conduct

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to contributors@neural-sdk.dev.

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- A Kalshi API account (for testing)

### Development Setup

1. **Fork the repository** on GitHub

2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/neural.git
   cd neural
   ```

3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/IntelIP/Neural.git
   ```

4. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

5. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

6. **Set up pre-commit hooks** (optional but recommended):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## 🔨 Making Changes

### Branch Naming Convention

Create a descriptive branch for your work:

- `feature/your-feature-name` - New features
- `fix/bug-description` - Bug fixes
- `docs/what-you-document` - Documentation updates
- `refactor/what-you-refactor` - Code refactoring
- `test/what-you-test` - Adding or updating tests

Example:
```bash
git checkout -b feature/add-sentiment-strategy
```

### Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(strategies): Add sentiment analysis strategy

Implement sentiment-based trading strategy using NLP
on market descriptions and news feeds.

Closes #42
```

```
fix(auth): Handle expired token refresh correctly

Previously, expired tokens would cause silent failures.
Now properly catches and refreshes expired tokens.

Fixes #123
```

### Keeping Your Fork Updated

Before making changes, sync with upstream:

```bash
git fetch upstream
git checkout main
git merge upstream/main
git push origin main
```

## 🎯 Pull Request Process

### Before Submitting

1. **Update your branch** with latest main:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run tests**:
   ```bash
   pytest
   ```

3. **Run linting**:
   ```bash
   ruff check .
   black --check .
   ```

4. **Run type checking**:
   ```bash
   mypy neural
   ```

5. **Update documentation** if needed

### Submitting Pull Request

1. **Push your branch**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create Pull Request** on GitHub

3. **Fill out PR template** completely

4. **Link related issues** using keywords:
   - `Fixes #123`
   - `Closes #456`
   - `Resolves #789`

### PR Review Process

- Maintainers will review within 48 hours
- Address review feedback by pushing new commits
- Once approved, maintainers will merge
- Don't force push after review starts

## 📝 Code Standards

### Python Style

We follow [PEP 8](https://pep8.org/) with some modifications:

- Line length: 100 characters (not 79)
- Use `black` for formatting
- Use `ruff` for linting
- Use type hints where possible

### Code Organization

```python
"""Module docstring describing purpose."""

import standard_library
import third_party
import neural

from neural.module import SpecificClass


class MyClass:
    """Class docstring."""

    def __init__(self, param: str) -> None:
        """Initialize with clear parameter descriptions."""
        self.param = param

    def method(self, arg: int) -> bool:
        """Method docstring with Args and Returns."""
        pass
```

### Documentation Strings

Use Google-style docstrings:

```python
def calculate_position_size(
    capital: float,
    edge: float,
    kelly_fraction: float = 0.25
) -> int:
    """Calculate position size using Kelly Criterion.

    Args:
        capital: Available trading capital in dollars
        edge: Expected edge (win_rate - loss_rate)
        kelly_fraction: Fraction of Kelly to use (default: 0.25 for quarter Kelly)

    Returns:
        Position size in number of contracts

    Raises:
        ValueError: If capital is negative or kelly_fraction > 1

    Example:
        >>> calculate_position_size(10000, 0.05, 0.25)
        125
    """
    if capital < 0:
        raise ValueError("Capital must be positive")

    return int(capital * edge * kelly_fraction)
```

## 🧪 Testing Guidelines

### Writing Tests

- Place tests in `tests/` directory
- Mirror source structure: `neural/auth/client.py` → `tests/auth/test_client.py`
- Use descriptive test names: `test_kelly_criterion_with_negative_capital_raises_error`

### Test Structure

```python
import pytest
from neural.risk import calculate_position_size


class TestPositionSizing:
    """Tests for position sizing functions."""

    def test_kelly_criterion_basic(self):
        """Test Kelly Criterion with standard inputs."""
        result = calculate_position_size(10000, 0.05, 0.25)
        assert result == 125

    def test_kelly_criterion_negative_capital(self):
        """Test that negative capital raises ValueError."""
        with pytest.raises(ValueError, match="Capital must be positive"):
            calculate_position_size(-1000, 0.05, 0.25)

    @pytest.mark.parametrize("capital,edge,expected", [
        (10000, 0.05, 125),
        (5000, 0.10, 125),
        (20000, 0.025, 125),
    ])
    def test_kelly_criterion_various_inputs(self, capital, edge, expected):
        """Test Kelly Criterion with various input combinations."""
        assert calculate_position_size(capital, edge, 0.25) == expected
```

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/auth/test_client.py

# Specific test
pytest tests/auth/test_client.py::TestAuthClient::test_login

# With coverage
pytest --cov=neural --cov-report=html

# Verbose output
pytest -v

# Stop on first failure
pytest -x
```

## 📚 Documentation

### Adding Documentation

Documentation lives in `docs/` and uses Mintlify:

```bash
docs/
├── mint.json          # Configuration
├── introduction.mdx   # Getting started
├── authentication/    # Auth guides
├── strategies/        # Strategy docs
└── api-reference/     # API docs
```

### Documentation Standards

1. **Clear and concise** - Get to the point quickly
2. **Include examples** - Show, don't just tell
3. **Keep updated** - Update docs with code changes
4. **Add links** - Cross-reference related content

### Example Documentation

```markdown
---
title: "Kelly Criterion Position Sizing"
description: "Calculate optimal position sizes using Kelly Criterion"
---

## Overview

The Kelly Criterion determines optimal position size based on edge and capital.

## Usage

```python
from neural.risk import calculate_position_size

# Calculate position size
size = calculate_position_size(
    capital=10000,
    edge=0.05,
    kelly_fraction=0.25  # Quarter Kelly for safety
)

print(f"Suggested position: {size} contracts")
```

## Parameters

- `capital` (float): Available trading capital
- `edge` (float): Expected edge (0.05 = 5% edge)
- `kelly_fraction` (float): Fraction of Kelly (0.25 = conservative)

## Best Practices

1. Use quarter Kelly (0.25) or less for safety
2. Recalculate position size as capital changes
3. Consider correlation across positions
```

## 🐛 Reporting Bugs

### Before Reporting

1. **Search existing issues** - Your bug may already be reported
2. **Update to latest version** - Bug might be fixed
3. **Verify it's a bug** - Not a usage question

### Bug Report Template

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce:
1. Initialize client with '...'
2. Call method '....'
3. See error

**Expected behavior**
What you expected to happen.

**Actual behavior**
What actually happened.

**Code sample**
\```python
# Minimal code to reproduce
from neural import ...
\```

**Environment**
- OS: [e.g., macOS 13.0, Ubuntu 22.04]
- Python version: [e.g., 3.11.4]
- Neural SDK version: [e.g., 0.1.0]

**Additional context**
Any other relevant information.
```

## 💡 Feature Requests

We welcome feature ideas! Open an issue with:

1. **Use case** - What problem does this solve?
2. **Proposed solution** - How should it work?
3. **Alternatives** - What other solutions did you consider?
4. **Additional context** - Anything else we should know?

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Recognition

Contributors are recognized in:
- GitHub contributors page
- Release notes
- Project README (for significant contributions)

## 📞 Questions?

- **Documentation**: https://neural-sdk.mintlify.app
- **Discussions**: https://github.com/IntelIP/Neural/discussions
- **Email**: contributors@neural-sdk.dev

---

Thank you for contributing to Neural SDK! 🚀