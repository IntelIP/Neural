# Contributing to Kalshi Trading Agent System

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)

## Code of Conduct

### Our Standards
- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive criticism
- Accept feedback gracefully
- Prioritize the project's best interests

### Unacceptable Behavior
- Harassment or discrimination
- Personal attacks
- Trolling or inflammatory comments
- Publishing private information
- Any unprofessional conduct

## Getting Started

### Prerequisites
- Python 3.10 or higher
- UV package manager (0.5.25+)
- Redis server
- Git
- GitHub account

### First Time Contributors
1. Look for issues labeled `good first issue` or `help wanted`
2. Comment on the issue to claim it
3. Ask questions if anything is unclear
4. Submit your PR when ready

## Development Setup

### 1. Fork and Clone
```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/Kalshi_Agentic_Agent.git
cd Kalshi_Agentic_Agent
git remote add upstream https://github.com/ORIGINAL_OWNER/Kalshi_Agentic_Agent.git
```

### 2. Create Development Environment
```bash
# Install UV if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync

# Install development dependencies
uv pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### 3. Configure Environment
```bash
# Copy example environment file
cp .env.example .env.development

# Edit with your configuration
vim .env.development
```

### 4. Start Redis
```bash
# Using Docker
docker run -d -p 6379:6379 redis:alpine

# Or locally
redis-server
```

### 5. Verify Setup
```bash
# Run tests
uv run pytest tests/

# Start development server
uv run server.py
```

## How to Contribute

### Reporting Bugs
1. Check existing issues to avoid duplicates
2. Use the bug report template
3. Include:
   - Clear description
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details
   - Error messages/logs

### Suggesting Features
1. Check existing feature requests
2. Use the feature request template
3. Explain:
   - Problem being solved
   - Proposed solution
   - Alternative approaches
   - Use cases

### Contributing Code

#### 1. Choose an Issue
- Pick an unassigned issue
- Comment to claim it
- Ask for clarification if needed

#### 2. Create Branch
```bash
# Update your fork
git checkout develop
git pull upstream develop

# Create feature branch
git checkout -b feature/your-feature-name
```

#### 3. Make Changes
- Write clean, documented code
- Follow coding standards
- Add tests for new functionality
- Update documentation

#### 4. Test Your Changes
```bash
# Run all tests
uv run pytest tests/

# Run specific test file
uv run pytest tests/test_specific.py

# Check code style
uv run ruff check .
uv run black --check .

# Type checking
uv run mypy .
```

#### 5. Commit Your Changes
```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat(component): add new feature

- Detailed description
- List of changes
- Fixes #123"
```

## Coding Standards

### Python Style Guide
We follow PEP 8 with these additions:
- Line length: 100 characters
- Use type hints
- Docstrings for all public functions
- Async/await for I/O operations

### Code Structure
```python
"""
Module docstring describing purpose.
"""

import standard_library
import third_party
from local_module import Component

# Constants
DEFAULT_TIMEOUT = 30

class ExampleClass:
    """
    Class docstring with description.
    
    Attributes:
        name: Description of attribute
        value: Another attribute
    """
    
    def __init__(self, name: str, value: int) -> None:
        """Initialize the class."""
        self.name = name
        self.value = value
    
    async def process_data(self, data: Dict[str, Any]) -> bool:
        """
        Process incoming data.
        
        Args:
            data: Input data dictionary
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            ValueError: If data is invalid
        """
        try:
            # Implementation
            return True
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return False
```

### Import Organization
```python
# Standard library
import os
import sys
from datetime import datetime

# Third-party
import numpy as np
import pandas as pd
from redis import Redis

# Local application
from data_pipeline import StreamManager
from agent_consumers import BaseConsumer
from trading_logic import KellyCriterion
```

### Naming Conventions
- **Files/Modules**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions/Variables**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private**: `_leading_underscore`

## Testing Guidelines

### Test Structure
```
tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
├── fixtures/       # Test fixtures
└── conftest.py     # Pytest configuration
```

### Writing Tests
```python
import pytest
from unittest.mock import Mock, patch

class TestKellyCriterion:
    """Test Kelly Criterion calculations."""
    
    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return KellyCriterion(kelly_fraction=0.25)
    
    def test_calculate_position_size(self, calculator):
        """Test position size calculation."""
        # Arrange
        confidence = 0.65
        odds = 1.5
        bankroll = 10000
        
        # Act
        position = calculator.calculate(confidence, odds, bankroll)
        
        # Assert
        assert position > 0
        assert position <= bankroll * 0.25
    
    @pytest.mark.asyncio
    async def test_async_operation(self):
        """Test async operations."""
        result = await async_function()
        assert result is not None
```

### Test Coverage
- Aim for >80% coverage
- Test edge cases
- Test error conditions
- Mock external dependencies

## Documentation

### Docstring Format
```python
def calculate_position(
    confidence: float,
    odds: float,
    bankroll: float,
    kelly_fraction: float = 0.25
) -> Dict[str, float]:
    """
    Calculate optimal position size using Kelly Criterion.
    
    Uses the Kelly formula with a safety factor to determine
    the optimal bet size for a given opportunity.
    
    Args:
        confidence: Probability of winning (0-1)
        odds: Decimal odds offered
        bankroll: Total available capital
        kelly_fraction: Safety factor (default 0.25)
        
    Returns:
        Dictionary containing:
            - position_size: Recommended bet amount
            - kelly_percentage: Full Kelly percentage
            - risk_amount: Amount at risk
            
    Raises:
        ValueError: If confidence not in [0, 1]
        ValueError: If odds <= 0
        ValueError: If bankroll <= 0
        
    Example:
        >>> result = calculate_position(0.6, 1.5, 10000)
        >>> print(f"Bet ${result['position_size']}")
        
    Note:
        Never uses full Kelly for safety. The fraction
        parameter should typically be between 0.1 and 0.5.
    """
```

### README Updates
- Update README.md for new features
- Include usage examples
- Document configuration changes
- Add troubleshooting tips

### Architecture Documentation
- Update CLAUDE.md for architectural changes
- Document design decisions
- Explain data flow changes
- Update component diagrams

## Submitting Changes

### Pull Request Process

#### 1. Prepare Your PR
```bash
# Ensure your branch is up to date
git checkout develop
git pull upstream develop
git checkout feature/your-feature
git rebase develop

# Push to your fork
git push origin feature/your-feature
```

#### 2. Create Pull Request
- Use the PR template
- Link related issues
- Add clear description
- Include test evidence
- Request reviewers

#### 3. PR Title Format
```
[TYPE] Brief description (#issue)
```

Examples:
- `[FEATURE] Add WebSocket reconnection logic (#123)`
- `[BUGFIX] Fix Redis timeout handling (#456)`
- `[DOCS] Update installation guide (#789)`

#### 4. PR Checklist
- [ ] Tests pass
- [ ] Documentation updated
- [ ] Code follows style guide
- [ ] No secrets committed
- [ ] Changelog updated (for features)

### Code Review Process

#### For Authors
- Respond to feedback promptly
- Make requested changes
- Update PR description if scope changes
- Request re-review after changes

#### For Reviewers
- Be constructive and respectful
- Explain reasoning for changes
- Suggest specific improvements
- Approve when satisfied

### After Merge
- Delete your feature branch
- Update your local repository
- Close related issues
- Update project board

## Release Process

### Version Numbering
We use Semantic Versioning (MAJOR.MINOR.PATCH):
- MAJOR: Breaking changes
- MINOR: New features
- PATCH: Bug fixes

### Release Steps
1. Create release branch from develop
2. Update version numbers
3. Update CHANGELOG.md
4. Create PR to main
5. After merge, tag release
6. Create GitHub release

## Getting Help

### Resources
- [Documentation](docs/)
- [Architecture Guide](CLAUDE.md)
- [Git Workflow](docs/GIT_WORKFLOW.md)
- [API Documentation](docs/API.md)

### Communication Channels
- GitHub Issues: Bug reports and features
- GitHub Discussions: General questions
- Pull Requests: Code contributions

### Asking Questions
When asking questions:
1. Search existing issues/discussions
2. Provide context and examples
3. Include error messages
4. Describe what you've tried

## Recognition

### Contributors
All contributors are recognized in:
- GitHub contributors page
- Release notes
- CONTRIBUTORS.md file

### Types of Contributions
We value all contributions:
- Code improvements
- Bug reports
- Documentation updates
- Testing additions
- Design suggestions
- Community support

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

---

Thank you for contributing to the Kalshi Trading Agent System! Your efforts help make this project better for everyone.