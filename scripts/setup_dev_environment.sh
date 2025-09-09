#!/bin/bash
# Neural SDK Development Environment Setup
# Version: 1.4.0

echo "🏗️  Setting up Neural SDK development environment..."
echo ""

# Check Python version
python_version=$(python --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python $required_version or higher required. You have $python_version"
    exit 1
fi

echo "✅ Python $python_version detected"

# Install development tools
echo "📦 Installing development tools..."
pip install build twine pytest black isort mypy uv ruff

# Clone repository for development
if [ ! -d "neural-sdk" ]; then
    echo "📥 Cloning Neural SDK repository..."
    git clone https://github.com/neural/neural-sdk.git
fi

cd neural-sdk

# Install in development mode
echo "🔧 Installing Neural SDK in development mode..."
pip install -e .[dev]

# Run tests to verify installation
echo "🧪 Running tests to verify installation..."
pytest tests/unit/test_websocket_simple.py -v

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Development environment ready!"
    echo "🔥 WebSocket streaming tests passed"
    echo ""
    echo "🎯 You can now:"
    echo "1. Edit neural_sdk/ files directly"
    echo "2. Run tests with: pytest"
    echo "3. Build package with: python -m build"
    echo "4. Start coding real-time strategies!"
else
    echo "❌ Tests failed. Please check your setup."
    exit 1
fi