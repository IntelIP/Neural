#!/usr/bin/env python3
"""
Setup script for Neural SDK

This file enables pip installation of the Neural SDK package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read version from __init__.py
version_file = this_directory / "neural_sdk" / "__init__.py"
version = None
with open(version_file, "r") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split('"')[1]
            break

if not version:
    raise RuntimeError("Version not found in neural_sdk/__init__.py")

# Core dependencies
install_requires = [
    "kalshi-python>=2.0.0",
    "websockets>=11.0",
    "websocket-client>=1.6.0",
    "aiohttp>=3.9.0",
    "pydantic>=2.5.0",
    "python-dotenv>=1.0.0",
    "numpy>=1.24.0",
    "pandas>=2.1.0",
    "scipy>=1.11.0",
    "httpx>=0.25.0",
    "rich>=13.7.0",
    "click>=8.1.7",
    "pyyaml>=6.0.1",
    "pytz>=2023.3",
    "python-dateutil>=2.8.2",
    "redis[asyncio]>=4.6.0",
    "structlog>=23.1.0",
    "typer>=0.9.0",
]

# Optional dependencies
extras_require = {
    "dev": [
        "pytest>=7.0.0",
        "pytest-asyncio>=0.21.0",
        "pytest-cov>=4.0.0",
        "black>=23.0.0",
        "isort>=5.12.0",
        "mypy>=1.0.0",
        "pre-commit>=3.0.0",
    ],
    "docs": [
        "sphinx>=5.0.0",
        "sphinx-rtd-theme>=1.2.0",
        "myst-parser>=0.18.0",
    ],
    "llm": [
        "agno>=0.1.0",
    ],
    "s3": [
        "boto3>=1.26.0",
    ],
    "database": [
        "sqlalchemy>=2.0.0",
        "psycopg2-binary>=2.9.0",
    ],
    "plotting": [
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
    ],
}

# All optional dependencies
extras_require["all"] = sum(extras_require.values(), [])

setup(
    name="neural-sdk",
    version=version,
    author="Neural Team ~ Subsidiary of IntelIP",
    author_email="sdk@neural.dev",
    description="Open-source SDK for algorithmic prediction market trading",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/neural/neural-sdk",
    project_urls={
        "Documentation": "https://neural-sdk.readthedocs.io/",
        "Bug Tracker": "https://github.com/neural/neural-sdk/issues",
        "Source Code": "https://github.com/neural/neural-sdk",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "docs"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.10",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "neural-sdk=neural_sdk.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "neural_sdk": ["*.yaml", "*.json", "*.md"],
    },
    zip_safe=False,
    keywords="trading prediction-markets kalshi algorithmic-trading backtesting websocket",
)