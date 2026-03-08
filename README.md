# Neural SDK

[![PyPI version](https://badge.fury.io/py/neural-sdk.svg)](https://pypi.org/project/neural-sdk/)
[![Python Versions](https://img.shields.io/pypi/pyversions/neural-sdk.svg)](https://pypi.org/project/neural-sdk/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Professional-grade SDK for algorithmic trading on prediction markets, built as the public surface of the Neural stack.

[Documentation](https://neural-sdk.mintlify.app) | [Examples](./examples) | [Contributing](./CONTRIBUTING.md)

## Overview

Neural SDK is the public Python control surface for market access, paper trading, provider discovery, and the CLI bridge consumed by the Neural TUI.

## Install

Using `uv`:

```bash
uv add neural-sdk
uv add "neural-sdk[trading]"
```

Using `pip`:

```bash
pip install neural-sdk
pip install "neural-sdk[trading]"
```

## CLI Bridge

The base install ships a `neural` CLI intended to be the stable machine-readable bridge for the TypeScript Neural TUI.

```bash
neural doctor
neural --json capabilities
neural --json providers list
```

Current bridge commands:
- `doctor`
- `capabilities`
- `providers list`
- `markets list`
- `quote`
- `positions`
- `paper order`
- `deployments list`
- `deployments status`
- `deployments logs`
- `deployments stop`

## Credentials

Create a `.env` file with your Kalshi credentials:

```bash
KALSHI_API_KEY_ID=your_api_key_id
KALSHI_PRIVATE_KEY_BASE64=base64_encoded_private_key
KALSHI_ENV=prod
```

The SDK loads credentials from environment variables or your local secrets files.

## Development

```bash
git clone https://github.com/IntelIP/Neural.git
cd neural
uv sync --extra dev --extra trading --extra sentiment --extra analysis --extra deployment
uv run pytest
uv run ruff check .
uv run black --check .
```

## Testing

```bash
uv run pytest
uv run pytest --cov=neural tests/
```

## Resources

- Documentation: [neural-sdk.mintlify.app](https://neural-sdk.mintlify.app)
- Examples: [examples/](./examples)
- Issues: [GitHub Issues](https://github.com/IntelIP/Neural/issues)
- Discussions: [GitHub Discussions](https://github.com/IntelIP/Neural/discussions)

## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE).
