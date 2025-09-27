import os
from pathlib import Path
from typing import Optional

PROD_BASE_URL = "https://api.elections.kalshi.com"

# Project defaults (used only if env vars are not provided)
SECRETS_DIR = Path(__file__).resolve().parents[2] / "secrets"
DEFAULT_API_KEY_PATH = SECRETS_DIR / "kalshi_api_key_id.txt"
DEFAULT_PRIVATE_KEY_PATH = SECRETS_DIR / "kalshi_private_key.pem"


def get_base_url(env: Optional[str] = None) -> str:
    """Return the trading API host (production by default).
    Demo endpoints are not supported; raise if a non-prod env is requested.
    """
    env_value = (env or os.getenv("KALSHI_ENV", "prod")).lower()
    if env_value in ("prod", "production", "live", ""):  # allow empty for defaults
        return PROD_BASE_URL
    raise ValueError("Kalshi demo environment is unsupported; use production credentials.")


def get_api_key_id() -> str:
    """Prefer env var KALSHI_API_KEY_ID; fall back to a local file if present."""
    api_key = os.getenv("KALSHI_API_KEY_ID")
    if api_key:
        return api_key
    api_key_path = os.getenv("KALSHI_API_KEY_PATH") or str(DEFAULT_API_KEY_PATH)
    try:
        with open(api_key_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        # Provide a clearer error guiding users to set env vars in CI
        raise FileNotFoundError(
            f"Kalshi API key not found. Set KALSHI_API_KEY_ID or provide a file at {api_key_path}."
        )


def get_private_key_material() -> bytes:
    """Prefer base64 env var KALSHI_PRIVATE_KEY_BASE64; fall back to file path.
    Raises a helpful error if neither is available.
    """
    key_b64 = os.getenv("KALSHI_PRIVATE_KEY_BASE64")
    if key_b64:
        import base64

        return base64.b64decode(key_b64)
    key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH") or str(DEFAULT_PRIVATE_KEY_PATH)
    try:
        with open(key_path, "rb") as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Kalshi private key not found. Set KALSHI_PRIVATE_KEY_BASE64 or provide a file at {key_path}."
        )
