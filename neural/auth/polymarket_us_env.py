from __future__ import annotations

import base64
import binascii
import os
from pathlib import Path

POLYMARKET_US_API_URL = "https://api.polymarket.us"

SECRETS_DIR = Path(__file__).resolve().parents[2] / "secrets"
DEFAULT_API_KEY_PATH = SECRETS_DIR / "polymarket_us_api_key.txt"
DEFAULT_API_SECRET_PATH = SECRETS_DIR / "polymarket_us_api_secret.bin"
DEFAULT_PASSPHRASE_PATH = SECRETS_DIR / "polymarket_us_api_passphrase.txt"


def _read_text_file(path: str | Path, label: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{label} not found. Set env var or provide a file at {path}."
        ) from None


def _read_bytes_file(path: str | Path, label: str) -> bytes:
    try:
        return Path(path).read_bytes()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{label} not found. Set env var or provide a file at {path}."
        ) from None


def get_polymarket_us_base_url() -> str:
    return os.getenv("POLYMARKET_US_API_URL", POLYMARKET_US_API_URL).rstrip("/")


def get_polymarket_us_api_key() -> str:
    value = os.getenv("POLYMARKET_US_API_KEY")
    if value:
        return value
    path = os.getenv("POLYMARKET_US_API_KEY_PATH", str(DEFAULT_API_KEY_PATH))
    return _read_text_file(path, "Polymarket US API key")


def get_polymarket_us_passphrase() -> str:
    value = os.getenv("POLYMARKET_US_API_PASSPHRASE")
    if value:
        return value
    path = os.getenv("POLYMARKET_US_API_PASSPHRASE_PATH", str(DEFAULT_PASSPHRASE_PATH))
    return _read_text_file(path, "Polymarket US API passphrase")


def get_polymarket_us_api_secret() -> bytes:
    b64_value = os.getenv("POLYMARKET_US_API_SECRET_BASE64")
    if b64_value:
        return base64.b64decode(b64_value)

    raw_value = os.getenv("POLYMARKET_US_API_SECRET")
    if raw_value:
        if raw_value.lstrip().startswith("-----BEGIN"):
            return raw_value.encode("utf-8")
        # POLYMARKET_US_API_SECRET should be base64 unless PEM is explicitly provided.
        try:
            return base64.b64decode(raw_value, validate=True)
        except binascii.Error as exc:
            raise ValueError(
                "POLYMARKET_US_API_SECRET must be valid base64 or PEM-encoded key data"
            ) from exc

    path = os.getenv("POLYMARKET_US_API_SECRET_PATH", str(DEFAULT_API_SECRET_PATH))
    return _read_bytes_file(path, "Polymarket US API secret")


def get_polymarket_us_credentials() -> dict[str, object]:
    return {
        "api_key": get_polymarket_us_api_key(),
        "api_secret": get_polymarket_us_api_secret(),
        "passphrase": get_polymarket_us_passphrase(),
    }
