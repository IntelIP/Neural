import os
from pathlib import Path

PROD_BASE_URL = "https://api.elections.kalshi.com"

# Project defaults (used only if env vars are not provided)
SECRETS_DIR = Path(__file__).resolve().parents[2] / "secrets"
DEFAULT_API_KEY_PATH = SECRETS_DIR / "kalshi_api_key_id.txt"
DEFAULT_PRIVATE_KEY_PATH = SECRETS_DIR / "kalshi_private_key.pem"

def get_base_url(env: str | None = None) -> str:
	"""Return the production trading API host. Demo endpoints are no longer supported."""
	env_value = (env or os.getenv("KALSHI_ENV", "prod")).lower()
	if env_value in ("prod", "production", "live", ""):  # allow empty for defaults
		return PROD_BASE_URL
	raise ValueError("Kalshi demo environment is unsupported; use production credentials.")

def get_api_key_id() -> str:
	api_key = os.getenv("KALSHI_API_KEY_ID")
	if api_key:
		return api_key
	api_key_path = os.getenv("KALSHI_API_KEY_PATH") or str(DEFAULT_API_KEY_PATH)
	with open(api_key_path, "r", encoding="utf-8") as f:
		return f.read().strip()

def get_private_key_material() -> bytes:
	key_b64 = os.getenv("KALSHI_PRIVATE_KEY_BASE64")
	if key_b64:
		import base64
		return base64.b64decode(key_b64)
	key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
	if not key_path:
		# Fallback to repo secrets path
		key_path = str(DEFAULT_PRIVATE_KEY_PATH)
	with open(key_path, "rb") as f:
		return f.read()
