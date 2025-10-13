import os


def load_kalshi_credentials(
    api_key_path: str = "secrets/kalshi_api_key_id.txt",
    private_key_path: str = "secrets/kalshi_private_key.pem",
) -> tuple[str, bytes]:
    """
    Load Kalshi API credentials from the secrets directory and export env vars
    for components that rely on environment configuration.

    Returns a tuple of (api_key_id, private_key_pem_bytes).
    """
    with open(api_key_path) as f:
        api_key_id = f.read().strip()
    with open(private_key_path, "rb") as f:
        private_key_pem = f.read()

    os.environ["KALSHI_API_KEY"] = api_key_id
    os.environ["KALSHI_PRIVATE_KEY_PATH"] = os.path.abspath(private_key_path)

    return api_key_id, private_key_pem
