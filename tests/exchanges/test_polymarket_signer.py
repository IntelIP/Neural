import base64

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from neural.auth.signers.polymarket_us import PolymarketUSSigner


def test_polymarket_signer_headers_are_deterministic() -> None:
    private = ed25519.Ed25519PrivateKey.generate()
    raw_key = private.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )

    signer = PolymarketUSSigner(
        api_key="key123",
        api_secret=raw_key,
        passphrase="passphrase",
        now_ms=lambda: 1700000000000,
    )

    headers = signer.headers("POST", "/api/v1/orders", body='{"x":1}')

    assert headers["PM-ACCESS-KEY"] == "key123"
    assert headers["PM-ACCESS-TIMESTAMP"] == "1700000000000"
    assert headers["PM-ACCESS-PASSPHRASE"] == "passphrase"

    sig = base64.b64decode(headers["PM-ACCESS-SIGNATURE"])
    assert len(sig) == 64

    headers2 = signer.headers("POST", "/api/v1/orders", body='{"x":1}')
    assert headers == headers2


def test_polymarket_signer_from_env_requires_all_fields() -> None:
    with pytest.raises(ValueError, match="Missing required Polymarket signer config"):
        PolymarketUSSigner.from_env({"api_key": "k", "api_secret": b"1" * 32})


def test_polymarket_signer_invalid_pem_has_clear_error() -> None:
    with pytest.raises(ValueError, match="private key PEM is invalid"):
        PolymarketUSSigner(
            api_key="key123",
            api_secret=b"-----BEGIN PRIVATE KEY-----\nnot-a-key\n-----END PRIVATE KEY-----",
            passphrase="passphrase",
        )


def test_polymarket_signer_from_env_accepts_string_secret() -> None:
    signer = PolymarketUSSigner.from_env(
        {
            "api_key": "key123",
            "api_secret": "a" * 32,
            "passphrase": "passphrase",
        },
        now_ms=lambda: 1700000000000,
    )
    headers = signer.headers("GET", "/api/v1/markets")
    assert headers["PM-ACCESS-KEY"] == "key123"
