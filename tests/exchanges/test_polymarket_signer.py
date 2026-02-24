import base64

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
