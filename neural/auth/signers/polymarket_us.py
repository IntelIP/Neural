from __future__ import annotations

import base64
import time
from collections.abc import Callable
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

TimestampFn = Callable[[], int]


class PolymarketUSSigner:
    """Ed25519 signer for Polymarket US API headers.

    Signature payload format used by this SDK:
    ``<timestamp><METHOD><path><body>``

    The body value should be the exact serialized request body string when present,
    otherwise an empty string.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: bytes,
        passphrase: str,
        now_ms: TimestampFn | None = None,
    ) -> None:
        self.api_key = api_key
        self.passphrase = passphrase
        self._priv = self._load_private_key(api_secret)
        self._now_ms = now_ms or (lambda: int(time.time() * 1000))

    @staticmethod
    def _load_private_key(secret: bytes) -> ed25519.Ed25519PrivateKey:
        # Accept PEM-encoded keys and raw 32-byte seeds.
        if secret.lstrip().startswith(b"-----BEGIN"):
            try:
                key = serialization.load_pem_private_key(secret, password=None)
            except ValueError as exc:
                raise ValueError("Polymarket US private key PEM is invalid") from exc
            if not isinstance(key, ed25519.Ed25519PrivateKey):
                raise ValueError("Polymarket US requires an Ed25519 private key")
            return key

        if len(secret) != 32:
            raise ValueError("Polymarket US secret must be 32 raw bytes or PEM")
        return ed25519.Ed25519PrivateKey.from_private_bytes(secret)

    def headers(self, method: str, path: str, body: str = "") -> dict[str, str]:
        ts = self._now_ms()
        msg = f"{ts}{method.upper()}{path}{body}".encode()
        sig = self._priv.sign(msg)
        return {
            "PM-ACCESS-KEY": self.api_key,
            "PM-ACCESS-TIMESTAMP": str(ts),
            "PM-ACCESS-SIGNATURE": base64.b64encode(sig).decode("utf-8"),
            "PM-ACCESS-PASSPHRASE": self.passphrase,
        }

    @classmethod
    def from_env(
        cls, values: dict[str, Any], now_ms: TimestampFn | None = None
    ) -> PolymarketUSSigner:
        api_key = values.get("api_key")
        api_secret = values.get("api_secret")
        passphrase = values.get("passphrase")
        if api_key is None or api_secret is None or passphrase is None:
            raise ValueError(
                "Missing required Polymarket signer config: api_key, api_secret, passphrase"
            )

        if isinstance(api_secret, str):
            secret_bytes = api_secret.encode("utf-8")
        else:
            secret_bytes = bytes(api_secret)

        return cls(
            api_key=str(api_key),
            api_secret=secret_bytes,
            passphrase=str(passphrase),
            now_ms=now_ms,
        )
