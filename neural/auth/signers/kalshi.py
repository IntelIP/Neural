import base64, time
from typing import Callable, Protocol, Mapping
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

class Signer(Protocol):
	def headers(self, method: str, path: str) -> Mapping[str, str]: ...

TimestampFn = Callable[[], int]

class KalshiSigner:
	def __init__(self, api_key_id: str, private_key_pem: bytes, now_ms: TimestampFn | None = None):
		self.api_key_id = api_key_id
		self._priv = self._load_private_key(private_key_pem)
		self._now_ms = now_ms or (lambda: int(time.time() * 1000))

	@staticmethod
	def _load_private_key(pem: bytes) -> rsa.RSAPrivateKey:
		return serialization.load_pem_private_key(pem, password=None)

	def headers(self, method: str, path: str) -> dict[str, str]:
		ts = self._now_ms()
		msg = f"{ts}{method.upper()}{path}".encode("utf-8")
		sig = self._priv.sign(
			msg,
			padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
			hashes.SHA256(),
		)
		return {
			"KALSHI-ACCESS-KEY": self.api_key_id,
			"KALSHI-ACCESS-TIMESTAMP": str(ts),
			"KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode("utf-8"),
		}


