from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

try:
	from kalshi_python import ApiInstance, Configuration
except ImportError as exc:
	raise ImportError("kalshi_python>=2 required for neural.trading; install the optional trading extras.") from exc

from neural.auth.env import get_api_key_id, get_private_key_material, get_base_url


def _serialize(obj: Any) -> Any:
	if hasattr(obj, "model_dump"):
		return obj.model_dump()
	if isinstance(obj, dict):
		return {key: _serialize(value) for key, value in obj.items()}
	if isinstance(obj, (list, tuple)):
		type_factory: Callable[[Any], Any] = list if isinstance(obj, list) else tuple
		return type_factory(_serialize(value) for value in obj)
	return obj


class _ServiceProxy:
	def __init__(self, api: Any):
		self._api = api

	def __getattr__(self, name: str) -> Callable[..., Any]:
		attr = getattr(self._api, name)
		if not callable(attr):
			return attr

		def wrapped(*args: Any, **kwargs: Any) -> Any:
			result = attr(*args, **kwargs)
			return _serialize(result)

		wrapped.__doc__ = getattr(attr, "__doc__", None)
		wrapped.__name__ = getattr(attr, "__name__", name)
		return wrapped


@dataclass(slots=True)
class TradingClient:
	"""Composite Kalshi client exposing portfolio, markets, events, communications, and exchange APIs."""

	api_key_id: Optional[str] = None
	private_key_pem: Optional[bytes] = None
	env: Optional[str] = None
	timeout: int = 15
	_client: KalshiClient = field(init=False)
	portfolio: _ServiceProxy = field(init=False)
	markets: _ServiceProxy = field(init=False)
	events: _ServiceProxy = field(init=False)
	communications: _ServiceProxy = field(init=False)
	exchange: _ServiceProxy = field(init=False)
	series: _ServiceProxy = field(init=False)

	def __post_init__(self) -> None:
		api_key = self.api_key_id or get_api_key_id()
		priv_key = self.private_key_pem or get_private_key_material()
		priv_key_material = priv_key.decode("utf-8") if isinstance(priv_key, (bytes, bytearray)) else priv_key

		host = f"{get_base_url(self.env)}/trade-api/v2"
		config = Configuration(host=host)
		config.api_key_id = api_key
		config.private_key_pem = priv_key_material
		config.timeout = self.timeout
		disable_tls = os.getenv("KALSHI_DISABLE_TLS_VERIFY", "").lower() in {"1", "true", "yes", "on"}
		if disable_tls:
			config.verify_ssl = False

		self._client = ApiInstance(config)
		self.portfolio = _ServiceProxy(self._client._portfolio_api)
		self.markets = _ServiceProxy(self._client._markets_api)
		self.events = _ServiceProxy(self._client._events_api)
		self.communications = _ServiceProxy(self._client._communications_api)
		self.exchange = _ServiceProxy(self._client._exchange_api)
		self.series = _ServiceProxy(self._client._series_api)

	def close(self) -> None:
		if hasattr(self._client, "close"):
			self._client.close()

	def __enter__(self) -> "TradingClient":
		return self

	def __exit__(self, exc_type, exc, tb) -> None:
		self.close()
