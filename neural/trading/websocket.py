from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlparse, urlunparse

try:
	import websocket
except ImportError as exc:
	raise ImportError("websocket-client is required for Neural Kalshi WebSocket support.") from exc

from neural.auth.env import get_api_key_id, get_private_key_material, get_base_url
from neural.auth.signers.kalshi import KalshiSigner

_LOG = logging.getLogger(__name__)


@dataclass
class KalshiWebSocketClient:
	"""Thin wrapper over the Kalshi WebSocket RPC channel."""

	signer: KalshiSigner | None = None
	api_key_id: Optional[str] = None
	private_key_pem: Optional[bytes] = None
	env: Optional[str] = None
	url: Optional[str] = None
	path: str = "/trade-api/ws/v2"
	on_message: Optional[Callable[[Dict[str, Any]], None]] = None
	on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None
	sslopt: Optional[Dict[str, Any]] = None
	ping_interval: float = 25.0
	ping_timeout: float = 10.0
	_connect_timeout: float = 10.0
	_request_id: int = field(init=False, default=1)

	def __post_init__(self) -> None:
		if self.signer is None:
			api_key = self.api_key_id or get_api_key_id()
			priv = self.private_key_pem or get_private_key_material()
			priv_material = priv.decode("utf-8") if isinstance(priv, (bytes, bytearray)) else priv
			self.signer = KalshiSigner(api_key, priv_material.encode("utf-8") if isinstance(priv_material, str) else priv_material)

		self._ws_app: websocket.WebSocketApp | None = None
		self._thread: threading.Thread | None = None
		self._ready = threading.Event()
		self._closing = threading.Event()

		self._resolved_url = self.url or self._build_default_url()
		parsed = urlparse(self._resolved_url)
		self._path = parsed.path or "/"

	def _build_default_url(self) -> str:
		base = get_base_url(self.env)
		parsed = urlparse(base)
		scheme = "wss" if parsed.scheme == "https" else "ws"
		return urlunparse((scheme, parsed.netloc, self.path, "", "", ""))

	def _sign_headers(self) -> Dict[str, str]:
		assert self.signer is not None
		return dict(self.signer.headers("GET", self._path))

	def _handle_message(self, _ws: websocket.WebSocketApp, message: str) -> None:
		try:
			payload = json.loads(message)
		except json.JSONDecodeError:
			_LOG.debug("non-json websocket payload: %s", message)
			return
		if self.on_message:
			self.on_message(payload)
		if self.on_event and (msg_type := payload.get("type")):
			self.on_event(msg_type, payload)

	def _handle_open(self, _ws: websocket.WebSocketApp) -> None:
		self._ready.set()
		_LOG.debug("Kalshi websocket connection opened")

	def _handle_close(self, _ws: websocket.WebSocketApp, status_code: int, msg: str) -> None:
		self._ready.clear()
		self._thread = None
		if not self._closing.is_set():
			_LOG.warning("Kalshi websocket closed (%s) %s", status_code, msg)

	def _handle_error(self, _ws: websocket.WebSocketApp, error: Exception) -> None:
		_LOG.error("Kalshi websocket error: %s", error)

	def connect(self, *, block: bool = True) -> None:
		"""Open the WebSocket connection in a background thread."""
		if self._ws_app is not None:
			return

		signed_headers = self._sign_headers()
		header_list = [f"{k}: {v}" for k, v in signed_headers.items()]
		self._ws_app = websocket.WebSocketApp(
			self._resolved_url,
			header=header_list,
			on_message=self._handle_message,
			on_error=self._handle_error,
			on_close=self._handle_close,
			on_open=self._handle_open,
		)

		sslopt = self.sslopt or {}
		self._thread = threading.Thread(
			target=self._ws_app.run_forever,
			kwargs={"sslopt": sslopt, "ping_interval": self.ping_interval, "ping_timeout": self.ping_timeout},
			daemon=True,
		)
		self._thread.start()
		if block:
			connected = self._ready.wait(self._connect_timeout)
			if not connected:
				raise TimeoutError("Timed out waiting for Kalshi websocket to open")

	def close(self) -> None:
		self._closing.set()
		if self._ws_app is not None:
			self._ws_app.close()
		self._ws_app = None
		if self._thread and self._thread.is_alive():
			self._thread.join(timeout=2)
		self._thread = None
		self._ready.clear()
		self._closing.clear()

	def send(self, payload: Dict[str, Any]) -> None:
		if not self._ws_app or not self._ready.is_set():
			raise RuntimeError("WebSocket connection is not ready")
		self._ws_app.send(json.dumps(payload))

	def _next_id(self) -> int:
		request_id = self._request_id
		self._request_id += 1
		return request_id

	def subscribe(self, channels: list[str], *, params: Optional[Dict[str, Any]] = None, request_id: Optional[int] = None) -> int:
		req_id = request_id or self._next_id()
		payload = {
			"id": req_id,
			"cmd": "subscribe",
			"params": {"channels": channels, **(params or {})},
		}
		self.send(payload)
		return req_id

	def unsubscribe(self, subscription_ids: list[int], *, request_id: Optional[int] = None) -> int:
		req_id = request_id or self._next_id()
		payload = {
			"id": req_id,
			"cmd": "unsubscribe",
			"params": {"sids": subscription_ids},
		}
		self.send(payload)
		return req_id

	def update_subscription(self, subscription_id: int, *, action: str, market_tickers: list[str] | None = None, events: list[str] | None = None, request_id: Optional[int] = None) -> int:
		req_id = request_id or self._next_id()
		params: Dict[str, Any] = {"sid": subscription_id, "action": action}
		if market_tickers:
			params["market_tickers"] = market_tickers
		if events:
			params["event_tickers"] = events
		payload = {"id": req_id, "cmd": "update_subscription", "params": params}
		self.send(payload)
		return req_id

	def send_command(self, cmd: str, params: Optional[Dict[str, Any]] = None, *, request_id: Optional[int] = None) -> int:
		req_id = request_id or self._next_id()
		payload = {"id": req_id, "cmd": cmd}
		if params:
			payload["params"] = params
		self.send(payload)
		return req_id

	def __enter__(self) -> "KalshiWebSocketClient":
		self.connect()
		return self

	def __exit__(self, exc_type, exc, tb) -> None:
		self.close()
