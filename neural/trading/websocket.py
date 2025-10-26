from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse, urlunparse

try:
    import websocket
except ImportError as exc:
    raise ImportError("websocket-client is required for Neural Kalshi WebSocket support.") from exc

from neural.auth.env import get_api_key_id, get_base_url, get_private_key_material
from neural.auth.signers.kalshi import KalshiSigner

_LOG = logging.getLogger(__name__)


@dataclass
class KalshiWebSocketClient:
    """Thin wrapper over the Kalshi WebSocket RPC channel."""

    signer: KalshiSigner | None = None
    api_key_id: str | None = None
    private_key_pem: bytes | None = None
    env: str | None = None
    url: str | None = None
    path: str = "/trade-api/ws/v2"
    on_message: Callable[[dict[str, Any]], None] | None = None
    on_event: Callable[[str, dict[str, Any]], None] | None = None
    risk_manager: Any = None  # RiskManager instance for real-time risk monitoring
    sslopt: dict[str, Any] | None = None
    ping_interval: float = 25.0
    ping_timeout: float = 10.0
    _connect_timeout: float = 10.0
    _request_id: int = field(init=False, default=1)

    def __post_init__(self) -> None:
        if self.signer is None:
            api_key = self.api_key_id or get_api_key_id()
            priv = self.private_key_pem or get_private_key_material()
            priv_material = priv.decode("utf-8") if isinstance(priv, (bytes, bytearray)) else priv
            self.signer = KalshiSigner(
                api_key,
                priv_material.encode("utf-8") if isinstance(priv_material, str) else priv_material,
            )

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

    def _sign_headers(self) -> dict[str, str]:
        """
        Generate authentication headers for WebSocket handshake.

        Bug Fix #11 Note: This method generates PSS (Probabilistic Signature Scheme)
        signatures required by Kalshi's WebSocket API. The signature must be included
        in the initial HTTP upgrade request headers.

        Returns:
                Dict with KALSHI-ACCESS-KEY, KALSHI-ACCESS-SIGNATURE, and KALSHI-ACCESS-TIMESTAMP
        """
        assert self.signer is not None
        return dict(self.signer.headers("GET", self._path))

    def _handle_message(self, _ws: websocket.WebSocketApp, message: str) -> None:
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            _LOG.debug("non-json websocket payload: %s", message)
            return

        # Process risk monitoring for price updates
        self._process_risk_monitoring(payload)

        if self.on_message:
            self.on_message(payload)
        if self.on_event and (msg_type := payload.get("type")):
            self.on_event(msg_type, payload)

    def _process_risk_monitoring(self, payload: dict[str, Any]) -> None:
        """Process websocket messages for risk monitoring."""
        if not self.risk_manager:
            return

        msg_type = payload.get("type")

        # Handle market price updates
        if msg_type == "market_price":
            market_data = payload.get("market", {})
            market_id = market_data.get("id")
            price_data = market_data.get("price", {})

            # Extract latest price (assuming yes_price for simplicity)
            latest_price = price_data.get("latest_price")
            if market_id and latest_price:
                # Update risk manager with new price
                if hasattr(self.risk_manager, "update_position_price"):
                    events = self.risk_manager.update_position_price(market_id, latest_price)
                    if events:
                        _LOG.info(
                            f"Risk events triggered for {market_id}: {[e.value for e in events]}"
                        )

        # Handle trade executions that might affect positions
        elif msg_type == "trade":
            trade_data = payload.get("trade", {})
            market_id = trade_data.get("market_id")
            if market_id and self.risk_manager:
                # Could trigger position updates or P&L recalculations
                _LOG.debug(f"Trade executed in market {market_id}")

        # Handle position updates
        elif msg_type == "position_update":
            position_data = payload.get("position", {})
            market_id = position_data.get("market_id")
            if market_id and self.risk_manager:
                # Update position in risk manager
                _LOG.debug(f"Position update for market {market_id}")

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
        """
        Open the WebSocket connection in a background thread.

        Bug Fix #11 Note: For proper SSL certificate verification, pass sslopt parameter
        when initializing the client. Example:
                import ssl, certifi
                sslopt = {"cert_reqs": ssl.CERT_REQUIRED, "ca_certs": certifi.where()}
                client = KalshiWebSocketClient(sslopt=sslopt)

        Args:
                block: If True, wait for connection to establish before returning

        Raises:
                TimeoutError: If connection doesn't establish within timeout period
        """
        if self._ws_app is not None:
            return

        signed_headers = self._sign_headers()
        self._ws_app = websocket.WebSocketApp(
            self._resolved_url,
            header=signed_headers,
            on_message=self._handle_message,
            on_error=self._handle_error,
            on_close=self._handle_close,
            on_open=self._handle_open,
        )

        sslopt = self.sslopt or {}
        self._thread = threading.Thread(
            target=self._ws_app.run_forever,
            kwargs={
                "sslopt": sslopt,
                "ping_interval": self.ping_interval,
                "ping_timeout": self.ping_timeout,
            },
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

    def send(self, payload: dict[str, Any]) -> None:
        if not self._ws_app or not self._ready.is_set():
            raise RuntimeError("WebSocket connection is not ready")
        self._ws_app.send(json.dumps(payload))

    def _next_id(self) -> int:
        request_id = self._request_id
        self._request_id += 1
        return request_id

    def subscribe(
        self,
        channels: list[str],
        *,
        market_tickers: list[str] | None = None,
        params: dict[str, Any] | None = None,
        request_id: int | None = None,
    ) -> int:
        """
        Subscribe to WebSocket channels with optional market filtering.

        Bug Fix #14: Added market_tickers parameter for server-side filtering.

        Args:
                channels: List of channel names (e.g., ["orderbook_delta", "trade"])
                market_tickers: Optional list of market tickers to filter (e.g., ["KXNFLGAME-..."])
                params: Additional parameters to merge into subscription
                request_id: Optional request ID for tracking

        Returns:
                Request ID used for this subscription
        """
        req_id = request_id or self._next_id()

        # Bug Fix #14: Build params with market_tickers support
        subscribe_params = {"channels": channels}
        if market_tickers:
            subscribe_params["market_tickers"] = market_tickers
        if params:
            subscribe_params.update(params)

        payload = {"id": req_id, "cmd": "subscribe", "params": subscribe_params}
        self.send(payload)
        return req_id

    def unsubscribe(self, subscription_ids: list[int], *, request_id: int | None = None) -> int:
        req_id = request_id or self._next_id()
        payload = {
            "id": req_id,
            "cmd": "unsubscribe",
            "params": {"sids": subscription_ids},
        }
        self.send(payload)
        return req_id

    def update_subscription(
        self,
        subscription_id: int,
        *,
        action: str,
        market_tickers: list[str] | None = None,
        events: list[str] | None = None,
        request_id: int | None = None,
    ) -> int:
        req_id = request_id or self._next_id()
        params: dict[str, Any] = {"sid": subscription_id, "action": action}
        if market_tickers:
            params["market_tickers"] = market_tickers
        if events:
            params["event_tickers"] = events
        payload = {"id": req_id, "cmd": "update_subscription", "params": params}
        self.send(payload)
        return req_id

    def send_command(
        self, cmd: str, params: dict[str, Any] | None = None, *, request_id: int | None = None
    ) -> int:
        req_id = request_id or self._next_id()
        payload = {"id": req_id, "cmd": cmd}
        if params:
            payload["params"] = params
        self.send(payload)
        return req_id

    def __enter__(self) -> KalshiWebSocketClient:
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
