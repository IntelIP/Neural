from __future__ import annotations

import asyncio
import json
import logging
import ssl
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse, urlunparse

try:
    import certifi
    import websockets
except ImportError as exc:
    raise ImportError(
        "websockets and certifi are required for Neural Kalshi WebSocket support."
    ) from exc

from neural.auth.env import get_api_key_id, get_base_url, get_private_key_material
from neural.auth.signers.kalshi import KalshiSigner

_LOG = logging.getLogger(__name__)


@dataclass
class KalshiWebSocketClient:
    """Thin wrapper over Kalshi WebSocket RPC channel using websockets library."""

    signer: KalshiSigner | None = None
    api_key_id: str | None = None
    private_key_pem: bytes | None = None
    env: str | None = None
    url: str | None = None
    path: str = "/trade-api/ws/v2"
    on_message: Callable[[dict[str, Any]], None] | None = None
    on_event: Callable[[str, dict[str, Any]], None] | None = None
    risk_manager: Any = None  # RiskManager instance for real-time risk monitoring
    ssl_context: ssl.SSLContext | None = None
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

        # Use websockets library instead of websocket-client
        self._websocket: Any | None = None
        self._task: asyncio.Task | None = None
        self._ready = threading.Event()
        self._closing = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None

        self._resolved_url = self.url or self._build_default_url()
        parsed = urlparse(self._resolved_url)
        self._path = parsed.path or "/"

        # Create SSL context with certifi for proper certificate verification
        if self.ssl_context is None:
            self.ssl_context = ssl.create_default_context(cafile=certifi.where())

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
        in initial HTTP upgrade request headers.

        Returns:
                Dict with KALSHI-ACCESS-KEY, KALSHI-ACCESS-SIGNATURE, and KALSHI-ACCESS-TIMESTAMP
        """
        assert self.signer is not None
        return dict(self.signer.headers("GET", self._path))

    def _handle_message(self, message: str) -> None:
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

    async def _listen(self) -> None:
        """Background task to listen for WebSocket messages."""
        try:
            if self._websocket:
                async for message in self._websocket:
                    self._handle_message(message)
        except websockets.exceptions.ConnectionClosed:
            _LOG.info("WebSocket connection closed")
        except Exception as e:
            _LOG.error("WebSocket error: %s", e)
        finally:
            self._ready.clear()

    def connect(self, *, block: bool = True) -> None:
        """
        Open WebSocket connection in a background thread.

        Bug Fix #11 Note: Uses websockets.connect with proper SSL context and authentication
        headers to fix 403 Forbidden issues with websocket-client library.

        Args:
                block: If True, wait for connection to establish before returning

        Raises:
                TimeoutError: If connection doesn't establish within timeout period
        """
        if self._websocket is not None:
            return

        def _run_in_thread():
            """Run the async connection in a separate thread."""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop

            async def _connect_async():
                try:
                    signed_headers = self._sign_headers()
                    self._websocket = await websockets.connect(
                        self._resolved_url,
                        additional_headers=signed_headers,
                        ssl=self.ssl_context,
                        ping_interval=self.ping_interval,
                        ping_timeout=self.ping_timeout,
                    )
                    self._ready.set()
                    _LOG.debug("Kalshi websocket connection opened")

                    # Start listening for messages
                    await self._listen()
                except Exception as e:
                    _LOG.error("Failed to connect to Kalshi websocket: %s", e)
                    self._ready.set()  # Unblock waiting threads
                finally:
                    if self._websocket:
                        await self._websocket.close()
                    self._websocket = None

            try:
                loop.run_until_complete(_connect_async())
            except Exception as e:
                _LOG.error("WebSocket thread error: %s", e)
            finally:
                loop.close()

        self._thread = threading.Thread(target=_run_in_thread, daemon=True)
        self._thread.start()

        if block:
            connected = self._ready.wait(self._connect_timeout)
            if not connected:
                raise TimeoutError("Timed out waiting for Kalshi websocket to open")
            if self._websocket is None:
                raise ConnectionError("Failed to establish WebSocket connection")

    def close(self) -> None:
        self._closing.set()
        if self._loop and self._websocket:
            # Schedule close on the event loop
            asyncio.run_coroutine_threadsafe(self._websocket.close(), self._loop)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        self._thread = None
        self._ready.clear()
        self._closing.clear()

    def send(self, payload: dict[str, Any]) -> None:
        if not self._websocket or not self._ready.is_set():
            raise RuntimeError("WebSocket connection is not ready")

        def _send_in_loop():
            if self._loop and self._websocket:
                asyncio.run_coroutine_threadsafe(
                    self._websocket.send(json.dumps(payload)), self._loop
                )

        # Send in the event loop thread
        threading.Thread(target=_send_in_loop, daemon=True).start()

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
