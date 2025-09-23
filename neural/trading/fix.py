from __future__ import annotations

import asyncio
import base64
import contextlib
import ssl
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence

import simplefix
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.serialization import load_pem_private_key

from neural.auth.env import get_api_key_id, get_private_key_material

SENDER_SIDE_BUY = {"buy", "yes", "1", 1}
SENDER_SIDE_SELL = {"sell", "no", "2", 2}

ORD_TYPE_LIMIT = {"limit", "2", 2}
ORD_TYPE_MARKET = {"market", "1", 1}

TIF_MAP = {
	"day": "0",
	"gtc": "1",
	"ioc": "3",
	"fok": "4",
	"gtd": "6",
}

EXEC_INST_MAP = {
	"post_only": "6",
}


@dataclass(slots=True)
class FIXConnectionConfig:
	host: str = "fix.elections.kalshi.com"
	port: int = 8228
	target_comp_id: str = "KalshiNR"
	sender_comp_id: Optional[str] = None
	heartbeat_interval: int = 30
	reset_seq_num: bool = True
	cancel_on_disconnect: bool = False
	skip_pending_exec_reports: bool = False
	listener_session: bool = False
	receive_settlement_reports: bool = False
	use_tls: bool = True


class KalshiFIXClient:
	"""Asynchronous FIX 5.0 SP2 client tailored for Kalshi order entry."""

	def __init__(
		self,
		config: FIXConnectionConfig | None = None,
		*,
		api_key_id: Optional[str] = None,
		private_key_pem: Optional[bytes] = None,
		on_message: Optional[Callable[[simplefix.FixMessage], None]] = None,
		loop: Optional[asyncio.AbstractEventLoop] = None,
	):
		self.config = config or FIXConnectionConfig()
		self.config.sender_comp_id = self.config.sender_comp_id or api_key_id or get_api_key_id()
		if not self.config.sender_comp_id:
			raise ValueError("sender_comp_id (FIX API key) must be provided")

		pem = private_key_pem or get_private_key_material()
		self._private_key = load_pem_private_key(pem, password=None)

		self.on_message = on_message
		self._loop = loop or asyncio.get_event_loop()
		self._reader: asyncio.StreamReader | None = None
		self._writer: asyncio.StreamWriter | None = None
		self._parser = simplefix.FixParser()
		self._seq_num = 1
		self._send_lock = asyncio.Lock()
		self._reader_task: asyncio.Task[None] | None = None
		self._heartbeat_task: asyncio.Task[None] | None = None
		self._logon_event = asyncio.Event()
		self._logout_event = asyncio.Event()
		self._running = False

	async def connect(self, *, timeout: float = 10.0) -> None:
		if self._reader is not None:
			return

		ssl_context = ssl.create_default_context() if self.config.use_tls else None
		self._reader, self._writer = await asyncio.open_connection(
			self.config.host,
			self.config.port,
			ssl=ssl_context,
		)
		self._running = True
		self._reader_task = self._loop.create_task(self._read_loop())
		await self._send_logon()
		await asyncio.wait_for(self._logon_event.wait(), timeout=timeout)
		self._heartbeat_task = self._loop.create_task(self._heartbeat_loop())

	async def close(self) -> None:
		if not self._reader:
			return
		await self.logout()
		await asyncio.sleep(0)
		self._running = False
		if self._heartbeat_task:
			self._heartbeat_task.cancel()
			with contextlib.suppress(Exception):
				await self._heartbeat_task
			self._heartbeat_task = None
		if self._reader_task:
			self._reader_task.cancel()
			with contextlib.suppress(Exception):
				await self._reader_task
			self._reader_task = None
		if self._writer:
			self._writer.close()
			with contextlib.suppress(Exception):
				await self._writer.wait_closed()
		self._reader = None
		self._writer = None
		self._parser = simplefix.FixParser()
		self._seq_num = 1
		self._logon_event.clear()
		self._logout_event.clear()

	async def logout(self) -> None:
		if not self._writer or self._writer.is_closing():
			return
		self._logout_event.clear()
		await self._send_message("5", [])
		with contextlib.suppress(asyncio.TimeoutError):
			await asyncio.wait_for(self._logout_event.wait(), timeout=3.0)

	async def _heartbeat_loop(self) -> None:
		try:
			while self._running:
				await asyncio.sleep(self.config.heartbeat_interval)
				await self._send_message("0", [])
		except asyncio.CancelledError:
			return

	async def _send_logon(self) -> None:
		fields: List[tuple[int, Any]] = [
			(98, "0"),
			(108, str(self.config.heartbeat_interval)),
		]
		if self.config.reset_seq_num:
			fields.append((141, "Y"))
		if self.config.cancel_on_disconnect:
			fields.append((8013, "Y"))
		if self.config.listener_session:
			fields.append((20126, "Y"))
		if self.config.receive_settlement_reports:
			fields.append((20127, "Y"))
		if self.config.skip_pending_exec_reports:
			fields.append((21003, "Y"))

		await self._send_message("A", fields, include_signature=True)

	def _utc_timestamp(self, *, millis: bool = True) -> str:
		ts = datetime.utcnow()
		fmt = "%Y%m%d-%H:%M:%S.%f" if millis else "%Y%m%d-%H:%M:%S"
		value = ts.strftime(fmt)
		return value[:-3] if millis else value

	async def _send_message(self, msg_type: str, body_fields: Sequence[tuple[int, Any]], *, include_signature: bool = False) -> None:
		if not self._writer:
			raise RuntimeError("FIX connection not established")
		async with self._send_lock:
			seq_num = self._seq_num
			sending_time = self._utc_timestamp()
			message = simplefix.FixMessage()
			message.append_pair(8, "FIXT.1.1")
			message.append_pair(35, msg_type)
			message.append_pair(49, self.config.sender_comp_id)
			message.append_pair(56, self.config.target_comp_id)
			message.append_pair(34, str(seq_num))
			message.append_pair(52, sending_time)
			message.append_pair(1137, "9")
			if include_signature:
				signature_b64 = self._sign_logon_payload(sending_time, msg_type, seq_num)
				message.append_pair(95, str(len(signature_b64)))
				message.append_pair(96, signature_b64)
			for tag, value in body_fields:
				message.append_pair(tag, str(value))
			raw = message.encode()
			self._writer.write(raw)
			await self._writer.drain()
			self._seq_num += 1

	async def _read_loop(self) -> None:
		try:
			while self._running and self._reader:
				data = await self._reader.read(4096)
				if not data:
					break
				self._parser.append_buffer(data)
				while (msg := self._parser.get_message()) is not None:
					self._handle_incoming(msg)
		except asyncio.CancelledError:
			return
		finally:
			self._running = False
			self._logon_event.set()
			self._logout_event.set()

	def _handle_incoming(self, message: simplefix.FixMessage) -> None:
		msg_type = _get_field(message, 35)
		if msg_type == "A":
			self._logon_event.set()
		elif msg_type == "5":
			self._logout_event.set()
			self._running = False
		elif msg_type == "1":
			test_req_id = _get_field(message, 112)
			self._loop.create_task(self._send_message("0", [(112, test_req_id)]))
		if self.on_message:
			self.on_message(message)

	def _sign_logon_payload(self, sending_time: str, msg_type: str, seq_num: int) -> str:
		payload = "\x01".join(
			[
				sending_time,
				msg_type,
				str(seq_num),
				self.config.sender_comp_id,
				self.config.target_comp_id,
			]
		)
		signature = self._private_key.sign(
			payload.encode("utf-8"),
			padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
			hashes.SHA256(),
		)
		return base64.b64encode(signature).decode("ascii")

	async def new_order_single(
		self,
		cl_order_id: str,
		symbol: str,
		side: str,
		quantity: int,
		price: int,
		*,
		order_type: str = "limit",
		time_in_force: Optional[str] = None,
		exec_inst: Optional[str] = None,
		expire_time: Optional[str] = None,
		customer_account: Optional[str] = None,
		minimum_quantity: Optional[int] = None,
	) -> None:
		fields: List[tuple[int, Any]] = [
			(11, cl_order_id),
			(55, symbol),
			(54, _map_side(side)),
			(38, str(quantity)),
			(40, _map_order_type(order_type)),
		]
		if price is not None:
			fields.append((44, str(price)))
		if time_in_force:
			fields.append((59, _map_tif(time_in_force)))
		if exec_inst:
			fields.append((18, EXEC_INST_MAP.get(exec_inst, exec_inst)))
		if expire_time:
			fields.append((126, expire_time))
		if minimum_quantity is not None:
			fields.append((110, str(minimum_quantity)))
		if customer_account:
			fields.extend([(453, "1"), (448, customer_account), (452, "24")])
		await self._send_message("D", fields)

	async def cancel_order(
		self,
		cl_order_id: str,
		orig_cl_order_id: str,
		symbol: str,
		side: str,
		*,
		order_id: Optional[str] = None,
	) -> None:
		fields: List[tuple[int, Any]] = [
			(11, cl_order_id),
			(41, orig_cl_order_id),
			(55, symbol),
			(54, _map_side(side)),
		]
		if order_id:
			fields.append((37, order_id))
		await self._send_message("F", fields)

	async def replace_order(
		self,
		cl_order_id: str,
		orig_cl_order_id: str,
		symbol: str,
		side: str,
		*,
		quantity: Optional[int] = None,
		price: Optional[int] = None,
		time_in_force: Optional[str] = None,
	) -> None:
		fields: List[tuple[int, Any]] = [
			(11, cl_order_id),
			(41, orig_cl_order_id),
			(55, symbol),
			(54, _map_side(side)),
			(40, "2"),
		]
		if quantity is not None:
			fields.append((38, str(quantity)))
		if price is not None:
			fields.append((44, str(price)))
		if time_in_force:
			fields.append((59, _map_tif(time_in_force)))
		await self._send_message("G", fields)

	async def mass_cancel(self, cl_order_id: str) -> None:
		fields = [(11, cl_order_id), (530, "6")]
		await self._send_message("q", fields)

	async def test_request(self, test_id: str) -> None:
		await self._send_message("1", [(112, test_id)])

	async def __aenter__(self) -> "KalshiFIXClient":
		await self.connect()
		return self

	async def __aexit__(self, exc_type, exc, tb) -> None:
		await self.close()

	@staticmethod
	def to_dict(message: simplefix.FixMessage) -> Dict[int, str]:
		return {tag: value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else value for tag, value in message}


def _map_side(side: str | int) -> str:
	if side in SENDER_SIDE_BUY:
		return "1"
	if side in SENDER_SIDE_SELL:
		return "2"
	raise ValueError("side must be one of 'buy'/'sell' or 1/2")


def _map_order_type(order_type: str | int) -> str:
	if order_type in ORD_TYPE_LIMIT:
		return "2"
	if order_type in ORD_TYPE_MARKET:
		return "1"
	raise ValueError("Unsupported order type")


def _map_tif(tif: str | int) -> str:
	if isinstance(tif, int):
		return str(tif)
	mapped = TIF_MAP.get(tif.lower())
	if not mapped:
		raise ValueError("Unsupported time in force")
	return mapped


def _get_field(message: simplefix.FixMessage, tag: int) -> Optional[str]:
	value = message.get(tag)
	if value is None:
		return None
	if isinstance(value, (bytes, bytearray)):
		return value.decode("utf-8")
	return value
