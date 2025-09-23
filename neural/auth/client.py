from __future__ import annotations
import json, time
from typing import Any, Mapping
import requests
from requests import Session, Response
from .env import get_base_url
from .signers.kalshi import Signer

DEFAULT_TIMEOUT = 15
RETRY_STATUSES = {429, 500, 502, 503, 504}

class AuthClient:
	def __init__(self, signer: Signer, env: str | None = None, timeout: int = DEFAULT_TIMEOUT, session: Session | None = None):
		self.signer = signer
		self.base_url = get_base_url(env)
		self.timeout = timeout
		self._s = session or requests.Session()

	def _request(self, method: str, path: str, *, params: Mapping[str, Any] | None = None, json_body: Any | None = None) -> Response:
		url = f"{self.base_url}{path}"
		headers = dict(self.signer.headers(method, path))
		if json_body is not None:
			headers["Content-Type"] = "application/json"

		backoff = 0.5
		for attempt in range(5):
			resp = self._s.request(
				method=method.upper(),
				url=url,
				headers=headers,
				params=params,
				data=None if json_body is None else json.dumps(json_body),
				timeout=self.timeout,
			)
			if 200 <= resp.status_code < 300:
				return resp

			if resp.status_code == 401 and attempt == 0:
				headers = dict(self.signer.headers(method, path))
				if json_body is not None:
					headers["Content-Type"] = "application/json"
				continue

			if resp.status_code in RETRY_STATUSES:
				time.sleep(backoff)
				backoff = min(backoff * 2, 4.0)
				continue
			break
		return resp

	def get(self, path: str, params: Mapping[str, Any] | None = None) -> Any:
		r = self._request("GET", path, params=params)
		self._raise_for_status(r)
		return self._safe_json(r)

	def post(self, path: str, body: Any | None = None) -> Any:
		r = self._request("POST", path, json_body=body)
		self._raise_for_status(r)
		return self._safe_json(r)

	def delete(self, path: str) -> Any:
		r = self._request("DELETE", path)
		self._raise_for_status(r)
		return self._safe_json(r)

	@staticmethod
	def _safe_json(resp: Response) -> Any:
		if resp.content and resp.headers.get("Content-Type", "").startswith("application/json"):
			return resp.json()
		return {"status_code": resp.status_code, "text": resp.text}

	@staticmethod
	def _raise_for_status(resp: Response) -> None:
		try:
			resp.raise_for_status()
		except requests.HTTPError as e:
			msg = None
			try:
				msg = resp.json()
			except Exception:
				msg = resp.text
			raise requests.HTTPError(f"{e} | body={msg}") from None

