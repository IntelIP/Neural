"""Canonical serialization and signed-envelope verification."""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .registry import validate_contract, validate_json_schema


class ContractEnvelopeError(ValueError):
    """A signed contract envelope failed a named fail-closed check."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def canonical_json(value: Any) -> str:
    """Serialize JSON data with stable UTF-8 cross-runtime bytes."""
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ContractEnvelopeError("canonicalization_failed", str(exc)) from exc


def contract_payload_hash(contract: Mapping[str, Any]) -> str:
    """Hash a contract after excluding its self-referential payloadHash."""
    unsigned = dict(contract)
    unsigned.pop("payloadHash", None)
    return hashlib.sha256(canonical_json(unsigned).encode("utf-8")).hexdigest()


def with_payload_hash(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Return a detached contract with its canonical payload hash."""
    result = json.loads(canonical_json(contract))
    result["payloadHash"] = contract_payload_hash(result)
    return result


def _unsigned_envelope(envelope: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(envelope)
    unsigned.pop("signature", None)
    return unsigned


def _signature(envelope: Mapping[str, Any], secret: str) -> str:
    digest = hmac.new(
        secret.encode("utf-8"),
        canonical_json(_unsigned_envelope(envelope)).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"sha256={digest}"


def sign_envelope(
    contract: Mapping[str, Any],
    *,
    secret: str,
    key_id: str,
    issued_at: str,
    expires_at: str,
    nonce: str,
) -> dict[str, Any]:
    """Build and sign one v1 transport envelope."""
    validated = validate_contract(contract)
    expected_hash = contract_payload_hash(validated)
    if not hmac.compare_digest(str(validated["payloadHash"]), expected_hash):
        raise ContractEnvelopeError("payload_hash_mismatch", "contract payloadHash is invalid")
    envelope: dict[str, Any] = {
        "algorithm": "hmac-sha256-v1",
        "contract": validated,
        "envelopeSchemaName": "SignedContractEnvelope",
        "envelopeVersion": "1.0.0",
        "expiresAt": expires_at,
        "issuedAt": issued_at,
        "keyId": key_id,
        "nonce": nonce,
    }
    envelope["signature"] = _signature(envelope, secret)
    validate_json_schema("SignedContractEnvelope", envelope)
    return envelope


@dataclass
class InMemoryNonceReplayGuard:
    """Deterministic injected replay guard for local and test consumers."""

    _seen: set[tuple[str, str]] = field(default_factory=set)

    def consume(self, key_id: str, nonce: str) -> bool:
        """Return false for a replay; otherwise retain the nonce."""
        identity = (key_id, nonce)
        if identity in self._seen:
            return False
        self._seen.add(identity)
        return True


def _timestamp(value: str, *, field_name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractEnvelopeError("timestamp_invalid", field_name) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ContractEnvelopeError("timestamp_invalid", field_name)
    return parsed.astimezone(timezone.utc)


def verify_envelope(
    envelope: Mapping[str, Any],
    *,
    secrets: Mapping[str, str],
    replay_guard: InMemoryNonceReplayGuard,
    now: datetime,
) -> dict[str, Any]:
    """Verify schema, hash, lifetime, signature, and nonce exactly once."""
    try:
        validated_envelope = validate_json_schema("SignedContractEnvelope", dict(envelope))
        contract = validate_contract(validated_envelope["contract"])
    except ValueError as exc:
        raise ContractEnvelopeError("schema_invalid", str(exc)) from exc

    expected_hash = contract_payload_hash(contract)
    if not hmac.compare_digest(str(contract["payloadHash"]), expected_hash):
        raise ContractEnvelopeError("payload_hash_mismatch", "contract payloadHash is invalid")

    current = now.astimezone(timezone.utc)
    issued_at = _timestamp(str(validated_envelope["issuedAt"]), field_name="issuedAt")
    expires_at = _timestamp(str(validated_envelope["expiresAt"]), field_name="expiresAt")
    if issued_at >= expires_at:
        raise ContractEnvelopeError("lifetime_invalid", "issuedAt must precede expiresAt")
    if current < issued_at:
        raise ContractEnvelopeError("not_yet_valid", "envelope issuedAt is in the future")
    if current >= expires_at:
        raise ContractEnvelopeError("expired", "envelope lifetime ended")

    key_id = str(validated_envelope["keyId"])
    secret = secrets.get(key_id)
    if secret is None:
        raise ContractEnvelopeError("unknown_key", key_id)
    expected_signature = _signature(validated_envelope, secret)
    if not hmac.compare_digest(str(validated_envelope["signature"]), expected_signature):
        raise ContractEnvelopeError("bad_signature", "signature verification failed")

    nonce = str(validated_envelope["nonce"])
    if not replay_guard.consume(key_id, nonce):
        raise ContractEnvelopeError("nonce_replay", nonce)
    return contract
