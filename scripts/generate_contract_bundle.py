#!/usr/bin/env python3
"""Generate deterministic Neural contract fixtures and their digest manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neural.contracts import (  # noqa: E402
    CONTRACT_NAMES,
    CONTRACT_VERSION,
    canonical_json,
    sign_envelope,
    validate_contract,
    with_payload_hash,
)

CONTRACT_ROOT = ROOT / "neural" / "contracts"
SCHEMA_PATH = CONTRACT_ROOT / "schemas" / "neural-contracts-v1.schema.json"
FIXTURE_PATH = CONTRACT_ROOT / "fixtures" / "golden-v1.json"
MANIFEST_PATH = CONTRACT_ROOT / "manifest.json"
FIXTURE_SECRET = "nrcl-67-test-secret-not-for-production"


def _base(name: str, object_id: str, *, lineage: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "createdAt": "2026-07-29T12:00:00Z",
        "environment": "test",
        "lineageRefs": lineage,
        "objectId": object_id,
        "payloadHash": "0" * 64,
        "producer": {"name": "neural-sdk", "version": "0.4.0"},
        "redaction": "internal",
        "schemaName": name,
        "schemaVersion": CONTRACT_VERSION,
        "sourceRef": "fixture://nrcl-67/golden-v1",
    }


def _lineage(contract: dict[str, Any]) -> dict[str, str]:
    return {
        "objectId": str(contract["objectId"]),
        "payloadHash": str(contract["payloadHash"]),
        "schemaName": str(contract["schemaName"]),
        "schemaVersion": str(contract["schemaVersion"]),
    }


def build_contracts() -> dict[str, dict[str, Any]]:
    snapshot = with_payload_hash(
        {
            **_base("MarketSnapshot", "market-snapshot-001", lineage=[]),
            "payload": {
                "marketId": "KX-DEMO-YES",
                "noPrice": "0.48",
                "observedAt": "2026-07-29T12:00:00Z",
                "source": "kalshi-fixture",
                "ticker": "KX-DEMO-YES",
                "yesPrice": "0.52",
            },
        }
    )
    evidence = with_payload_hash(
        {
            **_base("ResearchEvidenceRef", "evidence-001", lineage=[]),
            "payload": {
                "capturedAt": "2026-07-29T11:59:00Z",
                "evidenceType": "fixture",
                "sha256": hashlib.sha256(b"nrcl-67-evidence").hexdigest(),
                "uri": "fixture://nrcl-67/evidence-001",
            },
        }
    )
    intent = with_payload_hash(
        {
            **_base(
                "ExecutionIntent",
                "execution-intent-001",
                lineage=[_lineage(snapshot), _lineage(evidence)],
            ),
            "payload": {
                "evidenceObjectIds": [evidence["objectId"]],
                "limitPrice": "0.52",
                "marketSnapshotObjectId": snapshot["objectId"],
                "maxContracts": 2,
                "rationale": "Deterministic fixture intent for cross-runtime validation.",
                "side": "buy_yes",
                "ticker": "KX-DEMO-YES",
            },
        }
    )
    risk = with_payload_hash(
        {
            **_base("RiskDecision", "risk-decision-001", lineage=[_lineage(intent)]),
            "payload": {
                "checkedAt": "2026-07-29T12:01:00Z",
                "decision": "pass",
                "intentObjectId": intent["objectId"],
                "intentPayloadHash": intent["payloadHash"],
                "policyVersion": "fixture-policy-v1",
                "violations": [],
            },
        }
    )
    paper = with_payload_hash(
        {
            **_base(
                "PaperOrder",
                "paper-order-001",
                lineage=[_lineage(intent), _lineage(risk)],
            ),
            "payload": {
                "averageFillPrice": "0.51",
                "countContracts": 2,
                "filledContracts": 2,
                "intentObjectId": intent["objectId"],
                "limitPrice": "0.52",
                "riskDecisionObjectId": risk["objectId"],
                "side": "buy_yes",
                "status": "filled",
            },
        }
    )
    review = with_payload_hash(
        {
            **_base("PostTradeReview", "post-trade-review-001", lineage=[_lineage(paper)]),
            "payload": {
                "lessons": ["Fixture preserves exact lineage and deterministic decimals."],
                "outcome": "win",
                "paperOrderObjectId": paper["objectId"],
                "realizedPnlDollars": "0.98",
                "reviewedAt": "2026-07-29T13:00:00Z",
            },
        }
    )
    contracts = {
        contract["schemaName"]: contract
        for contract in (snapshot, evidence, intent, risk, paper, review)
    }
    for contract in contracts.values():
        validate_contract(contract)
    return contracts


def _formatted(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def expected_outputs() -> tuple[str, str]:
    contracts = build_contracts()
    signed = sign_envelope(
        contracts["ExecutionIntent"],
        secret=FIXTURE_SECRET,
        key_id="fixture-key-v1",
        issued_at="2026-07-29T12:00:00Z",
        expires_at="2026-07-29T13:00:00Z",
        nonce="nrcl-67-fixture-0001",
    )
    fixture = {
        "bundleVersion": CONTRACT_VERSION,
        "contracts": contracts,
        "signedExecutionIntent": signed,
    }
    fixture_text = _formatted(fixture)
    schema_text = SCHEMA_PATH.read_text(encoding="utf-8")
    manifest = {
        "bundleVersion": CONTRACT_VERSION,
        "canonicalFixtureSha256": hashlib.sha256(
            canonical_json(fixture).encode("utf-8")
        ).hexdigest(),
        "contractNames": list(CONTRACT_NAMES),
        "files": {
            "fixtures/golden-v1.json": hashlib.sha256(fixture_text.encode("utf-8")).hexdigest(),
            "schemas/neural-contracts-v1.schema.json": hashlib.sha256(
                schema_text.encode("utf-8")
            ).hexdigest(),
        },
        "generator": "scripts/generate_contract_bundle.py",
    }
    return fixture_text, _formatted(manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    fixture_text, manifest_text = expected_outputs()
    expected = {FIXTURE_PATH: fixture_text, MANIFEST_PATH: manifest_text}

    if args.check:
        stale = [
            str(path.relative_to(ROOT))
            for path, content in expected.items()
            if not path.exists() or path.read_text(encoding="utf-8") != content
        ]
        if stale:
            raise SystemExit("stale generated contract files: " + ", ".join(stale))
        print("contract bundle is current")
        return 0

    for path, content in expected.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
