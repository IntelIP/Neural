from __future__ import annotations

import copy
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from neural.contracts import (
    CONTRACT_NAMES,
    ContractEnvelopeError,
    InMemoryNonceReplayGuard,
    contract_model,
    contract_payload_hash,
    validate_contract,
    validate_json_schema,
    verify_envelope,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = ROOT / "neural/contracts/fixtures/golden-v1.json"
FIXTURE_SECRET = "nrcl-67-test-secret-not-for-production"
FIXED_NOW = datetime(2026, 7, 29, 12, 30, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def fixture_bundle() -> dict[str, object]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_generated_bundle_is_current() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/generate_contract_bundle.py", "--check"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "contract bundle is current"


def test_all_six_golden_contracts_pass_json_schema_and_pydantic(
    fixture_bundle: dict[str, object],
) -> None:
    contracts = fixture_bundle["contracts"]
    assert isinstance(contracts, dict)
    assert set(contracts) == set(CONTRACT_NAMES)

    for name, contract in contracts.items():
        assert validate_json_schema(name, contract) == contract
        assert contract_model(name).model_validate(contract).root == contract
        assert validate_contract(contract) == contract
        assert contract_payload_hash(contract) == contract["payloadHash"]


def test_unknown_versions_and_fields_fail_closed(fixture_bundle: dict[str, object]) -> None:
    contracts = fixture_bundle["contracts"]
    assert isinstance(contracts, dict)
    intent = copy.deepcopy(contracts["ExecutionIntent"])

    intent["schemaVersion"] = "2.0.0"
    with pytest.raises(ValueError, match="unsupported_version"):
        validate_contract(intent)

    intent = copy.deepcopy(contracts["ExecutionIntent"])
    intent["unexpected"] = True
    with pytest.raises(ValueError):
        validate_contract(intent)


def test_malformed_lineage_fails_closed(fixture_bundle: dict[str, object]) -> None:
    contracts = fixture_bundle["contracts"]
    assert isinstance(contracts, dict)
    intent = copy.deepcopy(contracts["ExecutionIntent"])
    intent["lineageRefs"][0].pop("payloadHash")

    with pytest.raises(ValueError):
        validate_contract(intent)


def test_signed_fixture_accepts_once_then_rejects_replay(
    fixture_bundle: dict[str, object],
) -> None:
    envelope = fixture_bundle["signedExecutionIntent"]
    guard = InMemoryNonceReplayGuard()

    contract = verify_envelope(
        envelope,
        secrets={"fixture-key-v1": FIXTURE_SECRET},
        replay_guard=guard,
        now=FIXED_NOW,
    )
    assert contract["schemaName"] == "ExecutionIntent"

    with pytest.raises(ContractEnvelopeError) as replay:
        verify_envelope(
            envelope,
            secrets={"fixture-key-v1": FIXTURE_SECRET},
            replay_guard=guard,
            now=FIXED_NOW,
        )
    assert replay.value.code == "nonce_replay"


@pytest.mark.parametrize(
    ("mutation", "now", "expected_code"),
    [
        (
            lambda envelope: envelope.update({"signature": f"sha256={'0' * 64}"}),
            FIXED_NOW,
            "bad_signature",
        ),
        (
            lambda envelope: envelope["contract"]["payload"].update(
                {"rationale": "Deterministic fixture intent for cross-runtime validation!"}
            ),
            FIXED_NOW,
            "payload_hash_mismatch",
        ),
        (
            lambda envelope: None,
            datetime(2026, 7, 29, 13, 0, tzinfo=timezone.utc),
            "expired",
        ),
    ],
)
def test_signed_fixture_rejects_named_failures(
    fixture_bundle: dict[str, object],
    mutation: object,
    now: datetime,
    expected_code: str,
) -> None:
    envelope = copy.deepcopy(fixture_bundle["signedExecutionIntent"])
    assert callable(mutation)
    mutation(envelope)

    with pytest.raises(ContractEnvelopeError) as failure:
        verify_envelope(
            envelope,
            secrets={"fixture-key-v1": FIXTURE_SECRET},
            replay_guard=InMemoryNonceReplayGuard(),
            now=now,
        )
    assert failure.value.code == expected_code
