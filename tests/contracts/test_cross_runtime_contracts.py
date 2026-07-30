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
    load_contract_bundle,
    sign_envelope,
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


def test_contract_bundle_callers_cannot_mutate_cached_authority() -> None:
    first = load_contract_bundle()
    first["$defs"].clear()

    assert "ExecutionIntent" in load_contract_bundle()["$defs"]


def test_malformed_lineage_fails_closed(fixture_bundle: dict[str, object]) -> None:
    contracts = fixture_bundle["contracts"]
    assert isinstance(contracts, dict)
    intent = copy.deepcopy(contracts["ExecutionIntent"])
    intent["lineageRefs"][0].pop("payloadHash")

    with pytest.raises(ValueError):
        validate_contract(intent)


def test_lineage_ids_timestamps_and_uris_use_runtime_format_validation(
    fixture_bundle: dict[str, object],
) -> None:
    contracts = fixture_bundle["contracts"]
    assert isinstance(contracts, dict)

    intent = copy.deepcopy(contracts["ExecutionIntent"])
    intent["lineageRefs"][0]["objectId"] = "contains spaces"
    intent["payloadHash"] = contract_payload_hash(intent)
    with pytest.raises(ValueError):
        validate_contract(intent)

    snapshot = copy.deepcopy(contracts["MarketSnapshot"])
    snapshot["createdAt"] = "not-a-date"
    snapshot["payloadHash"] = contract_payload_hash(snapshot)
    with pytest.raises(ValueError):
        validate_contract(snapshot)

    evidence = copy.deepcopy(contracts["ResearchEvidenceRef"])
    evidence["payload"]["uri"] = "not a uri"
    evidence["payloadHash"] = contract_payload_hash(evidence)
    with pytest.raises(ValueError):
        validate_contract(evidence)


def test_identifiers_reject_trailing_newlines(fixture_bundle: dict[str, object]) -> None:
    contracts = fixture_bundle["contracts"]
    assert isinstance(contracts, dict)
    intent = copy.deepcopy(contracts["ExecutionIntent"])
    intent["objectId"] += "\n"
    intent["payloadHash"] = contract_payload_hash(intent)

    with pytest.raises(ValueError):
        validate_contract(intent)


def test_semantically_wrong_lineage_and_payload_hash_fail_closed(
    fixture_bundle: dict[str, object],
) -> None:
    contracts = fixture_bundle["contracts"]
    assert isinstance(contracts, dict)

    tampered = copy.deepcopy(contracts["ExecutionIntent"])
    tampered["payload"]["rationale"] += "!"
    with pytest.raises(ValueError, match="payloadHash"):
        validate_contract(tampered)

    wrong_lineage = copy.deepcopy(contracts["ExecutionIntent"])
    wrong_lineage["lineageRefs"] = []
    wrong_lineage["payloadHash"] = contract_payload_hash(wrong_lineage)
    with pytest.raises(ValueError, match="MarketSnapshot"):
        validate_contract(wrong_lineage)


def test_paper_fill_semantics_fail_closed(fixture_bundle: dict[str, object]) -> None:
    contracts = fixture_bundle["contracts"]
    assert isinstance(contracts, dict)
    paper = copy.deepcopy(contracts["PaperOrder"])
    paper["payload"]["countContracts"] = 1
    paper["payloadHash"] = contract_payload_hash(paper)

    with pytest.raises(ValueError, match="cannot exceed countContracts"):
        validate_contract(paper)


def test_cross_field_semantics_fail_closed(fixture_bundle: dict[str, object]) -> None:
    contracts = fixture_bundle["contracts"]
    assert isinstance(contracts, dict)

    risk = copy.deepcopy(contracts["RiskDecision"])
    risk["payload"]["decision"] = "reject"
    risk["payloadHash"] = contract_payload_hash(risk)
    with pytest.raises(ValueError, match="reject requires at least one violation"):
        validate_contract(risk)

    cancelled = copy.deepcopy(contracts["PaperOrder"])
    cancelled["payload"]["status"] = "cancelled"
    cancelled["payloadHash"] = contract_payload_hash(cancelled)
    with pytest.raises(ValueError, match="cancelled cannot be completely filled"):
        validate_contract(cancelled)

    review = copy.deepcopy(contracts["PostTradeReview"])
    review["payload"]["outcome"] = "unresolved"
    review["payloadHash"] = contract_payload_hash(review)
    with pytest.raises(ValueError, match="unresolved requires zero realized PnL"):
        validate_contract(review)


def test_maximum_evidence_intent_has_lineage_capacity(
    fixture_bundle: dict[str, object],
) -> None:
    contracts = fixture_bundle["contracts"]
    assert isinstance(contracts, dict)
    intent = copy.deepcopy(contracts["ExecutionIntent"])
    evidence = contracts["ResearchEvidenceRef"]
    intent["payload"]["evidenceObjectIds"] = [f"evidence-{index:03d}" for index in range(32)]
    intent["lineageRefs"] = [
        intent["lineageRefs"][0],
        *[
            {
                **intent["lineageRefs"][1],
                "objectId": evidence_id,
                "payloadHash": evidence["payloadHash"],
            }
            for evidence_id in intent["payload"]["evidenceObjectIds"]
        ],
    ]
    intent["payloadHash"] = contract_payload_hash(intent)

    assert validate_contract(intent) == intent


def test_duplicate_lineage_identity_fails_closed(fixture_bundle: dict[str, object]) -> None:
    contracts = fixture_bundle["contracts"]
    assert isinstance(contracts, dict)
    intent = copy.deepcopy(contracts["ExecutionIntent"])
    duplicate = copy.deepcopy(intent["lineageRefs"][0])
    duplicate["payloadHash"] = "f" * 64
    intent["lineageRefs"].append(duplicate)
    intent["payloadHash"] = contract_payload_hash(intent)

    with pytest.raises(ValueError, match="duplicate schemaName and objectId"):
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


def test_signed_fixture_rejects_a_naive_verifier_clock(
    fixture_bundle: dict[str, object],
) -> None:
    with pytest.raises(ContractEnvelopeError) as failure:
        verify_envelope(
            fixture_bundle["signedExecutionIntent"],
            secrets={"fixture-key-v1": FIXTURE_SECRET},
            replay_guard=InMemoryNonceReplayGuard(),
            now=datetime(2026, 7, 29, 12, 30),
        )
    assert failure.value.code == "timestamp_invalid"


def test_empty_hmac_secrets_fail_closed(fixture_bundle: dict[str, object]) -> None:
    envelope = fixture_bundle["signedExecutionIntent"]
    with pytest.raises(ContractEnvelopeError) as verification:
        verify_envelope(
            envelope,
            secrets={"fixture-key-v1": ""},
            replay_guard=InMemoryNonceReplayGuard(),
            now=FIXED_NOW,
        )
    assert verification.value.code == "unknown_key"

    contracts = fixture_bundle["contracts"]
    assert isinstance(contracts, dict)
    with pytest.raises(ContractEnvelopeError) as signing:
        sign_envelope(
            contracts["ExecutionIntent"],
            secret="",
            key_id="fixture-key-v1",
            issued_at="2026-07-29T12:00:00Z",
            expires_at="2026-07-29T13:00:00Z",
            nonce="nrcl-67-fixture-0002",
        )
    assert signing.value.code == "unknown_key"


def test_signing_rejects_an_unusable_lifetime(fixture_bundle: dict[str, object]) -> None:
    contracts = fixture_bundle["contracts"]
    assert isinstance(contracts, dict)

    with pytest.raises(ContractEnvelopeError) as failure:
        sign_envelope(
            contracts["ExecutionIntent"],
            secret=FIXTURE_SECRET,
            key_id="fixture-key-v1",
            issued_at="2026-07-29T13:00:00Z",
            expires_at="2026-07-29T13:00:00Z",
            nonce="nrcl-67-fixture-0003",
        )
    assert failure.value.code == "lifetime_invalid"


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
