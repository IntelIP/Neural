"""Schema-derived JSON Schema and Pydantic contract validators."""

from __future__ import annotations

import json
from copy import deepcopy
from functools import cache, lru_cache
from importlib import resources
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
from pydantic import RootModel, model_validator

CONTRACT_VERSION = "1.0.0"
CONTRACT_NAMES = (
    "MarketSnapshot",
    "ResearchEvidenceRef",
    "ExecutionIntent",
    "RiskDecision",
    "PaperOrder",
    "PostTradeReview",
)
_ENVELOPE_NAME = "SignedContractEnvelope"
_SCHEMA_RESOURCE = "schemas/neural-contracts-v1.schema.json"


class ContractValidationError(ValueError):
    """A payload failed its authoritative Neural JSON Schema."""

    def __init__(self, code: str, details: list[str]) -> None:
        self.code = code
        self.details = tuple(details)
        super().__init__(f"{code}: {'; '.join(details)}")


@lru_cache(maxsize=1)
def load_contract_bundle() -> dict[str, Any]:
    """Load and validate the packaged authoritative schema bundle."""
    content = resources.files(__package__).joinpath(_SCHEMA_RESOURCE).read_text(encoding="utf-8")
    bundle = json.loads(content)
    Draft202012Validator.check_schema(bundle)
    return bundle


def schema_for(name: str) -> dict[str, Any]:
    """Return a self-contained schema for one named contract or envelope."""
    bundle = load_contract_bundle()
    definitions = bundle.get("$defs", {})
    if name not in definitions or name not in (*CONTRACT_NAMES, _ENVELOPE_NAME):
        raise ContractValidationError("unsupported_contract", [name])
    return {
        "$schema": bundle["$schema"],
        "$defs": deepcopy(definitions),
        "$ref": f"#/$defs/{name}",
    }


@cache
def _validator(name: str) -> Draft202012Validator:
    return Draft202012Validator(schema_for(name), format_checker=FormatChecker())


def validate_json_schema(name: str, payload: Any) -> dict[str, Any]:
    """Validate with the authoritative JSON Schema and return a detached mapping."""
    errors = sorted(_validator(name).iter_errors(payload), key=lambda error: list(error.path))
    if errors:
        details = [
            f"{'.'.join(str(part) for part in error.absolute_path) or '$'}: {error.message}"
            for error in errors
        ]
        raise ContractValidationError("schema_invalid", details)
    if not isinstance(payload, dict):
        raise ContractValidationError("schema_invalid", ["$: contract must be an object"])
    return deepcopy(payload)


def _build_model(name: str) -> type[RootModel[dict[str, Any]]]:
    class GeneratedContractModel(RootModel[dict[str, Any]]):
        @model_validator(mode="after")
        def validate_authoritative_schema(self) -> Any:
            validate_json_schema(name, self.root)
            return self

    GeneratedContractModel.__name__ = name
    GeneratedContractModel.__qualname__ = name
    GeneratedContractModel.__doc__ = (
        f"Pydantic validator generated from the authoritative {name} JSON Schema."
    )
    return GeneratedContractModel


MarketSnapshot = _build_model("MarketSnapshot")
ResearchEvidenceRef = _build_model("ResearchEvidenceRef")
ExecutionIntent = _build_model("ExecutionIntent")
RiskDecision = _build_model("RiskDecision")
PaperOrder = _build_model("PaperOrder")
PostTradeReview = _build_model("PostTradeReview")

_MODELS = {
    "MarketSnapshot": MarketSnapshot,
    "ResearchEvidenceRef": ResearchEvidenceRef,
    "ExecutionIntent": ExecutionIntent,
    "RiskDecision": RiskDecision,
    "PaperOrder": PaperOrder,
    "PostTradeReview": PostTradeReview,
}


def contract_model(name: str) -> type[RootModel[dict[str, Any]]]:
    """Return the Pydantic validator generated for a supported contract."""
    try:
        return _MODELS[name]
    except KeyError as exc:
        raise ContractValidationError("unsupported_contract", [name]) from exc


def validate_contract(payload: Any) -> dict[str, Any]:
    """Reject unknown versions, then validate through generated Pydantic."""
    if not isinstance(payload, dict):
        raise ContractValidationError("schema_invalid", ["$: contract must be an object"])
    name = payload.get("schemaName")
    version = payload.get("schemaVersion")
    if name not in CONTRACT_NAMES:
        raise ContractValidationError("unsupported_contract", [str(name)])
    if version != CONTRACT_VERSION:
        raise ContractValidationError(
            "unsupported_version",
            [f"{name} version {version!r}; expected {CONTRACT_VERSION!r}"],
        )
    model = contract_model(str(name))
    return model.model_validate(payload).root
