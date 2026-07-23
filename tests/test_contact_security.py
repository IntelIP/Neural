from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTACT_SURFACES = (
    "pyproject.toml",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "docs/mint.json",
    "docs/openapi/authentication-schemes.yaml",
    "docs/openapi/data-collection-apis.yaml",
    "docs/openapi/data-models.yaml",
    "docs/openapi/fix-protocol.yaml",
    "docs/openapi/kalshi-trading-api.yaml",
    "docs/openapi/websocket-api.yaml",
    "scripts/generate_openapi_specs.py",
)
UNCONTROLLED_CONTACT_DOMAINS = ("neural-sdk.dev", "neural-sdk.com")


def _find_uncontrolled_domains(relative_path: str, content: str) -> list[str]:
    normalized_content = content.casefold()
    return [
        f"{relative_path}: {domain}"
        for domain in UNCONTROLLED_CONTACT_DOMAINS
        if domain.casefold() in normalized_content
    ]


def test_public_contact_surfaces_do_not_use_uncontrolled_domains() -> None:
    violations = []

    for relative_path in CONTACT_SURFACES:
        content = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        violations.extend(_find_uncontrolled_domains(relative_path, content))

    assert not violations, "Uncontrolled contact domains found: " + ", ".join(violations)


def test_uncontrolled_domain_detection_is_case_insensitive() -> None:
    assert _find_uncontrolled_domains("synthetic.txt", "Contact EVIL@NEURAL-SDK.DEV") == [
        "synthetic.txt: neural-sdk.dev"
    ]


def test_package_metadata_uses_accountable_maintainer_contact() -> None:
    metadata = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert '{ name = "Hudson Aikins", email = "hudson@intelip.co" }' in metadata
    assert '{ name = "Neural Contributors" }' in metadata
    assert '{ name = "Advanced Intellectual Labs LLC", email = "hudson@intelip.co" }' in metadata
