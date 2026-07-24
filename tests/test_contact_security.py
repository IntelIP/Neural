from pathlib import Path

from scripts.validate_release import validate_source

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_public_source_does_not_use_uncontrolled_domains() -> None:
    validate_source(REPO_ROOT)


def test_package_metadata_uses_accountable_maintainer_contact() -> None:
    metadata = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert '{ name = "Hudson Aikins", email = "hudson@intelip.co" }' in metadata
    assert '{ name = "Neural Contributors" }' in metadata
    assert '{ name = "Advanced Intellectual Labs LLC", email = "hudson@intelip.co" }' in metadata
