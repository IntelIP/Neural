import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from scripts.validate_release import (
    ReleaseValidationError,
    classify_tag,
    project_version,
    validate_artifacts,
    validate_source,
)


def _uncontrolled_contact(suffix: str) -> str:
    return f"EVIL@NEURAL-SDK.{suffix}"


@pytest.mark.parametrize("tag", ["v0.4.2", "v1.0.0", "v10.20.30"])
def test_stable_tags_route_only_to_pypi(tag: str) -> None:
    assert classify_tag(tag, tag.removeprefix("v")) == "pypi"


@pytest.mark.parametrize(
    "tag",
    [
        "v0.4.2rc1",
        "v0.4.2dev1",
        "v0.4.2.dev1",
        "v0.4.2-alpha.1",
        "v0.4.2-beta.2",
        "v0.4.2-dev.3",
    ],
)
def test_prerelease_and_dev_tags_route_only_to_testpypi(tag: str) -> None:
    assert classify_tag(tag, tag.removeprefix("v")) == "testpypi"


@pytest.mark.parametrize(
    "tag",
    [
        "0.4.2",
        "v0.4",
        "v01.4.2",
        "v0.4.2-01",
        "v0.4.2+build",
        "v0.4.2.post1",
        "release-v0.4.2",
    ],
)
def test_invalid_tags_are_rejected(tag: str) -> None:
    with pytest.raises(ReleaseValidationError):
        classify_tag(tag, tag.removeprefix("v"))


@pytest.mark.parametrize("tag", ["v99.0.0", "v0.4.2rc1"])
def test_tag_must_match_project_version(tag: str) -> None:
    with pytest.raises(ReleaseValidationError, match="does not match project version"):
        classify_tag(tag, "0.4.2")


def test_project_version_reads_pyproject(tmp_path: Path) -> None:
    project_file = tmp_path / "pyproject.toml"
    project_file.write_text('[project]\nversion = "1.2.3rc1"\n', encoding="utf-8")

    assert project_version(project_file) == "1.2.3rc1"


def test_source_scan_is_recursive_and_ignores_test_fixtures(tmp_path: Path) -> None:
    public_doc = tmp_path / "docs" / "nested" / "contact.md"
    public_doc.parent.mkdir(parents=True)
    public_doc.write_text("Contact hudson@intelip.co", encoding="utf-8")

    intentional_fixture = tmp_path / "tests" / "fixtures" / "malicious.txt"
    intentional_fixture.parent.mkdir(parents=True)
    intentional_fixture.write_text(_uncontrolled_contact("DEV"), encoding="utf-8")

    validate_source(tmp_path)

    public_doc.write_text(_uncontrolled_contact("DEV"), encoding="utf-8")
    with pytest.raises(ReleaseValidationError, match="docs/nested/contact.md"):
        validate_source(tmp_path)


def _metadata(
    *,
    version: str = "0.4.2",
    author: str = "Hudson Aikins <hudson@intelip.co>",
    maintainer: str = "Advanced Intellectual Labs LLC <hudson@intelip.co>",
) -> bytes:
    return (
        f"Metadata-Version: 2.4\nVersion: {version}\n"
        f"Author-email: {author}\nMaintainer-email: {maintainer}\n"
    ).encode()


def _write_wheel(path: Path, content: bytes) -> None:
    with zipfile.ZipFile(path, mode="w") as archive:
        archive.writestr("neural_sdk-0.4.2.dist-info/METADATA", content)


def _write_sdist(path: Path, content: bytes) -> None:
    with tarfile.open(path, mode="w:gz") as archive:
        member = tarfile.TarInfo("neural_sdk-0.4.2/PKG-INFO")
        member.size = len(content)
        archive.addfile(member, io.BytesIO(content))


def test_clean_wheel_and_sdist_pass(tmp_path: Path) -> None:
    wheel = tmp_path / "neural_sdk-0.4.2-py3-none-any.whl"
    sdist = tmp_path / "neural_sdk-0.4.2.tar.gz"
    _write_wheel(wheel, _metadata())
    _write_sdist(sdist, _metadata())

    validate_artifacts([wheel, sdist], "0.4.2")


@pytest.mark.parametrize("artifact_type", ["wheel", "sdist"])
def test_mixed_case_uncontrolled_contacts_block_artifacts(
    tmp_path: Path, artifact_type: str
) -> None:
    wheel = tmp_path / "neural_sdk-0.4.2-py3-none-any.whl"
    sdist = tmp_path / "neural_sdk-0.4.2.tar.gz"
    _write_wheel(
        wheel,
        _metadata()
        + (
            f"Contact: {_uncontrolled_contact('DEV')}".encode() if artifact_type == "wheel" else b""
        ),
    )
    _write_sdist(
        sdist,
        _metadata()
        + (
            f"Contact: {_uncontrolled_contact('COM')}".encode() if artifact_type == "sdist" else b""
        ),
    )

    with pytest.raises(ReleaseValidationError, match="uncontrolled contact domains"):
        validate_artifacts([wheel, sdist], "0.4.2")


@pytest.mark.parametrize(
    ("wheel_metadata", "sdist_metadata", "match"),
    [
        (_metadata(version="9.9.9"), _metadata(), "Version"),
        (_metadata(author="Attacker <attacker@example.org>"), _metadata(), "Author-email"),
        (_metadata(), _metadata(maintainer="Attacker <attacker@example.org>"), "Maintainer-email"),
        (_metadata(author=""), _metadata(), "Author-email"),
    ],
)
def test_artifact_metadata_must_match_release_contract(
    tmp_path: Path,
    wheel_metadata: bytes,
    sdist_metadata: bytes,
    match: str,
) -> None:
    wheel = tmp_path / "neural_sdk-0.4.2-py3-none-any.whl"
    sdist = tmp_path / "neural_sdk-0.4.2.tar.gz"
    _write_wheel(wheel, wheel_metadata)
    _write_sdist(sdist, sdist_metadata)

    with pytest.raises(ReleaseValidationError, match=match):
        validate_artifacts([wheel, sdist], "0.4.2")


def test_publish_workflow_has_mutually_exclusive_uploads() -> None:
    workflow = Path(".github/workflows/publish.yml").read_text(encoding="utf-8")

    assert "steps.release.outputs.channel == 'pypi'" in workflow
    assert "steps.release.outputs.channel == 'testpypi'" in workflow
    assert workflow.count("twine upload") == 2
    assert workflow.index("validate-artifacts") < workflow.index("twine upload")
