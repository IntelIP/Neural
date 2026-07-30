import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from scripts.release_candidate_smoke import (
    EXPECTED_DEMO_REPLAY_DIGEST,
    ReleaseSmokeError,
    validate_install,
    validate_kernel_replay,
)
from scripts.validate_release_candidate import (
    ReleaseCandidateError,
    project_version,
    validate_artifacts,
    validate_candidate_sha,
    validate_source,
)


def _uncontrolled_contact(suffix: str) -> str:
    return f"EVIL@NEURAL-SDK.{suffix}"


def _metadata(
    *,
    name: str = "neural-sdk",
    version: str = "0.4.2",
    author: str = "Hudson Aikins <hudson@intelip.co>",
    maintainer: str = "Advanced Intellectual Labs LLC <hudson@intelip.co>",
) -> bytes:
    return (
        f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n"
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


def test_candidate_sha_requires_immutable_full_commit() -> None:
    sha = "7aa6c856f5ad18b42375254fe8b52bc50cc868d4"
    assert validate_candidate_sha(sha) == sha

    for invalid in ("main", "7aa6c85", sha.upper(), f"{sha}0"):
        with pytest.raises(ReleaseCandidateError):
            validate_candidate_sha(invalid)


def test_project_version_reads_pyproject(tmp_path: Path) -> None:
    project_file = tmp_path / "pyproject.toml"
    project_file.write_text('[project]\nversion = "1.2.3rc1"\n', encoding="utf-8")

    assert project_version(project_file) == "1.2.3rc1"


def test_project_version_rejects_non_pep440_shell_content(tmp_path: Path) -> None:
    project_file = tmp_path / "pyproject.toml"
    project_file.write_text('[project]\nversion = "\\"; echo PWNED; #"\n', encoding="utf-8")

    with pytest.raises(ReleaseCandidateError, match="invalid PEP 440"):
        project_version(project_file)


def test_source_scan_is_case_insensitive_and_ignores_tests(tmp_path: Path) -> None:
    public_doc = tmp_path / "docs" / "contact.md"
    public_doc.parent.mkdir()
    public_doc.write_text("Contact hudson@intelip.co", encoding="utf-8")

    fixture = tmp_path / "tests" / "malicious.txt"
    fixture.parent.mkdir()
    fixture.write_text(_uncontrolled_contact("DEV"), encoding="utf-8")

    validate_source(tmp_path)

    public_doc.write_text(_uncontrolled_contact("COM"), encoding="utf-8")
    with pytest.raises(ReleaseCandidateError, match="docs/contact.md"):
        validate_source(tmp_path)


def test_source_scan_rejects_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("safe", encoding="utf-8")
    (tmp_path / "public-link.txt").symlink_to(target)

    with pytest.raises(ReleaseCandidateError, match="source symlink is not allowed"):
        validate_source(tmp_path)


def test_source_scan_rejects_missing_root(tmp_path: Path) -> None:
    with pytest.raises(ReleaseCandidateError, match="candidate source must be a directory"):
        validate_source(tmp_path / "missing")


def test_clean_wheel_and_sdist_pass(tmp_path: Path) -> None:
    wheel = tmp_path / "neural_sdk-0.4.2-py3-none-any.whl"
    sdist = tmp_path / "neural_sdk-0.4.2.tar.gz"
    _write_wheel(wheel, _metadata())
    _write_sdist(sdist, _metadata())

    validate_artifacts([wheel, sdist], "0.4.2")


def test_artifact_identity_requires_neural_distribution_name(tmp_path: Path) -> None:
    wheel = tmp_path / "attacker-0.4.2-py3-none-any.whl"
    sdist = tmp_path / "attacker-0.4.2.tar.gz"
    _write_wheel(wheel, _metadata(name="attacker"))
    _write_sdist(sdist, _metadata(name="attacker"))

    with pytest.raises(ReleaseCandidateError, match="invalid distribution name"):
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

    with pytest.raises(ReleaseCandidateError, match="uncontrolled contact domains"):
        validate_artifacts([wheel, sdist], "0.4.2")


@pytest.mark.parametrize(
    ("wheel_metadata", "sdist_metadata", "match"),
    [
        (_metadata(version="9.9.9"), _metadata(), "Version"),
        (_metadata(author="Attacker <attacker@example.org>"), _metadata(), "Author-email"),
        (_metadata(), _metadata(maintainer="Attacker <attacker@example.org>"), "Maintainer-email"),
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

    with pytest.raises(ReleaseCandidateError, match=match):
        validate_artifacts([wheel, sdist], "0.4.2")


def test_installed_smoke_requires_version_agreement_and_installed_path() -> None:
    validate_install(
        "0.4.2",
        "0.4.2",
        "0.4.2",
        Path("/tmp/venv/lib/python3.11/site-packages/neural/__init__.py"),
    )

    with pytest.raises(ReleaseSmokeError, match="distribution version"):
        validate_install(
            "0.4.2",
            "0.4.1",
            "0.4.2",
            Path("/tmp/venv/lib/python3.11/site-packages/neural/__init__.py"),
        )
    with pytest.raises(ReleaseSmokeError, match="installed package"):
        validate_install(
            "0.4.2",
            "0.4.2",
            "0.4.2",
            Path("/workspace/neural/__init__.py"),
        )


def test_installed_smoke_requires_deterministic_kernel_replay() -> None:
    validate_kernel_replay(
        {
            "digest": EXPECTED_DEMO_REPLAY_DIGEST,
            "event_count": 4,
            "market_count": 2,
            "snapshot": [
                {
                    "market_id": "KX-DEMO-NO",
                    "no_price": "0.42",
                    "observed_at": "2026-07-29T12:01:00Z",
                    "source": "kalshi-fixture",
                    "volume": "9",
                    "yes_price": "0.58",
                },
                {
                    "market_id": "KX-DEMO-YES",
                    "no_price": "0.48",
                    "observed_at": "2026-07-29T12:01:00Z",
                    "source": "kalshi-fixture",
                    "volume": "14",
                    "yes_price": "0.52",
                },
            ],
        }
    )

    with pytest.raises(ReleaseSmokeError, match="digest"):
        validate_kernel_replay({"digest": "", "event_count": 4, "market_count": 2})
    with pytest.raises(ReleaseSmokeError, match="digest changed"):
        validate_kernel_replay({"digest": "a" * 64, "event_count": 4, "market_count": 2})
    with pytest.raises(ReleaseSmokeError, match="event count"):
        validate_kernel_replay(
            {
                "digest": EXPECTED_DEMO_REPLAY_DIGEST,
                "event_count": 3,
                "market_count": 2,
            }
        )
    with pytest.raises(ReleaseSmokeError, match="snapshot"):
        validate_kernel_replay(
            {
                "digest": EXPECTED_DEMO_REPLAY_DIGEST,
                "event_count": 4,
                "market_count": 2,
                "snapshot": [],
            }
        )


def test_workflow_is_read_only_non_publishing_and_sha_bound() -> None:
    workflow = Path(".github/workflows/release-candidate-validation.yml").read_text(
        encoding="utf-8"
    )

    assert "workflow_dispatch:" in workflow
    assert "candidate_sha:" in workflow
    assert "permissions:\n  contents: read" in workflow
    assert 'test "${WORKFLOW_REF}" = "refs/heads/${DEFAULT_BRANCH}"' in workflow
    assert workflow.count("ref: ${{ github.workflow_sha }}") == 3
    assert workflow.count("persist-credentials: false") == 5
    assert 'test "${actual_sha}" = "${CANDIDATE_SHA}"' in workflow
    assert workflow.count("CANDIDATE_SHA: ${{ inputs.candidate_sha }}") == 5
    assert '"${{ inputs.candidate_sha }}"' not in workflow
    assert workflow.count("--no-deps") == 2
    assert "needs: [authorize, build]" in workflow
    assert "needs: [authorize, validate, wheel-smoke, sdist-smoke]" in workflow
    assert workflow.index("Upload fully validated artifacts and digests") > workflow.index(
        "Validate installed source distribution"
    )
    assert "name: neural-release-contract-${{ inputs.candidate_sha }}" in workflow
    assert "name: neural-release-validated-${{ inputs.candidate_sha }}" in workflow
    assert "uses: actions/checkout@v4" not in workflow
    assert "uses: actions/setup-python@v5" not in workflow
    assert "uses: actions/upload-artifact@v4" not in workflow
    assert "uses: actions/download-artifact@v4" not in workflow
    assert "twine upload" not in workflow
    assert "PYPI_API_TOKEN" not in workflow
    assert "TESTPYPI_API_TOKEN" not in workflow
    assert "deploy" not in workflow.lower()
