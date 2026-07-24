#!/usr/bin/env python3
"""Trusted validation for immutable Neural release candidates."""

from __future__ import annotations

import argparse
import re
import tarfile
import zipfile
from email.parser import BytesParser
from email.policy import default
from pathlib import Path
from typing import BinaryIO

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised by Python 3.10 CI
    import tomli as tomllib
from packaging.utils import (
    InvalidSdistFilename,
    InvalidWheelFilename,
    canonicalize_name,
    parse_sdist_filename,
    parse_wheel_filename,
)
from packaging.version import InvalidVersion, Version

_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_BLOCKED_DOMAINS = tuple(f"neural-sdk.{suffix}".encode("ascii") for suffix in ("dev", "com"))
_EXPECTED_METADATA = {
    "Name": "neural-sdk",
    "Author-email": "Hudson Aikins <hudson@intelip.co>",
    "Maintainer-email": "Advanced Intellectual Labs LLC <hudson@intelip.co>",
}
_IGNORED_SOURCE_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "tests",
    "venv",
}


class ReleaseCandidateError(ValueError):
    """An immutable release candidate failed a deterministic safety check."""


def validate_candidate_sha(value: str) -> str:
    """Require an immutable full Git commit SHA."""
    if not _COMMIT_SHA.fullmatch(value):
        raise ReleaseCandidateError("candidate_sha must be a lowercase 40-character commit SHA")
    return value


def project_version(project_file: Path) -> str:
    """Read the package version used to build release artifacts."""
    try:
        with project_file.open("rb") as stream:
            version = tomllib.load(stream)["project"]["version"]
    except (OSError, KeyError, tomllib.TOMLDecodeError) as exc:
        raise ReleaseCandidateError(
            f"cannot read project version from {project_file}: {exc}"
        ) from exc
    if not isinstance(version, str) or not version:
        raise ReleaseCandidateError(f"invalid project version in {project_file}")
    try:
        return str(Version(version))
    except InvalidVersion as exc:
        raise ReleaseCandidateError(
            f"invalid PEP 440 project version in {project_file}: {version!r}"
        ) from exc


def _blocked_domains_in_stream(stream: BinaryIO) -> set[str]:
    found: set[str] = set()
    overlap = max(map(len, _BLOCKED_DOMAINS)) - 1
    tail = b""

    while chunk := stream.read(64 * 1024):
        normalized = (tail + chunk).lower()
        found.update(domain.decode("ascii") for domain in _BLOCKED_DOMAINS if domain in normalized)
        tail = normalized[-overlap:]

    return found


def _source_files(root: Path):
    for path in root.rglob("*"):
        if path.is_symlink():
            relative = path.relative_to(root)
            raise ReleaseCandidateError(f"source symlink is not allowed: {relative}")
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in _IGNORED_SOURCE_PARTS for part in relative.parts):
            continue
        yield relative, path


def validate_source(root: Path) -> None:
    """Reject uncontrolled contact domains from public source surfaces."""
    if root.is_symlink() or not root.is_dir():
        raise ReleaseCandidateError(f"candidate source must be a directory: {root}")

    violations = []
    for relative, path in _source_files(root):
        with path.open("rb") as stream:
            domains = _blocked_domains_in_stream(stream)
        violations.extend(f"{relative}: {domain}" for domain in sorted(domains))

    if violations:
        raise ReleaseCandidateError(
            "uncontrolled contact domains in source: " + ", ".join(violations)
        )


def _validate_wheel(path: Path) -> list[str]:
    violations = []
    try:
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                with archive.open(member) as stream:
                    domains = _blocked_domains_in_stream(stream)
                violations.extend(
                    f"{path.name}:{member.filename}: {domain}" for domain in sorted(domains)
                )
    except (OSError, zipfile.BadZipFile) as exc:
        raise ReleaseCandidateError(f"invalid wheel {path}: {exc}") from exc
    return violations


def _validate_sdist(path: Path) -> list[str]:
    violations = []
    try:
        with tarfile.open(path, mode="r:*") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                stream = archive.extractfile(member)
                if stream is None:
                    continue
                with stream:
                    domains = _blocked_domains_in_stream(stream)
                violations.extend(
                    f"{path.name}:{member.name}: {domain}" for domain in sorted(domains)
                )
    except (OSError, tarfile.TarError) as exc:
        raise ReleaseCandidateError(f"invalid sdist {path}: {exc}") from exc
    return violations


def _wheel_metadata(path: Path, expected_version: str) -> bytes:
    expected_path = f"neural_sdk-{expected_version}.dist-info/METADATA"
    try:
        with zipfile.ZipFile(path) as archive:
            names = [
                member.filename
                for member in archive.infolist()
                if not member.is_dir() and member.filename.endswith(".dist-info/METADATA")
            ]
            if len(names) != 1:
                raise ReleaseCandidateError(
                    f"expected one wheel METADATA file in {path}; found {len(names)}"
                )
            if names[0] != expected_path:
                raise ReleaseCandidateError(
                    f"invalid wheel metadata path in {path.name}: "
                    f"expected {expected_path!r}, found {names[0]!r}"
                )
            return archive.read(names[0])
    except (OSError, zipfile.BadZipFile) as exc:
        raise ReleaseCandidateError(f"invalid wheel {path}: {exc}") from exc


def _sdist_metadata(path: Path, expected_version: str) -> bytes:
    expected_path = f"neural_sdk-{expected_version}/PKG-INFO"
    try:
        with tarfile.open(path, mode="r:*") as archive:
            members = [
                member
                for member in archive
                if member.isfile()
                and member.name.endswith("/PKG-INFO")
                and member.name.count("/") == 1
            ]
            if len(members) != 1:
                raise ReleaseCandidateError(
                    f"expected one top-level sdist PKG-INFO file in {path}; found {len(members)}"
                )
            if members[0].name != expected_path:
                raise ReleaseCandidateError(
                    f"invalid sdist metadata path in {path.name}: "
                    f"expected {expected_path!r}, found {members[0].name!r}"
                )
            stream = archive.extractfile(members[0])
            if stream is None:
                raise ReleaseCandidateError(f"cannot read sdist metadata from {path}")
            with stream:
                return stream.read()
    except (OSError, tarfile.TarError) as exc:
        raise ReleaseCandidateError(f"invalid sdist {path}: {exc}") from exc


def _validate_core_metadata(path: Path, content: bytes, expected_version: str) -> None:
    metadata = BytesParser(policy=default).parsebytes(content)
    expected = {"Version": expected_version, **_EXPECTED_METADATA}
    mismatches = [
        f"{header}: expected {value!r}, found {metadata.get(header)!r}"
        for header, value in expected.items()
        if metadata.get(header) != value
    ]
    if mismatches:
        raise ReleaseCandidateError(
            f"invalid core metadata in {path.name}: " + "; ".join(mismatches)
        )


def _validate_artifact_identity(path: Path, expected_version: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise ReleaseCandidateError(f"release artifact must be a regular file: {path}")

    try:
        if path.suffix == ".whl":
            name, version, _, _ = parse_wheel_filename(path.name)
            artifact_type = "wheel"
        elif path.name.endswith((".tar.gz", ".tgz")):
            name, version = parse_sdist_filename(path.name)
            artifact_type = "sdist"
        else:
            raise ReleaseCandidateError(f"unsupported release artifact: {path}")
    except (InvalidWheelFilename, InvalidSdistFilename) as exc:
        raise ReleaseCandidateError(
            f"invalid release artifact filename {path.name}: {exc}"
        ) from exc

    if canonicalize_name(name) != canonicalize_name("neural-sdk"):
        raise ReleaseCandidateError(
            f"invalid distribution name in {path.name}: expected 'neural-sdk', found {name!r}"
        )
    if version != Version(expected_version):
        raise ReleaseCandidateError(
            f"invalid version in {path.name}: expected {expected_version!r}, found {str(version)!r}"
        )
    return artifact_type


def validate_artifacts(paths: list[Path], expected_version: str) -> None:
    """Require one wheel and one sdist with approved contacts and version."""
    violations = []
    wheels = []
    sdists = []

    for path in paths:
        artifact_type = _validate_artifact_identity(path, expected_version)
        if artifact_type == "wheel":
            wheels.append(path)
            violations.extend(_validate_wheel(path))
        else:
            sdists.append(path)
            violations.extend(_validate_sdist(path))

    if len(wheels) != 1 or len(sdists) != 1:
        raise ReleaseCandidateError(
            f"expected one wheel and one sdist; found {len(wheels)} wheel(s) "
            f"and {len(sdists)} sdist(s)"
        )
    if violations:
        raise ReleaseCandidateError(
            "uncontrolled contact domains in artifacts: " + ", ".join(violations)
        )
    _validate_core_metadata(
        wheels[0], _wheel_metadata(wheels[0], expected_version), expected_version
    )
    _validate_core_metadata(
        sdists[0], _sdist_metadata(sdists[0], expected_version), expected_version
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    sha_parser = subparsers.add_parser("validate-sha")
    sha_parser.add_argument("candidate_sha")

    version_parser = subparsers.add_parser("project-version")
    version_parser.add_argument("project_file", type=Path)

    source_parser = subparsers.add_parser("validate-source")
    source_parser.add_argument("root", type=Path)

    artifact_parser = subparsers.add_parser("validate-artifacts")
    artifact_parser.add_argument("artifacts", nargs="+", type=Path)
    artifact_parser.add_argument("--project-file", type=Path, required=True)

    args = parser.parse_args()
    try:
        if args.command == "validate-sha":
            print(validate_candidate_sha(args.candidate_sha))
        elif args.command == "project-version":
            print(project_version(args.project_file))
        elif args.command == "validate-source":
            validate_source(args.root)
        else:
            validate_artifacts(args.artifacts, project_version(args.project_file))
    except ReleaseCandidateError as exc:
        parser.exit(2, f"release candidate validation failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
