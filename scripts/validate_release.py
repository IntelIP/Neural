#!/usr/bin/env python3
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

_VERSION_CORE = r"(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)"
_STABLE_VERSION = re.compile(rf"^{_VERSION_CORE}$")
_TEST_VERSION = re.compile(
    rf"^{_VERSION_CORE}(?:(?:a|b|rc)\d+|(?:\.?dev)\d+" rf"|-(?:alpha|beta|rc|dev)\.(?:0|[1-9]\d*))$"
)
_VERSION_ALIAS = re.compile(
    rf"^(?P<core>{_VERSION_CORE})-(?P<label>alpha|beta|rc|dev)\.(?P<number>\d+)$"
)
_DEV_WITHOUT_DOT = re.compile(rf"^(?P<core>{_VERSION_CORE})dev(?P<number>\d+)$")

_BLOCKED_DOMAINS = tuple(f"neural-sdk.{suffix}".encode("ascii") for suffix in ("dev", "com"))
_EXPECTED_METADATA = {
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


class ReleaseValidationError(ValueError):
    """Release input failed a deterministic safety check."""


def normalized_artifact_version(version: str) -> str:
    """Return the PEP 440 version setuptools writes into artifact metadata."""
    if alias := _VERSION_ALIAS.fullmatch(version):
        label = {"alpha": "a", "beta": "b", "rc": "rc", "dev": ".dev"}[alias.group("label")]
        return f"{alias.group('core')}{label}{alias.group('number')}"
    if dev := _DEV_WITHOUT_DOT.fullmatch(version):
        return f"{dev.group('core')}.dev{dev.group('number')}"
    return version


def project_version(project_file: Path) -> str:
    """Read the package version used to build release artifacts."""
    try:
        with project_file.open("rb") as stream:
            version = tomllib.load(stream)["project"]["version"]
    except (OSError, KeyError, tomllib.TOMLDecodeError) as exc:
        raise ReleaseValidationError(
            f"cannot read project version from {project_file}: {exc}"
        ) from exc
    if not isinstance(version, str) or not version:
        raise ReleaseValidationError(f"invalid project version in {project_file}")
    return version


def classify_tag(tag: str, expected_version: str) -> str:
    """Return the only package index allowed for a release tag."""
    if tag != f"v{expected_version}":
        raise ReleaseValidationError(
            f"release tag {tag!r} does not match project version {expected_version!r}"
        )
    if _STABLE_VERSION.fullmatch(expected_version):
        return "pypi"
    if _TEST_VERSION.fullmatch(expected_version):
        return "testpypi"
    raise ReleaseValidationError(f"invalid project release version: {expected_version!r}")


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
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in _IGNORED_SOURCE_PARTS for part in relative.parts):
            continue
        yield relative, path


def validate_source(root: Path) -> None:
    """Reject uncontrolled contact domains from public source surfaces."""
    violations = []
    for relative, path in _source_files(root):
        with path.open("rb") as stream:
            domains = _blocked_domains_in_stream(stream)
        violations.extend(f"{relative}: {domain}" for domain in sorted(domains))

    if violations:
        raise ReleaseValidationError(
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
        raise ReleaseValidationError(f"invalid wheel {path}: {exc}") from exc
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
        raise ReleaseValidationError(f"invalid sdist {path}: {exc}") from exc
    return violations


def _wheel_metadata(path: Path) -> bytes:
    try:
        with zipfile.ZipFile(path) as archive:
            names = [
                member.filename
                for member in archive.infolist()
                if not member.is_dir() and member.filename.endswith(".dist-info/METADATA")
            ]
            if len(names) != 1:
                raise ReleaseValidationError(
                    f"expected one wheel METADATA file in {path}; found {len(names)}"
                )
            return archive.read(names[0])
    except (OSError, zipfile.BadZipFile) as exc:
        raise ReleaseValidationError(f"invalid wheel {path}: {exc}") from exc


def _sdist_metadata(path: Path) -> bytes:
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
                raise ReleaseValidationError(
                    f"expected one top-level sdist PKG-INFO file in {path}; found {len(members)}"
                )
            stream = archive.extractfile(members[0])
            if stream is None:
                raise ReleaseValidationError(f"cannot read sdist metadata from {path}")
            with stream:
                return stream.read()
    except (OSError, tarfile.TarError) as exc:
        raise ReleaseValidationError(f"invalid sdist {path}: {exc}") from exc


def _validate_core_metadata(path: Path, content: bytes, expected_version: str) -> None:
    metadata = BytesParser(policy=default).parsebytes(content)
    expected = {"Version": normalized_artifact_version(expected_version), **_EXPECTED_METADATA}
    mismatches = [
        f"{header}: expected {value!r}, found {metadata.get(header)!r}"
        for header, value in expected.items()
        if metadata.get(header) != value
    ]
    if mismatches:
        raise ReleaseValidationError(
            f"invalid core metadata in {path.name}: " + "; ".join(mismatches)
        )


def validate_artifacts(paths: list[Path], expected_version: str) -> None:
    """Require one wheel and one sdist, both free of uncontrolled contacts."""
    violations = []
    wheels = []
    sdists = []

    for path in paths:
        if path.suffix == ".whl":
            wheels.append(path)
            violations.extend(_validate_wheel(path))
        elif path.name.endswith((".tar.gz", ".tgz")):
            sdists.append(path)
            violations.extend(_validate_sdist(path))
        else:
            raise ReleaseValidationError(f"unsupported release artifact: {path}")

    if len(wheels) != 1 or len(sdists) != 1:
        raise ReleaseValidationError(
            f"expected one wheel and one sdist; found {len(wheels)} wheel(s) "
            f"and {len(sdists)} sdist(s)"
        )
    if violations:
        raise ReleaseValidationError(
            "uncontrolled contact domains in artifacts: " + ", ".join(violations)
        )
    _validate_core_metadata(wheels[0], _wheel_metadata(wheels[0]), expected_version)
    _validate_core_metadata(sdists[0], _sdist_metadata(sdists[0]), expected_version)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Neural release inputs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    classify_parser = subparsers.add_parser("classify-tag")
    classify_parser.add_argument("tag")
    classify_parser.add_argument("--project-file", type=Path, default=Path("pyproject.toml"))

    source_parser = subparsers.add_parser("validate-source")
    source_parser.add_argument("root", type=Path)

    artifact_parser = subparsers.add_parser("validate-artifacts")
    artifact_parser.add_argument("artifacts", nargs="+", type=Path)
    artifact_parser.add_argument("--project-file", type=Path, default=Path("pyproject.toml"))

    args = parser.parse_args()
    try:
        if args.command == "classify-tag":
            print(f"channel={classify_tag(args.tag, project_version(args.project_file))}")
        elif args.command == "validate-source":
            validate_source(args.root)
        else:
            validate_artifacts(args.artifacts, project_version(args.project_file))
    except ReleaseValidationError as exc:
        parser.exit(2, f"release validation failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
