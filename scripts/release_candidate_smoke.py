#!/usr/bin/env python3
"""Verify that an installed Neural artifact exposes its declared version."""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


class ReleaseSmokeError(AssertionError):
    """An installed release artifact failed its smoke contract."""


def validate_install(
    expected_version: str,
    distribution_version: str,
    module_version: str | None,
    module_file: Path,
) -> None:
    """Validate version agreement and prove import came from an installed artifact."""
    if distribution_version != expected_version:
        raise ReleaseSmokeError(
            f"distribution version {distribution_version!r} != expected {expected_version!r}"
        )
    if module_version != expected_version:
        raise ReleaseSmokeError(
            f"neural.__version__ {module_version!r} != expected {expected_version!r}"
        )
    if not {"site-packages", "dist-packages"}.intersection(module_file.parts):
        raise ReleaseSmokeError(f"neural imported outside an installed package: {module_file}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-version", required=True)
    args = parser.parse_args()

    try:
        distribution_version = version("neural-sdk")
    except PackageNotFoundError as exc:
        raise ReleaseSmokeError("neural-sdk distribution is not installed") from exc

    import neural

    validate_install(
        args.expected_version,
        distribution_version,
        getattr(neural, "__version__", None),
        Path(neural.__file__).resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
