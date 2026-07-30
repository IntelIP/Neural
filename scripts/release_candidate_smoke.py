#!/usr/bin/env python3
"""Verify that an installed Neural artifact exposes its declared version."""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

EXPECTED_DEMO_REPLAY_DIGEST = "569f87947428d4d093425134181b754ca974675aa7c74fc2cb73a8e193f5b4e6"


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


def validate_kernel_replay(result: dict[str, object]) -> None:
    """Validate the installed dependency-free replay contract."""
    digest = result.get("digest")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ReleaseSmokeError("kernel replay did not produce a SHA-256 digest")
    if digest != EXPECTED_DEMO_REPLAY_DIGEST:
        raise ReleaseSmokeError("kernel replay digest changed")
    if result.get("event_count") != 4:
        raise ReleaseSmokeError("kernel replay event count changed")
    if result.get("market_count") != 2:
        raise ReleaseSmokeError("kernel replay market count changed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-version", required=True)
    args = parser.parse_args()

    try:
        distribution_version = version("neural-sdk")
    except PackageNotFoundError as exc:
        raise ReleaseSmokeError("neural-sdk distribution is not installed") from exc

    import neural
    from neural.kernel import run_demo_replay

    validate_install(
        args.expected_version,
        distribution_version,
        getattr(neural, "__version__", None),
        Path(neural.__file__).resolve(),
    )
    validate_kernel_replay(run_demo_replay().as_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
