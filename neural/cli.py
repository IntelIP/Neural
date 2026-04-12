from __future__ import annotations

import argparse
import json
import platform
import sys
from collections.abc import Sequence
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from neural._version import __version__
from neural.auth.env import DEFAULT_API_KEY_PATH, DEFAULT_PRIVATE_KEY_PATH
from neural.auth.polymarket_us_env import (
    DEFAULT_API_KEY_PATH as DEFAULT_POLYMARKET_US_API_KEY_PATH,
)
from neural.auth.polymarket_us_env import (
    DEFAULT_API_SECRET_PATH as DEFAULT_POLYMARKET_US_API_SECRET_PATH,
)
from neural.auth.polymarket_us_env import (
    DEFAULT_PASSPHRASE_PATH as DEFAULT_POLYMARKET_US_PASSPHRASE_PATH,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neural",
        description="Neural SDK command-line interface.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print the installed Neural SDK version.",
    )

    subparsers = parser.add_subparsers(dest="command")

    doctor = subparsers.add_parser(
        "doctor",
        help="Inspect local Neural SDK environment and credential readiness.",
    )
    doctor.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of a text report.",
    )

    return parser


def _has_env_or_file(env_var: str, path: Path) -> bool:
    return bool(path.exists() or __import__("os").getenv(env_var))


def _doctor_report() -> dict[str, Any]:
    kalshi_key_ready = _has_env_or_file("KALSHI_API_KEY_ID", DEFAULT_API_KEY_PATH)
    kalshi_private_key_ready = _has_env_or_file(
        "KALSHI_PRIVATE_KEY_BASE64",
        DEFAULT_PRIVATE_KEY_PATH,
    ) or _has_env_or_file("KALSHI_PRIVATE_KEY_PATH", DEFAULT_PRIVATE_KEY_PATH)

    polymarket_key_ready = _has_env_or_file(
        "POLYMARKET_US_API_KEY",
        DEFAULT_POLYMARKET_US_API_KEY_PATH,
    ) or _has_env_or_file("POLYMARKET_US_API_KEY_PATH", DEFAULT_POLYMARKET_US_API_KEY_PATH)
    polymarket_secret_ready = _has_env_or_file(
        "POLYMARKET_US_API_SECRET_BASE64",
        DEFAULT_POLYMARKET_US_API_SECRET_PATH,
    ) or _has_env_or_file("POLYMARKET_US_API_SECRET_PATH", DEFAULT_POLYMARKET_US_API_SECRET_PATH)
    polymarket_passphrase_ready = _has_env_or_file(
        "POLYMARKET_US_API_PASSPHRASE",
        DEFAULT_POLYMARKET_US_PASSPHRASE_PATH,
    ) or _has_env_or_file(
        "POLYMARKET_US_API_PASSPHRASE_PATH",
        DEFAULT_POLYMARKET_US_PASSPHRASE_PATH,
    )

    return {
        "version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "credentials": {
            "kalshi": {
                "api_key": kalshi_key_ready,
                "private_key": kalshi_private_key_ready,
                "ready": kalshi_key_ready and kalshi_private_key_ready,
            },
            "polymarket_us": {
                "api_key": polymarket_key_ready,
                "api_secret": polymarket_secret_ready,
                "passphrase": polymarket_passphrase_ready,
                "ready": (
                    polymarket_key_ready
                    and polymarket_secret_ready
                    and polymarket_passphrase_ready
                ),
            },
        },
        "optional_dependencies": {
            "kalshi_python": find_spec("kalshi_python") is not None,
            "docker": find_spec("docker") is not None,
        },
    }


def _render_doctor_text(report: dict[str, Any]) -> str:
    kalshi = report["credentials"]["kalshi"]
    polymarket = report["credentials"]["polymarket_us"]
    deps = report["optional_dependencies"]
    lines = [
        f"Neural SDK {report['version']}",
        f"Python: {report['python']}",
        f"Platform: {report['platform']}",
        "",
        "Credentials:",
        f"  Kalshi: {'ready' if kalshi['ready'] else 'not ready'} "
        f"(api_key={kalshi['api_key']}, private_key={kalshi['private_key']})",
        f"  Polymarket US: {'ready' if polymarket['ready'] else 'not ready'} "
        f"(api_key={polymarket['api_key']}, api_secret={polymarket['api_secret']}, "
        f"passphrase={polymarket['passphrase']})",
        "",
        "Optional dependencies:",
        f"  kalshi_python: {deps['kalshi_python']}",
        f"  docker: {deps['docker']}",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(__version__)
        return 0

    if args.command == "doctor":
        report = _doctor_report()
        if args.json:
            print(json.dumps(report, sort_keys=True))
        else:
            print(_render_doctor_text(report))
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
