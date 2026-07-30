from __future__ import annotations

import json
import subprocess
import sys

import pytest

from neural.kernel import CapabilityStatus, capability_matrix, get_capability


def test_capability_matrix_is_unique_and_has_stable_kernel() -> None:
    matrix = capability_matrix()
    names = [item["name"] for item in matrix]

    assert len(names) == len(set(names))
    assert get_capability("kernel.replay").status is CapabilityStatus.STABLE
    assert get_capability("deployment").status is CapabilityStatus.DEPRECATED


def test_unknown_capability_fails_clearly() -> None:
    with pytest.raises(KeyError, match="Unknown Neural capability"):
        get_capability("missing")


def test_stable_kernel_import_does_not_load_optional_stacks() -> None:
    code = """
import json
import sys
import neural
from neural.kernel import run_demo_replay
run_demo_replay()
blocked = [
    name for name in (
        "aiohttp", "docker", "jinja2", "numpy", "pandas", "plotly",
        "pydantic", "requests", "simplefix", "sqlalchemy", "torch",
        "transformers", "websocket", "websockets"
    )
    if name in sys.modules
]
print(json.dumps(blocked))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == []
