import datetime
from collections.abc import Callable
from typing import Any


class DataTransformer:
    """Transforms raw data from sources into normalized format for analysis."""

    def __init__(
        self, transformations: list[Callable[[dict[str, Any]], dict[str, Any]]] | None = None
    ):
        self.transformations = transformations or []

    def add_transformation(self, func: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        """Add a transformation function."""
        self.transformations.append(func)

    def transform(self, data: dict[str, Any]) -> dict[str, Any]:
        """Apply all transformations to the data."""
        for transform in self.transformations:
            data = transform(data)
        # Add timestamp if not present
        if "timestamp" not in data:
            data["timestamp"] = datetime.datetime.utcnow().isoformat()
        return data

    @staticmethod
    def flatten_keys(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
        """Flatten nested dict keys."""
        flattened = {}
        for key, value in data.items():
            new_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                flattened.update(DataTransformer.flatten_keys(value, new_key))
            else:
                flattened[new_key] = value
        return flattened

    @staticmethod
    def normalize_types(data: dict[str, Any]) -> dict[str, Any]:
        """Normalize data types (e.g., strings to numbers where possible)."""
        normalized: dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(value, str):
                try:
                    # Try int
                    normalized[key] = int(value)
                    continue
                except ValueError:
                    pass
                try:
                    # Try float
                    normalized[key] = float(value)
                    continue
                except ValueError:
                    pass
            normalized[key] = value
        return normalized
