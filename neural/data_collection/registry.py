from typing import Dict, Type, Any
from .base import DataSource
from .transformer import DataTransformer


class DataSourceRegistry:
    """Registry for managing data sources."""

    def __init__(self):
        self.sources: Dict[str, Type[DataSource]] = {}
        self.transformers: Dict[str, DataTransformer] = {}

    def register_source(self, source_class: Type[DataSource], transformer: DataTransformer = None):
        """Register a data source class."""
        self.sources[source_class.__name__] = source_class
        if transformer:
            self.transformers[source_class.__name__] = transformer

    def get_source(self, name: str, *args, **kwargs) -> DataSource:
        """Get an instance of a registered source."""
        if name not in self.sources:
            raise ValueError(f"Source {name} not registered")
        return self.sources[name](*args, **kwargs)

    def get_transformer(self, source_name: str) -> DataTransformer:
        """Get the transformer for a source."""
        return self.transformers.get(source_name, DataTransformer())


# Global registry
registry = DataSourceRegistry()


def register_source(transformer: DataTransformer = None):
    """Decorator to register a data source class."""
    def decorator(cls: Type[DataSource]):
        registry.register_source(cls, transformer)
        return cls
    return decorator