#!/usr/bin/env python3
"""
Generate API documentation for Neural SDK modules.

This script automatically generates comprehensive API documentation
by scanning the neural package and creating structured documentation
files for each module.
"""

import os
import sys
import inspect
import importlib
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse


class APIDocGenerator:
    """Generate API documentation for Neural SDK modules."""

    def __init__(self, output_dir: str = "docs/api"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.modules_to_document = [
            "neural.auth",
            "neural.data_collection",
            "neural.trading",
            "neural.analysis",
            "neural.analysis.strategies",
            "neural.analysis.risk",
            "neural.analysis.execution",
        ]

    def generate_all(self) -> bool:
        """Generate documentation for all modules."""
        try:
            # Create main API index
            self._create_api_index()

            # Generate documentation for each module
            for module_name in self.modules_to_document:
                try:
                    self._generate_module_docs(module_name)
                    print(f"✅ Generated docs for {module_name}")
                except Exception as e:
                    print(f"❌ Failed to generate docs for {module_name}: {e}")
                    return False

            print(f"📚 API documentation generated in {self.output_dir}")
            return True

        except Exception as e:
            print(f"❌ Failed to generate API documentation: {e}")
            return False

    def _create_api_index(self) -> None:
        """Create the main API index file."""
        content = """---
title: API Reference
description: Complete API documentation for the Neural SDK
---

# API Reference

This section contains automatically generated documentation for all Neural SDK modules.

## Modules

"""

        for module_name in self.modules_to_document:
            module_path = module_name.replace(".", "/")
            content += f"- [{module_name}](api/{module_path})\n"

        index_file = self.output_dir / "overview.mdx"
        with open(index_file, "w") as f:
            f.write(content)

    def _generate_module_docs(self, module_name: str) -> None:
        """Generate documentation for a specific module."""
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            print(f"⚠️  Could not import {module_name}: {e}")
            return

        # Create module directory
        module_path = self.output_dir / module_name.replace(".", "/")
        module_path.mkdir(parents=True, exist_ok=True)

        # Generate module documentation
        content = self._generate_module_content(module, module_name)

        # Write to index file
        index_file = module_path / "index.mdx"
        with open(index_file, "w") as f:
            f.write(content)

    def _generate_module_content(self, module: Any, module_name: str) -> str:
        """Generate content for a module."""
        content = f"""---
title: {module_name}
description: API documentation for {module_name}
---

# {module_name}

"""

        # Add module docstring
        if module.__doc__:
            content += f"{module.__doc__}\n\n"

        # Get all classes and functions
        classes = []
        functions = []

for name, obj in inspect.getmembers(module):
            is_class = inspect.isclass(obj)
            is_function = inspect.isfunction(obj)
            obj_module = getattr(obj, '__module__', None)
            
            if is_class and obj_module == module_name:
                classes.append((name, obj))
            elif is_function and obj_module == module_name:
                functions.append((name, obj))

        # Add classes
        if classes:
            content += "## Classes\n\n"
            for name, cls in sorted(classes):
                content += self._generate_class_docs(name, cls)

        # Add functions
        if functions:
            content += "## Functions\n\n"
            for name, func in sorted(functions):
                content += self._generate_function_docs(name, func)

        return content

    def _generate_class_docs(self, name: str, cls: type) -> str:
        """Generate documentation for a class."""
        content = f"### {name}\n\n"

        # Add class docstring
        if cls.__doc__:
            content += f"{cls.__doc__}\n\n"

        # Get methods
        methods = []
        for method_name, method in inspect.getmembers(cls):
            if (
                inspect.ismethod(method) or inspect.isfunction(method)
            ) and not method_name.startswith("_"):
                methods.append((method_name, method))

        if methods:
            content += "#### Methods\n\n"
            for method_name, method in sorted(methods):
                content += self._generate_method_docs(method_name, method)

        return content

    def _generate_function_docs(self, name: str, func: callable) -> str:
        """Generate documentation for a function."""
        content = f"#### {name}\n\n"

        # Add function signature
        try:
            sig = inspect.signature(func)
            content += f"```python\n{name}{sig}\n```\n\n"
        except:
            content += f"```python\n{name}()\n```\n\n"

        # Add docstring
        if func.__doc__:
            content += f"{func.__doc__}\n\n"

        return content

    def _generate_method_docs(self, name: str, method: callable) -> str:
        """Generate documentation for a method."""
        content = f"##### {name}\n\n"

        # Add method signature
        try:
            sig = inspect.signature(method)
            # Remove 'self' parameter for instance methods
            params = list(sig.parameters.values())
            if params and params[0].name == "self":
                params = params[1:]
            new_sig = sig.replace(parameters=params)
            content += f"```python\n{name}{new_sig}\n```\n\n"
        except:
            content += f"```python\n{name}()\n```\n\n"

        # Add docstring
        if method.__doc__:
            content += f"{method.__doc__}\n\n"

        return content


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate API documentation")
    parser.add_argument(
        "--output-dir", default="docs/api", help="Output directory for generated documentation"
    )

    args = parser.parse_args()

    generator = APIDocGenerator(args.output_dir)
    success = generator.generate_all()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
