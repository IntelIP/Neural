#!/usr/bin/env python3
"""
Examples validation script for Neural SDK.
Validates that all examples are functional and documented.
"""

import ast
import sys
from pathlib import Path


class ExamplesValidator:
    def __init__(self, examples_dir: Path = Path("examples")):
        self.examples_dir = examples_dir
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def validate_all(self) -> bool:
        """Validate all examples."""
        print("🔍 Validating examples...")

        if not self.examples_dir.exists():
            self.errors.append("Examples directory not found")
            return False

        example_files = list(self.examples_dir.glob("*.py"))
        if not example_files:
            self.warnings.append("No example files found")
            return True

        for example_file in example_files:
            self._validate_example(example_file)

        self._print_results()
        return len(self.errors) == 0

    def _validate_example(self, example_file: Path) -> None:
        """Validate a single example file."""
        try:
            # Check syntax
            self._check_syntax(example_file)

            # Check imports
            self._check_imports(example_file)

            # Check documentation
            self._check_documentation(example_file)

            # Check for common issues
            self._check_common_issues(example_file)

        except Exception as e:
            self.errors.append(f"Error validating {example_file.name}: {e}")

    def _check_syntax(self, example_file: Path) -> None:
        """Check Python syntax."""
        try:
            with open(example_file, encoding="utf-8") as f:
                content = f.read()
            ast.parse(content)
        except SyntaxError as e:
            self.errors.append(f"Syntax error in {example_file.name}: {e}")

    def _check_imports(self, example_file: Path) -> None:
        """Check that imports are valid."""
        try:
            with open(example_file, encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)
            imports = []

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        imports.append(f"{module}.{alias.name}")

            # Check for neural SDK imports
            neural_imports = [imp for imp in imports if imp.startswith("neural")]
            if not neural_imports:
                self.warnings.append(f"{example_file.name}: No neural SDK imports found")

        except Exception as e:
            self.warnings.append(f"Could not check imports in {example_file.name}: {e}")

    def _check_documentation(self, example_file: Path) -> None:
        """Check that example has documentation."""
        try:
            with open(example_file, encoding="utf-8") as f:
                content = f.read()

            # Check for docstring
            tree = ast.parse(content)
            if not ast.get_docstring(tree):
                self.warnings.append(f"{example_file.name}: Missing module docstring")

            # Check for comments
            if "#" not in content and '"""' not in content:
                self.warnings.append(f"{example_file.name}: No comments or documentation found")

        except Exception as e:
            self.warnings.append(f"Could not check documentation in {example_file.name}: {e}")

    def _check_common_issues(self, example_file: Path) -> None:
        """Check for common issues in examples."""
        try:
            with open(example_file, encoding="utf-8") as f:
                content = f.read()

            # Check for hardcoded credentials
            if any(
                keyword in content.lower() for keyword in ["password", "secret", "key", "token"]
            ):
                lines = content.split("\n")
                for i, line in enumerate(lines, 1):
                    if any(
                        keyword in line.lower()
                        for keyword in ["password", "secret", "key", "token"]
                    ):
                        if "=" in line and not line.strip().startswith("#"):
                            self.warnings.append(
                                f"{example_file.name}:{i}: Possible hardcoded credential"
                            )

            # Check for TODO/FIXME comments
            if "todo" in content.lower() or "fixme" in content.lower():
                self.warnings.append(f"{example_file.name}: Contains TODO/FIXME comments")

            # Check for print statements (should use logging in production)
            if "print(" in content:
                self.warnings.append(
                    f"{example_file.name}: Contains print statements (consider using logging)"
                )

            # Check for main execution block
            if 'if __name__ == "__main__"' not in content:
                self.warnings.append(f"{example_file.name}: Missing main execution block")

        except Exception as e:
            self.warnings.append(f"Could not check common issues in {example_file.name}: {e}")

    def _print_results(self) -> None:
        """Print validation results."""
        if self.errors:
            print(f"\n❌ Found {len(self.errors)} errors:")
            for error in self.errors:
                print(f"  • {error}")

        if self.warnings:
            print(f"\n⚠️ Found {len(self.warnings)} warnings:")
            for warning in self.warnings:
                print(f"  • {warning}")

        if not self.errors and not self.warnings:
            print("✅ All examples passed validation!")


if __name__ == "__main__":
    validator = ExamplesValidator()
    success = validator.validate_all()
    sys.exit(0 if success else 1)
