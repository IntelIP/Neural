#!/usr/bin/env python3
"""
Documentation validation script for Neural SDK.
Ensures documentation quality and completeness.
"""

import ast
import json
import re
from pathlib import Path


class DocumentationValidator:
    def __init__(self, docs_dir: Path = Path("docs")):
        self.docs_dir = docs_dir
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def validate_all(self) -> bool:
        """Run all validation checks."""
        print("🔍 Validating documentation...")

        self.validate_mint_json()
        self.validate_required_sections()
        self.validate_code_blocks()
        self.validate_internal_links()
        self.validate_api_coverage()
        self.validate_examples_coverage()

        return self.report_results()

    def validate_mint_json(self) -> None:
        """Validate mint.json configuration."""
        mint_file = self.docs_dir / "mint.json"
        if not mint_file.exists():
            self.errors.append("mint.json not found")
            return

        try:
            with open(mint_file) as f:
                config = json.load(f)

            # Check required fields
            required_fields = ["name", "navigation"]
            for field in required_fields:
                if field not in config:
                    self.errors.append(f"mint.json missing required field: {field}")

            # Validate navigation structure
            if "navigation" in config:
                self._validate_navigation(config["navigation"])

        except json.JSONDecodeError as e:
            self.errors.append(f"Invalid JSON in mint.json: {e}")

    def _validate_navigation(self, navigation: list[dict]) -> None:
        """Validate navigation structure."""
        for group in navigation:
            if "group" not in group or "pages" not in group:
                self.errors.append("Navigation group missing 'group' or 'pages'")
                continue

            for page in group["pages"]:
                if isinstance(page, str):
                    page_path = self.docs_dir / f"{page}.mdx"
                    if not page_path.exists():
                        self.errors.append(f"Navigation page not found: {page}.mdx")

    def validate_required_sections(self) -> None:
        """Check for required documentation sections."""
        required_sections = [
            "getting-started.mdx",
            "README.mdx",
            "architecture/start-here.mdx",
            "data-collection/overview.mdx",
            "analysis/overview.mdx",
            "trading/overview.mdx",
        ]

        for section in required_sections:
            section_path = self.docs_dir / section
            if not section_path.exists():
                self.errors.append(f"Required documentation section missing: {section}")

    def validate_code_blocks(self) -> None:
        """Validate code blocks in documentation."""
        for mdx_file in self.docs_dir.rglob("*.mdx"):
            try:
                with open(mdx_file) as f:
                    content = f.read()

                # Find Python code blocks
                code_blocks = re.findall(r"```python\n(.*?)\n```", content, re.DOTALL)

                for i, code in enumerate(code_blocks):
                    try:
                        ast.parse(code)
                    except SyntaxError as e:
                        self.errors.append(
                            f"Syntax error in {mdx_file.relative_to(self.docs_dir)} "
                            f"code block {i + 1}: {e}"
                        )

            except Exception as e:
                self.warnings.append(f"Could not read {mdx_file}: {e}")

    def validate_internal_links(self) -> None:
        """Validate internal documentation links."""
        for mdx_file in self.docs_dir.rglob("*.mdx"):
            try:
                with open(mdx_file) as f:
                    content = f.read()

                # Find internal links
                links = re.findall(r"\[([^\]]+)\]\(([^)]+\.mdx)\)", content)

                for text, target in links:
                    # Handle relative paths
                    if target.startswith("./"):
                        target_path = mdx_file.parent / target
                    elif target.startswith("/"):
                        target_path = self.docs_dir / target.lstrip("/")
                    else:
                        target_path = self.docs_dir / target

                    if not target_path.exists():
                        self.errors.append(
                            f"Broken link in {mdx_file.relative_to(self.docs_dir)}: "
                            f"[{text}]({target})"
                        )

            except Exception as e:
                self.warnings.append(f"Could not validate links in {mdx_file}: {e}")

    def validate_api_coverage(self) -> None:
        """Check if all public modules are documented."""
        neural_dir = Path("neural")
        if not neural_dir.exists():
            return

        documented_modules: set[str] = set()

        # Find documented modules
        api_dir = self.docs_dir / "api"
        if api_dir.exists():
            for module_file in api_dir.rglob("*.mdx"):
                rel_path = module_file.relative_to(api_dir)
                if rel_path.name == "index.mdx":
                    module_name = str(rel_path.parent).replace("/", ".")
                    documented_modules.add(module_name)

        # Find actual modules
        actual_modules: set[str] = set()
        for py_file in neural_dir.rglob("__init__.py"):
            rel_path = py_file.relative_to(neural_dir)
            if rel_path == Path("__init__.py"):
                actual_modules.add("neural")
            else:
                module_name = "neural." + str(rel_path.parent).replace("/", ".")
                actual_modules.add(module_name)

        # Check for undocumented modules
        undocumented = actual_modules - documented_modules
        for module in sorted(undocumented):
            if not any(skip in module for skip in ["__pycache__", "tests"]):
                self.warnings.append(f"Module not documented in API reference: {module}")

    def validate_examples_coverage(self) -> None:
        """Check if examples are documented."""
        examples_dir = Path("examples")
        if not examples_dir.exists():
            return

        documented_examples: set[str] = set()

        # Find documented examples
        examples_docs = self.docs_dir / "examples"
        if examples_docs.exists():
            for doc_file in examples_docs.rglob("*.mdx"):
                documented_examples.add(doc_file.stem)

        # Find actual examples
        actual_examples: set[str] = set()
        for py_file in examples_dir.glob("*.py"):
            actual_examples.add(py_file.stem)

        # Check for undocumented examples
        undocumented = actual_examples - documented_examples
        for example in sorted(undocumented):
            if example != "README":
                self.warnings.append(f"Example not documented: {example}.py")

    def report_results(self) -> bool:
        """Report validation results."""
        if self.errors:
            print(f"\n❌ Found {len(self.errors)} errors:")
            for error in self.errors:
                print(f"  • {error}")

        if self.warnings:
            print(f"\n⚠️  Found {len(self.warnings)} warnings:")
            for warning in self.warnings:
                print(f"  • {warning}")

        if not self.errors and not self.warnings:
            print("✅ All documentation validation checks passed!")

        return len(self.errors) == 0


if __name__ == "__main__":
    validator = DocumentationValidator()
    success = validator.validate_all()
    exit(0 if success else 1)
