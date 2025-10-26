#!/usr/bin/env python3
"""
Generate documentation for Python examples.
Automatically creates documentation from example scripts.
"""

import ast
import re
from pathlib import Path


class ExampleDocumentationGenerator:
    def __init__(
        self,
        examples_dir: Path = Path("examples"),
        docs_dir: Path = Path("docs/examples/generated"),
    ):
        self.examples_dir = examples_dir
        self.docs_dir = docs_dir
        self.docs_dir.mkdir(parents=True, exist_ok=True)

    def generate_all(self) -> None:
        """Generate documentation for all examples."""
        print("📝 Generating example documentation...")

        examples = list(self.examples_dir.glob("*.py"))
        examples.sort()

        # Generate index
        self._generate_index(examples)

        # Generate individual example docs
        for example_file in examples:
            if example_file.name.startswith("README"):
                continue
            self._generate_example_doc(example_file)

        print(f"✅ Generated documentation for {len(examples)} examples")

    def _generate_index(self, examples: list[Path]) -> None:
        """Generate index page for examples."""
        index_content = """---
title: Examples
description: Complete collection of Neural SDK examples
---

# Examples

This section contains comprehensive examples demonstrating various aspects of the Neural SDK.

## Quick Start Examples

"""

        # Categorize examples
        categories = self._categorize_examples(examples)

        for category, category_examples in categories.items():
            index_content += f"### {category}\n\n"

            for example in category_examples:
                doc_info = self._extract_doc_info(example)
                example_name = example.stem

                index_content += f"- **[{doc_info['title']}]({example_name})**\n"
                index_content += f"  {doc_info['description']}\n\n"

        index_content += """
## Running Examples

All examples can be run directly:

```bash
python examples/01_data_collection.py
```

Make sure you have the Neural SDK installed:

```bash
pip install neural-sdk
```

## Prerequisites

Some examples require additional setup:

1. **Authentication**: Set up your Kalshi credentials
2. **API Keys**: Configure required API keys in your environment
3. **Dependencies**: Install optional dependencies for specific features

See the [Getting Started](../getting-started) guide for detailed setup instructions.
"""

        with open(self.docs_dir / "index.mdx", "w") as f:
            f.write(index_content)

    def _categorize_examples(self, examples: list[Path]) -> dict[str, list[Path]]:
        """Categorize examples by functionality."""
        categories = {
            "Data Collection": [],
            "Trading & Execution": [],
            "Strategy Development": [],
            "Analysis & Backtesting": [],
            "Complete Workflows": [],
            "Advanced Features": [],
        }

        for example in examples:
            name = example.stem.lower()

            if any(keyword in name for keyword in ["data", "collection", "historical", "stream"]):
                categories["Data Collection"].append(example)
            elif any(keyword in name for keyword in ["order", "trading", "fix", "client", "live"]):
                categories["Trading & Execution"].append(example)
            elif any(keyword in name for keyword in ["strategy", "sentiment", "bot"]):
                categories["Strategy Development"].append(example)
            elif any(keyword in name for keyword in ["backtest", "analysis", "test"]):
                categories["Analysis & Backtesting"].append(example)
            elif any(keyword in name for keyword in ["complete", "demo", "workflow"]):
                categories["Complete Workflows"].append(example)
            else:
                categories["Advanced Features"].append(example)

        # Remove empty categories
        return {k: v for k, v in categories.items() if v}

    def _generate_example_doc(self, example_file: Path) -> None:
        """Generate documentation for a single example."""
        doc_info = self._extract_doc_info(example_file)
        example_name = example_file.stem

        content = f"""---
title: {doc_info["title"]}
description: {doc_info["description"]}
---

# {doc_info["title"]}

{doc_info["description"]}

## Overview

{doc_info["overview"]}

## Prerequisites

{doc_info["prerequisites"]}

## Code

```python
{self._read_example_code(example_file)}
```

## Running the Example

```bash
python examples/{example_file.name}
```

## Expected Output

{doc_info["expected_output"]}

## Key Concepts Demonstrated

{doc_info["key_concepts"]}

## Related Documentation

{doc_info["related_docs"]}
"""

        with open(self.docs_dir / f"{example_name}.mdx", "w") as f:
            f.write(content)

    def _extract_doc_info(self, example_file: Path) -> dict[str, str]:
        """Extract documentation information from example file."""
        try:
            with open(example_file) as f:
                content = f.read()

            # Parse AST to extract docstrings and comments
            tree = ast.parse(content)

            # Extract module docstring
            module_doc = ast.get_docstring(tree) or ""

            # Extract imports
            imports = self._extract_imports(tree)

            # Extract functions and classes
            functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

            # Extract main execution block
            main_code = self._extract_main_block(content)

            # Generate documentation based on filename and content
            example_name = example_file.stem
            doc_info = self._generate_doc_info(
                example_name, module_doc, imports, functions, classes, main_code
            )

            return doc_info

        except Exception as e:
            print(f"Warning: Could not fully process {example_file}: {e}")
            return self._generate_fallback_doc_info(example_file.stem)

    def _extract_imports(self, tree: ast.AST) -> list[str]:
        """Extract import statements."""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")
        return imports

    def _extract_main_block(self, content: str) -> str:
        """Extract main execution block."""
        # Look for if __name__ == "__main__" block
        match = re.search(
            r'if __name__ == ["\']__main__["\']:(.*?)(?=\n\n|\nclass|\ndef|\Z)', content, re.DOTALL
        )
        if match:
            return match.group(1).strip()
        return ""

    def _generate_doc_info(
        self,
        example_name: str,
        module_doc: str,
        imports: list[str],
        functions: list[str],
        classes: list[str],
        main_code: str,
    ) -> dict[str, str]:
        """Generate documentation info based on analysis."""

        # Default values
        title = example_name.replace("_", " ").replace("-", " ").title()
        description = module_doc.split("\n")[0] if module_doc else f"Example: {title}"

        # Customize based on example name
        if "data_collection" in example_name.lower():
            overview = "This example demonstrates how to collect market data from various sources using the Neural SDK's data collection modules."
            prerequisites = "- Neural SDK installed\n- API credentials for data sources"
            expected_output = "Market data printed to console or saved to file"
            key_concepts = "- Data sources configuration\n- Market data aggregation\n- Real-time data streaming"
            related_docs = "- [Data Collection Overview](../../data-collection/overview)\n- [Data Sources](../../data-collection/sources)"

        elif "trading" in example_name.lower() or "order" in example_name.lower():
            overview = "This example shows how to execute trades and manage orders using the Neural SDK's trading client."
            prerequisites = "- Neural SDK installed\n- Kalshi account and API credentials\n- Paper trading account recommended"
            expected_output = "Order confirmations and trade execution details"
            key_concepts = "- Order placement\n- Position management\n- Risk management"
            related_docs = "- [Trading Overview](../../trading/overview)\n- [Trading Client](../../trading/trading-client)"

        elif "strategy" in example_name.lower():
            overview = "This example demonstrates strategy development and implementation using the Neural SDK's strategy framework."
            prerequisites = "- Neural SDK installed\n- Understanding of trading strategies\n- Historical data for backtesting"
            expected_output = "Strategy performance metrics and trading signals"
            key_concepts = "- Strategy design patterns\n- Signal generation\n- Performance analysis"
            related_docs = "- [Strategy Foundations](../../analysis/strategy-foundations)\n- [Strategy Library](../../analysis/strategy-library)"

        else:
            overview = module_doc or "This example demonstrates key features of the Neural SDK."
            prerequisites = "- Neural SDK installed\n- Basic understanding of Python"
            expected_output = "Example output demonstrating the functionality"
            key_concepts = "- Neural SDK usage\n- Best practices\n- Common patterns"
            related_docs = "- [Getting Started](../../getting-started)\n- [Architecture Overview](../../architecture/overview)"

        return {
            "title": title,
            "description": description,
            "overview": overview,
            "prerequisites": prerequisites,
            "expected_output": expected_output,
            "key_concepts": key_concepts,
            "related_docs": related_docs,
        }

    def _generate_fallback_doc_info(self, example_name: str) -> dict[str, str]:
        """Generate fallback documentation info."""
        title = example_name.replace("_", " ").replace("-", " ").title()

        return {
            "title": title,
            "description": f"Example: {title}",
            "overview": "This example demonstrates Neural SDK functionality.",
            "prerequisites": "- Neural SDK installed",
            "expected_output": "Example output",
            "key_concepts": "- Neural SDK usage",
            "related_docs": "- [Getting Started](../../getting-started)",
        }

    def _read_example_code(self, example_file: Path) -> str:
        """Read and format example code."""
        with open(example_file) as f:
            return f.read()


if __name__ == "__main__":
    generator = ExampleDocumentationGenerator()
    generator.generate_all()
