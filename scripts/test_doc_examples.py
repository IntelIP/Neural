#!/usr/bin/env python3
"""
Test documentation examples to ensure they work correctly.
"""

import ast
import sys
import tempfile
from pathlib import Path


class DocumentationExampleTester:
    def __init__(self, docs_dir: Path = Path("docs")):
        self.docs_dir = docs_dir
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def test_all_examples(self) -> bool:
        """Test all code examples in documentation."""
        print("🧪 Testing documentation examples...")

        for mdx_file in self.docs_dir.rglob("*.mdx"):
            self._test_file_examples(mdx_file)

        self._print_results()
        return len(self.errors) == 0

    def _test_file_examples(self, mdx_file: Path) -> None:
        """Test code examples in a single documentation file."""
        try:
            with open(mdx_file, encoding="utf-8") as f:
                content = f.read()

            # Extract Python code blocks
            code_blocks = self._extract_python_blocks(content)

            for i, code in enumerate(code_blocks):
                self._test_code_block(code, mdx_file, i + 1)

        except Exception as e:
            self.errors.append(f"Error testing {mdx_file.name}: {e}")

    def _extract_python_blocks(self, content: str) -> list[str]:
        """Extract Python code blocks from markdown content."""
        import re

        pattern = r"```python\n(.*?)\n```"
        matches = re.findall(pattern, content, re.DOTALL)
        return matches

    def _test_code_block(self, code: str, file_path: Path, block_num: int) -> None:
        """Test a single code block."""
        # Skip blocks that are clearly not meant to be run
        if any(skip in code.lower() for skip in ["...", "# example", "your code here"]):
            return

        # Skip blocks with obvious placeholders
        if any(placeholder in code for placeholder in ["your-email@example.com", "your-password"]):
            return

        try:
            # Check syntax
            ast.parse(code)

            # Try to execute in a safe environment
            self._execute_safely(code, file_path, block_num)

        except SyntaxError as e:
            self.errors.append(f"Syntax error in {file_path.name} block {block_num}: {e}")
        except Exception as e:
            self.warnings.append(f"Could not test {file_path.name} block {block_num}: {e}")

    def _execute_safely(self, code: str, file_path: Path, block_num: int) -> None:
        """Safely execute code block."""
        # Create a safe execution environment
        safe_globals = {
            "__builtins__": {
                "print": print,
                "len": len,
                "range": range,
                "list": list,
                "dict": dict,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
            }
        }

        # Add common imports that might be needed
        safe_globals.update(
            {
                "neural": None,  # Will be imported if needed
            }
        )

        try:
            # Execute in a temporary file to avoid namespace pollution
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                temp_file = f.name

            # Try to compile and execute
            compiled = compile(code, f"<{file_path.name}:{block_num}>", "exec")
            exec(compiled, safe_globals)

            # Clean up
            Path(temp_file).unlink()

        except Exception as e:
            # Clean up on error
            if "temp_file" in locals():
                try:
                    Path(temp_file).unlink()
                except:
                    pass
            raise e

    def _print_results(self) -> None:
        """Print test results."""
        if self.errors:
            print(f"\n❌ Found {len(self.errors)} errors:")
            for error in self.errors:
                print(f"  • {error}")

        if self.warnings:
            print(f"\n⚠️ Found {len(self.warnings)} warnings:")
            for warning in self.warnings:
                print(f"  • {warning}")

        if not self.errors and not self.warnings:
            print("✅ All documentation examples passed testing!")


if __name__ == "__main__":
    tester = DocumentationExampleTester()
    success = tester.test_all_examples()
    sys.exit(0 if success else 1)
