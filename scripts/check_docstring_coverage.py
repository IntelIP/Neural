#!/usr/bin/env python3
"""
Docstring coverage checker for Neural SDK.
Analyzes code to ensure proper documentation coverage.
"""

import argparse
import ast
import sys
from pathlib import Path


class DocstringCoverageChecker:
    def __init__(self, source_dir: Path = Path("neural")):
        self.source_dir = source_dir
        self.results: dict[str, dict] = {}
        self.total_modules = 0
        self.total_classes = 0
        self.total_functions = 0
        self.documented_modules = 0
        self.documented_classes = 0
        self.documented_functions = 0

    def check_coverage(self) -> bool:
        """Check docstring coverage for all Python files."""
        print("🔍 Checking docstring coverage...")

        for py_file in self.source_dir.rglob("*.py"):
            if py_file.name.startswith("__"):
                continue

            self._check_file(py_file)

        self._print_summary()
        return self._get_overall_coverage() >= 80.0

    def _check_file(self, file_path: Path) -> None:
        """Check docstring coverage for a single file."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)
            module_name = str(file_path.relative_to(self.source_dir).with_suffix(""))

            file_results = {
                "module_docstring": bool(ast.get_docstring(tree)),
                "classes": {},
                "functions": {},
                "total_classes": 0,
                "documented_classes": 0,
                "total_functions": 0,
                "documented_functions": 0,
            }

            # Check classes
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    file_results["classes"][node.name] = {
                        "has_docstring": bool(ast.get_docstring(node)),
                        "methods": {},
                    }
                    file_results["total_classes"] += 1

                    if ast.get_docstring(node):
                        file_results["documented_classes"] += 1

                    # Check methods
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            has_docstring = bool(ast.get_docstring(item))
                            file_results["classes"][node.name]["methods"][item.name] = has_docstring
                            file_results["total_functions"] += 1

                            if has_docstring:
                                file_results["documented_functions"] += 1

                elif isinstance(node, ast.FunctionDef):
                    # Module-level functions
                    if not any(
                        isinstance(parent, ast.ClassDef)
                        for parent in ast.walk(tree)
                        if hasattr(parent, "body") and node in parent.body
                    ):
                        has_docstring = bool(ast.get_docstring(node))
                        file_results["functions"][node.name] = has_docstring
                        file_results["total_functions"] += 1

                        if has_docstring:
                            file_results["documented_functions"] += 1

            self.results[module_name] = file_results
            self.total_modules += 1
            self.total_classes += file_results["total_classes"]
            self.total_functions += file_results["total_functions"]

            if file_results["module_docstring"]:
                self.documented_modules += 1
            self.documented_classes += file_results["documented_classes"]
            self.documented_functions += file_results["documented_functions"]

        except Exception as e:
            print(f"⚠️ Could not analyze {file_path}: {e}")

    def _get_overall_coverage(self) -> float:
        """Calculate overall docstring coverage percentage."""
        total_items = self.total_modules + self.total_classes + self.total_functions
        documented_items = (
            self.documented_modules + self.documented_classes + self.documented_functions
        )

        if total_items == 0:
            return 100.0

        return (documented_items / total_items) * 100.0

    def _print_summary(self) -> None:
        """Print coverage summary."""
        overall_coverage = self._get_overall_coverage()

        print("\n📊 Docstring Coverage Summary")
        print("=" * 50)
        print(
            f"Modules: {self.documented_modules}/{self.total_modules} ({self._get_percentage(self.documented_modules, self.total_modules)}%)"
        )
        print(
            f"Classes: {self.documented_classes}/{self.total_classes} ({self._get_percentage(self.documented_classes, self.total_classes)}%)"
        )
        print(
            f"Functions: {self.documented_functions}/{self.total_functions} ({self._get_percentage(self.documented_functions, self.total_functions)}%)"
        )
        print(f"\nOverall Coverage: {overall_coverage:.1f}%")

        if overall_coverage >= 90:
            print("🎉 Excellent documentation coverage!")
        elif overall_coverage >= 80:
            print("✅ Good documentation coverage")
        elif overall_coverage >= 70:
            print("⚠️ Acceptable documentation coverage")
        else:
            print("❌ Poor documentation coverage - needs improvement")

        # Print files with low coverage
        print("\n📋 Files needing attention:")
        for module_name, results in self.results.items():
            file_coverage = self._get_file_coverage(results)
            if file_coverage < 80:
                print(f"  • {module_name}: {file_coverage:.1f}%")

    def _get_percentage(self, documented: int, total: int) -> str:
        """Get percentage as string."""
        if total == 0:
            return "100"
        return f"{(documented / total) * 100:.1f}"

    def _get_file_coverage(self, results: dict) -> float:
        """Get coverage percentage for a single file."""
        total = 1 + results["total_classes"] + results["total_functions"]  # 1 for module
        documented = (
            (1 if results["module_docstring"] else 0)
            + results["documented_classes"]
            + results["documented_functions"]
        )

        if total == 0:
            return 100.0

        return (documented / total) * 100.0

    def generate_report(self, output_file: str = None) -> str:
        """Generate detailed coverage report."""
        report = []
        report.append("# Docstring Coverage Report\n")
        report.append(
            f"Generated on: {ast.literal_eval(str(__import__('datetime').datetime.now()))}"
        )
        report.append(f"Overall Coverage: {self._get_overall_coverage():.1f}%\n")

        report.append("## Summary\n")
        report.append(f"- Modules: {self.documented_modules}/{self.total_modules}")
        report.append(f"- Classes: {self.documented_classes}/{self.total_classes}")
        report.append(f"- Functions: {self.documented_functions}/{self.total_functions}\n")

        report.append("## Detailed Results\n")
        for module_name, results in sorted(self.results.items()):
            coverage = self._get_file_coverage(results)
            report.append(f"### {module_name} ({coverage:.1f}%)\n")

            if not results["module_docstring"]:
                report.append("- ❌ Missing module docstring")

            for class_name, class_info in results["classes"].items():
                if not class_info["has_docstring"]:
                    report.append(f"- ❌ Class `{class_name}` missing docstring")

                for method_name, has_docstring in class_info["methods"].items():
                    if not has_docstring and not method_name.startswith("_"):
                        report.append(f"- ❌ Method `{class_name}.{method_name}` missing docstring")

            for func_name, has_docstring in results["functions"].items():
                if not has_docstring:
                    report.append(f"- ❌ Function `{func_name}` missing docstring")

            report.append("")

        report_text = "\n".join(report)

        if output_file:
            with open(output_file, "w") as f:
                f.write(report_text)
            print(f"📄 Detailed report saved to {output_file}")

        return report_text


def main():
    parser = argparse.ArgumentParser(description="Check docstring coverage")
    parser.add_argument("--source", default="neural", help="Source directory to check")
    parser.add_argument("--output", help="Output file for detailed report")
    parser.add_argument("--threshold", type=float, default=80.0, help="Coverage threshold")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    checker = DocstringCoverageChecker(Path(args.source))
    success = checker.check_coverage()

    if args.output or args.verbose:
        checker.generate_report(args.output)

    # Exit with error code if coverage is below threshold
    if checker._get_overall_coverage() < args.threshold:
        print(f"\n❌ Coverage below threshold of {args.threshold}%")
        sys.exit(1)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
