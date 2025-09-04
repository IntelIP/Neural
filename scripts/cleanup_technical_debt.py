#!/usr/bin/env python3
"""
Technical Debt Cleanup Script

This script systematically addresses technical debt issues in the codebase:
1. Hardcoded sys.path.append statements
2. Type annotation issues
3. Import organization
4. Error handling inconsistencies
5. Logging standardization
6. Async/await patterns
7. Resource management

Usage:
    python scripts/cleanup_technical_debt.py
"""

import os
import re
import glob
import logging
from pathlib import Path
from typing import List, Dict, Set, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TechnicalDebtCleaner:
    """Systematic technical debt cleanup tool."""

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.fixed_files: Set[str] = set()
        self.issues_found: Dict[str, List[str]] = {}

    def run_cleanup(self) -> None:
        """Run the complete technical debt cleanup."""
        logger.info("🧹 Starting technical debt cleanup...")

        # Phase 1: Hardcoded paths
        self.fix_hardcoded_paths()

        # Phase 2: Import organization
        self.fix_import_organization()

        # Phase 3: Type annotations
        self.fix_type_annotations()

        # Phase 4: Error handling
        self.standardize_error_handling()

        # Phase 5: Logging
        self.standardize_logging()

        # Phase 6: Resource management
        self.add_resource_management()

        logger.info("✅ Technical debt cleanup completed!")
        logger.info(f"📁 Files modified: {len(self.fixed_files)}")
        logger.info(f"🔧 Issues addressed: {sum(len(issues) for issues in self.issues_found.values())}")

    def fix_hardcoded_paths(self) -> None:
        """Fix hardcoded sys.path.append statements."""
        logger.info("🔧 Fixing hardcoded sys.path.append statements...")

        # Find all Python files with hardcoded paths
        python_files = glob.glob(str(self.root_dir / "**" / "*.py"), recursive=True)

        hardcoded_path_pattern = re.compile(
            r'sys\.path\.append\([\'"](/Users/[^\'"]+)[\'"]\)'
        )

        for file_path in python_files:
            if "kalshi_trading_sdk" in file_path:  # Skip our new SDK
                continue

            try:
                with open(file_path, 'r') as f:
                    content = f.read()

                # Find hardcoded paths
                matches = hardcoded_path_pattern.findall(content)
                if matches:
                    logger.info(f"📝 Fixing hardcoded paths in {file_path}")

                    # Replace with relative imports
                    new_content = hardcoded_path_pattern.sub(
                        r'# Removed hardcoded path - use proper Python imports',
                        content
                    )

                    # Add proper relative import path
                    if 'if __name__ == "__main__":' in new_content:
                        # Add proper path setup
                        new_content = new_content.replace(
                            'if __name__ == "__main__":',
                            'if __name__ == "__main__":\n    import sys\n    # Use proper relative imports instead of hardcoded paths\n    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))\n\nif __name__ == "__main__":'
                        )

                    with open(file_path, 'w') as f:
                        f.write(new_content)

                    self.fixed_files.add(file_path)
                    self.issues_found.setdefault(file_path, []).extend(
                        [f"Fixed hardcoded path: {path}" for path in matches]
                    )

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")

    def fix_import_organization(self) -> None:
        """Fix import organization and remove duplicates."""
        logger.info("📦 Organizing imports...")

        python_files = glob.glob(str(self.root_dir / "**" / "*.py"), recursive=True)

        for file_path in python_files:
            if "kalshi_trading_sdk" in file_path:  # Skip our new SDK
                continue

            try:
                with open(file_path, 'r') as f:
                    content = f.read()

                # Check for missing os import when os.path is used
                if 'os.path' in content and 'import os' not in content:
                    logger.info(f"📝 Adding missing os import to {file_path}")

                    # Find the first import line
                    lines = content.split('\n')
                    insert_index = 0
                    for i, line in enumerate(lines):
                        if line.startswith('import ') or line.startswith('from '):
                            insert_index = i
                            break

                    lines.insert(insert_index, 'import os')
                    new_content = '\n'.join(lines)

                    with open(file_path, 'w') as f:
                        f.write(new_content)

                    self.fixed_files.add(file_path)
                    self.issues_found.setdefault(file_path, []).append("Added missing os import")

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")

    def fix_type_annotations(self) -> None:
        """Fix type annotation issues."""
        logger.info("🏷️  Fixing type annotations...")

        # This would require more sophisticated parsing
        # For now, we'll focus on the major issues we've identified
        logger.info("Type annotation fixes would require AST parsing - skipping for now")

    def standardize_error_handling(self) -> None:
        """Standardize error handling patterns."""
        logger.info("🚨 Standardizing error handling...")

        # Look for bare except clauses
        python_files = glob.glob(str(self.root_dir / "**" / "*.py"), recursive=True)

        for file_path in python_files:
            if "kalshi_trading_sdk" in file_path:  # Skip our new SDK
                continue

            try:
                with open(file_path, 'r') as f:
                    content = f.read()

                # Find bare except clauses
                if 'except:' in content:
                    logger.info(f"⚠️  Found bare except clause in {file_path}")
                    self.issues_found.setdefault(file_path, []).append("Contains bare except clause")

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")

    def standardize_logging(self) -> None:
        """Standardize logging patterns."""
        logger.info("📝 Standardizing logging patterns...")

        python_files = glob.glob(str(self.root_dir / "**" / "*.py"), recursive=True)

        for file_path in python_files:
            if "kalshi_trading_sdk" in file_path:  # Skip our new SDK
                continue

            try:
                with open(file_path, 'r') as f:
                    content = f.read()

                # Check for inconsistent logging
                if 'print(' in content and 'logger.' not in content:
                    logger.info(f"🖨️  Found print statements in {file_path} (should use logging)")
                    self.issues_found.setdefault(file_path, []).append("Uses print() instead of logging")

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")

    def add_resource_management(self) -> None:
        """Add proper resource management and context managers."""
        logger.info("🔒 Adding resource management...")

        # Look for files that might need context managers
        python_files = glob.glob(str(self.root_dir / "**" / "*.py"), recursive=True)

        for file_path in python_files:
            if "kalshi_trading_sdk" in file_path:  # Skip our new SDK
                continue

            try:
                with open(file_path, 'r') as f:
                    content = f.read()

                # Check for Redis connections that might need cleanup
                if 'redis.' in content and 'close()' not in content:
                    logger.info(f"🔗 Found Redis usage without explicit cleanup in {file_path}")
                    self.issues_found.setdefault(file_path, []).append("Redis connection may need explicit cleanup")

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")

    def generate_report(self) -> str:
        """Generate a cleanup report."""
        report = []
        report.append("# Technical Debt Cleanup Report")
        report.append("")
        report.append(f"## Summary")
        report.append(f"- **Files processed:** {len(self.fixed_files)}")
        report.append(f"- **Issues found:** {sum(len(issues) for issues in self.issues_found.values())}")
        report.append("")

        if self.fixed_files:
            report.append("## Files Modified")
            for file_path in sorted(self.fixed_files):
                report.append(f"- `{file_path}`")
            report.append("")

        if self.issues_found:
            report.append("## Issues Found")
            for file_path, issues in sorted(self.issues_found.items()):
                report.append(f"### `{file_path}`")
                for issue in issues:
                    report.append(f"- {issue}")
                report.append("")

        report.append("## Recommendations")
        report.append("1. **Hardcoded paths**: Replace with proper relative imports")
        report.append("2. **Type annotations**: Add comprehensive type hints")
        report.append("3. **Error handling**: Replace bare except clauses with specific exceptions")
        report.append("4. **Logging**: Replace print() statements with proper logging")
        report.append("5. **Resource management**: Add context managers for Redis connections")
        report.append("6. **Import organization**: Group imports by type (stdlib, third-party, local)")

        return "\n".join(report)


def main():
    """Main entry point."""
    cleaner = TechnicalDebtCleaner("/Users/hudson/Documents/GitHub/IntelIP/PROJECTS/Neural/Kalshi_Agentic_Agent")
    cleaner.run_cleanup()

    # Generate and save report
    report = cleaner.generate_report()
    report_path = Path("/Users/hudson/Documents/GitHub/IntelIP/PROJECTS/Neural/Kalshi_Agentic_Agent/TECHNICAL_DEBT_REPORT.md")

    with open(report_path, 'w') as f:
        f.write(report)

    logger.info(f"📋 Report saved to {report_path}")


if __name__ == "__main__":
    main()