#!/usr/bin/env python3
"""
Documentation health check script.
Monitors deployed documentation for issues.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests


class DocumentationHealthChecker:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.issues: list[dict[str, Any]] = []

    def run_health_check(self) -> bool:
        """Run comprehensive health check."""
        print(f"🏥 Running health check for {self.base_url}")

        # Check main page
        self._check_page("/")

        # Check key sections
        key_sections = [
            "/docs",
            "/docs/getting-started",
            "/docs/data-collection/overview",
            "/docs/trading/overview",
            "/docs/analysis/overview",
        ]

        for section in key_sections:
            self._check_page(section)

        self._generate_report()
        return len(self.issues) == 0

    def _check_page(self, path: str) -> None:
        """Check a specific page."""
        url = urljoin(self.base_url, path)

        try:
            response = requests.get(url, timeout=10)

            if response.status_code != 200:
                self.issues.append(
                    {
                        "type": "page_error",
                        "url": url,
                        "status_code": response.status_code,
                        "message": f"Page returned {response.status_code}",
                    }
                )
            elif response.text.strip() == "":
                self.issues.append({"type": "empty_page", "url": url, "message": "Page is empty"})
            else:
                # Check for common error indicators
                error_indicators = ["404", "not found", "error", "undefined"]
                content_lower = response.text.lower()

                for indicator in error_indicators:
                    if indicator in content_lower and len(response.text) < 1000:
                        self.issues.append(
                            {
                                "type": "content_error",
                                "url": url,
                                "message": f"Page contains error indicator: {indicator}",
                            }
                        )
                        break

        except requests.exceptions.RequestException as e:
            self.issues.append({"type": "request_error", "url": url, "message": str(e)})

    def _generate_report(self) -> None:
        """Generate health check report."""
        print("\n📊 Health Check Report")
        print("=" * 50)

        if not self.issues:
            print("✅ All health checks passed!")
            return

        # Group issues by type
        issue_types = {}
        for issue in self.issues:
            issue_type = issue["type"]
            if issue_type not in issue_types:
                issue_types[issue_type] = []
            issue_types[issue_type].append(issue)

        for issue_type, issues in issue_types.items():
            print(f"\n❌ {issue_type.replace('_', ' ').title()} ({len(issues)} issues):")
            for issue in issues:
                print(f"  • {issue['url']}: {issue['message']}")

        # Save detailed report
        report_data = {
            "timestamp": str(__import__("datetime").datetime.now()),
            "base_url": self.base_url,
            "total_issues": len(self.issues),
            "issues": self.issues,
        }

        report_file = Path("health-check-report.json")
        with open(report_file, "w") as f:
            json.dump(report_data, f, indent=2)

        print(f"\n📄 Detailed report saved to {report_file}")


def main():
    parser = argparse.ArgumentParser(description="Documentation health check")
    parser.add_argument("--url", required=True, help="Deployed Fumadocs base URL to check")
    parser.add_argument("--output", help="Output file for report")

    args = parser.parse_args()

    checker = DocumentationHealthChecker(args.url)
    success = checker.run_health_check()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
