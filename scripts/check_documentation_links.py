#!/usr/bin/env python3
"""
Documentation link checker for Neural SDK.
Checks for broken internal and external links.
"""

import re
import sys
from pathlib import Path
from urllib.parse import urlparse

import requests


class DocumentationLinkChecker:
    def __init__(self, docs_dir: Path = Path("docs")):
        self.docs_dir = docs_dir
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.checked_urls: set[str] = set()

    def check_all_links(self) -> bool:
        """Check all links in documentation."""
        print("🔗 Checking documentation links...")

        for mdx_file in self.docs_dir.rglob("*.mdx"):
            self._check_file_links(mdx_file)

        self._print_results()
        return len(self.errors) == 0

    def _check_file_links(self, mdx_file: Path) -> None:
        """Check links in a single documentation file."""
        try:
            with open(mdx_file, encoding="utf-8") as f:
                content = f.read()

            # Find all links
            links = self._extract_links(content)

            for link_text, link_url in links:
                self._check_link(link_url, mdx_file, link_text)

        except Exception as e:
            self.errors.append(f"Error checking links in {mdx_file.name}: {e}")

    def _extract_links(self, content: str) -> list[tuple[str, str]]:
        """Extract all links from markdown content."""
        links = []

        # Markdown links: [text](url)
        markdown_links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", content)
        links.extend(markdown_links)

        # Reference links: [text][ref]
        reference_links = re.findall(r"\[([^\]]+)\]\[([^\]]+)\]", content)
        for text, ref in reference_links:
            # Find reference definition
            ref_pattern = rf"\[{ref}\]:\s*(.+)"
            ref_match = re.search(ref_pattern, content)
            if ref_match:
                links.append((text, ref_match.group(1).strip()))

        return links

    def _check_link(self, url: str, file_path: Path, link_text: str) -> None:
        """Check a single link."""
        if url.startswith("#"):
            # Internal anchor link
            self._check_anchor_link(url, file_path, link_text)
        elif url.startswith("http://") or url.startswith("https://"):
            # External link
            self._check_external_link(url, file_path, link_text)
        elif url.startswith("/"):
            # Absolute internal link
            self._check_absolute_internal_link(url, file_path, link_text)
        else:
            # Relative internal link
            self._check_relative_internal_link(url, file_path, link_text)

    def _check_anchor_link(self, url: str, file_path: Path, link_text: str) -> None:
        """Check internal anchor link."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # Remove # and URL encode
            anchor = url[1:].lower().replace("-", " ").replace("_", " ")

            # Look for matching header
            headers = re.findall(r"^#+\s+(.+)$", content, re.MULTILINE)
            header_texts = [h.lower().replace("-", " ").replace("_", " ") for h in headers]

            if anchor not in header_texts:
                self.errors.append(f"Broken anchor in {file_path.name}: [{link_text}]({url})")

        except Exception as e:
            self.warnings.append(f"Could not check anchor {url} in {file_path.name}: {e}")

    def _check_external_link(self, url: str, file_path: Path, link_text: str) -> None:
        """Check external link."""
        if url in self.checked_urls:
            return

        self.checked_urls.add(url)

        try:
            # Skip certain domains that might block requests
            skip_domains = ["localhost", "127.0.0.1", "example.com"]
            parsed = urlparse(url)
            if any(domain in parsed.netloc for domain in skip_domains):
                return

            # Make request with timeout
            response = requests.head(url, timeout=10, allow_redirects=True)

            if response.status_code >= 400:
                self.errors.append(
                    f"Broken external link in {file_path.name}: [{link_text}]({url}) - {response.status_code}"
                )

        except requests.exceptions.RequestException as e:
            self.warnings.append(f"Could not check external link {url} in {file_path.name}: {e}")

    def _check_absolute_internal_link(self, url: str, file_path: Path, link_text: str) -> None:
        """Check absolute internal link."""
        target_path = self.docs_dir / url.lstrip("/")

        if url.endswith(".mdx"):
            if not target_path.exists():
                self.errors.append(
                    f"Broken internal link in {file_path.name}: [{link_text}]({url})"
                )
        elif url.endswith("/"):
            # Link to directory - check for index.mdx
            index_path = target_path / "index.mdx"
            if not index_path.exists():
                self.errors.append(
                    f"Broken internal link in {file_path.name}: [{link_text}]({url})"
                )
        else:
            # Link to directory without trailing slash
            index_path = target_path / "index.mdx"
            if not index_path.exists():
                self.errors.append(
                    f"Broken internal link in {file_path.name}: [{link_text}]({url})"
                )

    def _check_relative_internal_link(self, url: str, file_path: Path, link_text: str) -> None:
        """Check relative internal link."""
        base_dir = file_path.parent
        target_path = base_dir / url

        if url.endswith(".mdx"):
            if not target_path.exists():
                self.errors.append(
                    f"Broken relative link in {file_path.name}: [{link_text}]({url})"
                )
        elif url.endswith("/"):
            # Link to directory - check for index.mdx
            index_path = target_path / "index.mdx"
            if not index_path.exists():
                self.errors.append(
                    f"Broken relative link in {file_path.name}: [{link_text}]({url})"
                )
        else:
            # Link to directory without trailing slash
            index_path = target_path / "index.mdx"
            if not index_path.exists():
                self.errors.append(
                    f"Broken relative link in {file_path.name}: [{link_text}]({url})"
                )

    def _print_results(self) -> None:
        """Print check results."""
        if self.errors:
            print(f"\n❌ Found {len(self.errors)} broken links:")
            for error in self.errors:
                print(f"  • {error}")

        if self.warnings:
            print(f"\n⚠️ Found {len(self.warnings)} warnings:")
            for warning in self.warnings:
                print(f"  • {warning}")

        if not self.errors and not self.warnings:
            print("✅ All links are valid!")


if __name__ == "__main__":
    checker = DocumentationLinkChecker()
    success = checker.check_all_links()
    sys.exit(0 if success else 1)
