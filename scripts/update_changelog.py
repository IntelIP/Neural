#!/usr/bin/env python3
"""
Automatic changelog updater for Neural SDK.
Analyzes git commits and updates CHANGELOG.md.
"""

import re
import subprocess
from datetime import datetime
from pathlib import Path


class ChangelogUpdater:
    def __init__(self, changelog_path: Path = Path("CHANGELOG.md")):
        self.changelog_path = changelog_path
        self.version_pattern = r"^## \[(\d+\.\d+\.\d+)\]"

    def update_changelog(self) -> None:
        """Update changelog with latest changes."""
        print("📝 Updating changelog...")

        # Get current version
        current_version = self._get_current_version()
        if not current_version:
            print("Could not determine current version")
            return

        # Get changes since last tag
        changes = self._get_changes_since_last_tag()
        if not changes:
            print("No changes to add to changelog")
            return

        # Categorize changes
        categorized = self._categorize_changes(changes)

        # Update changelog
        self._update_changelog_file(current_version, categorized)

        print(f"✅ Updated changelog for version {current_version}")

    def _get_current_version(self) -> str:
        """Get current version from pyproject.toml."""
        try:
            with open("pyproject.toml") as f:
                content = f.read()

            match = re.search(r'version = "([^"]+)"', content)
            if match:
                return match.group(1)
        except FileNotFoundError:
            pass

        return ""

    def _get_changes_since_last_tag(self) -> list[dict[str, str]]:
        """Get commit messages since last tag."""
        try:
            # Get last tag
            result = subprocess.run(
                ["git", "describe", "--tags", "--abbrev=0"], capture_output=True, text=True
            )

            if result.returncode != 0:
                # No tags found, get all commits
                commit_range = ""
            else:
                last_tag = result.stdout.strip()
                commit_range = f"{last_tag}..HEAD"

            # Get commit messages
            result = subprocess.run(
                ["git", "log", "--pretty=format:%H|%s|%b", commit_range],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                return []

            commits = []
            for line in result.stdout.strip().split("\n"):
                if line:
                    hash_val, subject, body = line.split("|", 2)
                    commits.append({"hash": hash_val, "subject": subject, "body": body})

            return commits

        except Exception as e:
            print(f"Error getting git commits: {e}")
            return []

    def _categorize_changes(self, commits: list[dict[str, str]]) -> dict[str, list[str]]:
        """Categorize commits by type."""
        categories = {
            "Added": [],
            "Changed": [],
            "Deprecated": [],
            "Removed": [],
            "Fixed": [],
            "Security": [],
            "Documentation": [],
            "Performance": [],
            "Code Quality": [],
        }

        for commit in commits:
            message = f"{commit['subject']} {commit['body']}".strip()

            # Skip merge commits and chore commits
            if message.startswith("Merge") or message.startswith("chore"):
                continue

            # Categorize based on conventional commits
            if message.startswith("feat") or message.startswith("add"):
                categories["Added"].append(self._clean_message(message))
            elif message.startswith("fix") or message.startswith("bugfix"):
                categories["Fixed"].append(self._clean_message(message))
            elif message.startswith("docs") or message.startswith("documentation"):
                categories["Documentation"].append(self._clean_message(message))
            elif message.startswith("perf") or message.startswith("performance"):
                categories["Performance"].append(self._clean_message(message))
            elif message.startswith("refactor") or message.startswith("style"):
                categories["Code Quality"].append(self._clean_message(message))
            elif message.startswith("change") or message.startswith("update"):
                categories["Changed"].append(self._clean_message(message))
            elif message.startswith("deprecate"):
                categories["Deprecated"].append(self._clean_message(message))
            elif message.startswith("remove"):
                categories["Removed"].append(self._clean_message(message))
            elif message.startswith("security"):
                categories["Security"].append(self._clean_message(message))
            else:
                # Try to infer from content
                if any(keyword in message.lower() for keyword in ["add", "new", "implement"]):
                    categories["Added"].append(self._clean_message(message))
                elif any(
                    keyword in message.lower() for keyword in ["fix", "bug", "error", "issue"]
                ):
                    categories["Fixed"].append(self._clean_message(message))
                elif any(keyword in message.lower() for keyword in ["doc", "readme", "example"]):
                    categories["Documentation"].append(self._clean_message(message))
                elif any(
                    keyword in message.lower() for keyword in ["performance", "optimize", "speed"]
                ):
                    categories["Performance"].append(self._clean_message(message))
                elif any(keyword in message.lower() for keyword in ["lint", "format", "refactor"]):
                    categories["Code Quality"].append(self._clean_message(message))
                else:
                    categories["Changed"].append(self._clean_message(message))

        # Remove empty categories
        return {k: v for k, v in categories.items() if v}

    def _clean_message(self, message: str) -> str:
        """Clean commit message for changelog."""
        # Remove conventional commit prefixes
        message = re.sub(
            r"^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)(\(.+\))?:\s*",
            "",
            message,
        )

        # Remove issue numbers and PR references
        message = re.sub(r"\(#\d+\)", "", message)
        message = re.sub(r"\[skip ci\]", "", message)

        # Clean up whitespace
        message = re.sub(r"\s+", " ", message).strip()

        # Capitalize first letter
        if message:
            message = message[0].upper() + message[1:]

        return message

    def _update_changelog_file(self, version: str, changes: dict[str, list[str]]) -> None:
        """Update the changelog file with new changes."""
        if not self.changelog_path.exists():
            self._create_initial_changelog()

        # Read current changelog
        with open(self.changelog_path) as f:
            content = f.read()

        # Create new version entry
        today = datetime.now().strftime("%Y-%m-%d")
        new_entry = f"## [{version}] - {today}\n\n"

        # Add changes
        for category, items in changes.items():
            if items:
                new_entry += f"### {category}\n"
                for item in items:
                    new_entry += f"- {item}\n"
                new_entry += "\n"

        # Insert new entry after the header
        header_end = content.find("\n\n")
        if header_end == -1:
            updated_content = content + "\n" + new_entry
        else:
            updated_content = content[: header_end + 2] + new_entry + content[header_end + 2 :]

        # Write updated changelog
        with open(self.changelog_path, "w") as f:
            f.write(updated_content)

    def _create_initial_changelog(self) -> None:
        """Create initial changelog file."""
        initial_content = """# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog and this project adheres to Semantic Versioning.

"""
        with open(self.changelog_path, "w") as f:
            f.write(initial_content)


if __name__ == "__main__":
    updater = ChangelogUpdater()
    updater.update_changelog()
