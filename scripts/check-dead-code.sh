#!/usr/bin/env bash
# Managed by CodexSkills bootstrap-code-quality.py
set -euo pipefail
cd "$(dirname "$0")/.."
py_files=()
while IFS= read -r -d '' file; do py_files+=("$file"); done < <(git ls-files --cached --others --exclude-standard -z -- '*.py')
if ((${#py_files[@]})); then uvx --from vulture==2.16 vulture "${py_files[@]}" --min-confidence 90; fi
