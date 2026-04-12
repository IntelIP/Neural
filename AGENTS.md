# AGENTS.md

## Purpose

This repository uses three systems for product development work:

- `Linear` is the execution system of record.
- `Notion` is the long-form product and developer knowledge base.
- `GitHub` is the implementation and review system of record.

Codex should keep them aligned whenever the user asks about roadmap, SDK scope, release planning, developer adoption, weekly updates, or follow-up work from PR reviews.

## Required Workflow

1. Read `docs/product-ops.md` before making product-management updates.
2. Update `Linear` first for execution changes.
3. Update repo docs second only when they preserve the operating model.
4. Use `docs/notion/` for Notion-ready Markdown when direct Notion access is unavailable.

## Linear Rules

- Prefer the existing Linear project `Neural – Core Library.` unless the user explicitly asks to create a new project.
- Keep the project structure simple:
  - milestones for durable SDK and release phases
  - issues for actionable work
  - existing team labels before new labels
- Every implementation PR should map to one Linear issue when possible.
- Important reliability, release, or documentation follow-up work should become Linear issues instead of living only in PR comments.

## Notion Rules

- Use Notion for:
  - product vision
  - roadmap narrative
  - SDK specs
  - weekly updates
  - release notes
  - developer adoption briefs
  - decision logs
- If direct Notion access is unavailable, stage content in `docs/notion/`.

## GitHub Rules

- Branch names should include the Linear issue identifier when available.
- PR titles should include the Linear issue identifier or link it in the body.
- When summarizing work, map:
  - GitHub PR outcome -> Linear issue status update
  - release or developer follow-up -> new Linear issue
  - weekly or release summary -> Notion-ready Markdown draft

## Daily Shipping Rules

- Prefer one implementation branch and one PR per Linear issue.
- Branch names should use: `hudson/int-123-short-kebab-summary`
- PR titles should use: `INT-123 Short imperative summary`
- PR bodies should include:
  - `Problem`
  - `Outcome`
  - `Acceptance Criteria`
  - `Validation`
- Keep PR scope small enough to merge in one day when possible.
- If a task expands beyond one PR-sized unit, split the overflow into new Linear issues instead of widening the branch scope.
