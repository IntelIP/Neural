# Product Operations

This repo uses a simple product-development operating model so Codex can update planning artifacts without inventing process each time.

## Systems

- `Linear`: execution, milestones, issues, release work, reliability work
- `Notion`: product strategy, roadmap narrative, SDK specs, weekly updates, release notes
- `GitHub`: implementation, pull requests, CI, and review follow-up

## Default Flow

1. Start in `Notion` or a Notion-ready Markdown draft when defining a product or release change.
2. Break the work into `Linear` milestones and issues.
3. Implement through `GitHub` branches and PRs.
4. Convert important review, release, or adoption outcomes back into `Linear`.
5. Publish long-form summaries back into `Notion`.

## Daily PR Conventions

- Prefer one PR-sized Linear issue per branch.
- Branch names should follow `hudson/int-123-short-kebab-summary`.
- PR titles should follow `INT-123 Short imperative summary`.
- PR bodies should restate:
  - `Problem`
  - `Outcome`
  - `Acceptance Criteria`
  - `Validation`
- When a branch starts absorbing unrelated work, split the overflow into follow-up issues instead of widening the PR.
- Use release-facing issues to track docs, smoke checks, nightly verification, landing-page launch work, and premium-boundary decisions.

## Minimal Linear Setup

- Project: `Neural – Core Library.`
- Team: current owning team in Linear
- Milestones:
  - `SDK Foundation`
  - `Trading and Data Flows`
  - `Developer Adoption`
  - `Release Readiness`
- Labels:
  - prefer existing team labels first
  - use `backend`, `release`, `tech-debt`, `testing` when helpful
- Issue scope:
  - one issue per actionable unit of work
  - issue descriptions should include `Problem`, `Outcome`, and `Acceptance Criteria`

## Notion Workspace Map

- `Product / Vision`
- `Product / Roadmap`
- `Product / SDK Specs`
- `Delivery / Weekly Updates`
- `Engineering / Release Notes`
- `Developer / Adoption`
- `Engineering / Decisions`

If direct Notion access is unavailable, use `docs/notion/` as the staging directory.
