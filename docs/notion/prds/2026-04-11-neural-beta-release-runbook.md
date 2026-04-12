# Neural Beta Release Runbook

## Purpose

Define the repeatable release motion for the Neural public beta so releases stop depending on local knowledge.

## Release Preconditions

- The supported beta scope is unchanged or explicitly documented.
- README, docs, examples, and landing-page copy agree on the beta boundary.
- Package smoke checks pass for the supported install paths.
- The landing page validates with its documented build path.
- New launch blockers have corresponding Linear issues.

## Required Checks

### SDK

- lint and targeted tests pass
- wheel and sdist smoke path pass
- installed-package smoke path validates from a non-repo cwd
- CLI health and JSON output remain clean

### Live Verification

- nightly Kalshi workflow exists and is green, or any failures are explicitly triaged
- required secrets for credentialed verification are present in the runtime
- if the nightly job is red, check the step summary first for missing secret wiring before rerunning

### Launch Surface

- landing-page `bun run check` passes
- landing-page `bun run build` passes
- waitlist path and required environment variables are documented

## Release Flow

1. Confirm the target Linear issues are in `Done` or explicitly deferred.
2. Review the module-status matrix, open-core feature matrix, and low-confidence module decision log.
3. Run the packaging and release validation path.
4. Prepare the release checklist, release notes, beta FAQ, and the external beta summary.
5. Publish the package artifact.
6. Verify install and CLI behavior from a clean environment.
7. Confirm the landing page still matches the shipped SDK contract.

## Secrets And Environment

- package publish credentials must be configured in the release environment
- Kalshi credentialed smoke secrets must be configured in GitHub Actions:
  - `KALSHI_API_KEY_ID`
  - one private key source, either `KALSHI_PRIVATE_KEY_BASE64` or `KALSHI_PRIVATE_KEY_PATH`
  - optional `KALSHI_API_BASE`, which defaults to `https://api.elections.kalshi.com`
- landing-page deploy secrets must include Cloudflare auth and `DATABASE_URL`

## Nightly Kalshi Failure Handling

- If the workflow fails before tests start, treat it as a secret or runtime wiring issue rather than a product regression.
- If the smoke test returns 401 or 403, confirm the API key and private key belong to the same Kalshi account and that the key is still active.
- If the job fails with a network or TLS error, retry once after confirming GitHub Actions did not have a transient outage.
- If the job keeps failing after secret wiring is fixed, open or update a Linear issue with the exact failure output and the workflow run URL.

## Rollback Rules

- If package smoke, live verification, or install validation fails, stop the release and open a follow-up Linear issue before retrying.
- If the landing page message drifts from the shipped SDK surface, revert the copy or delay the release.
- If a publish succeeds but a clean install fails, treat the release as broken and publish corrective notes immediately.

## Release Outputs

- one release-checklist draft
- one release-notes draft
- one beta FAQ draft
- one concise external beta summary
- one Linear update covering what shipped, what remains blocked, and the next PR-sized tasks

## Follow-On

- convert release surprises into new Linear issues the same day
- update the weekly update draft after each release
- rebuild the next 30-day backlog before the current release cycle ends
