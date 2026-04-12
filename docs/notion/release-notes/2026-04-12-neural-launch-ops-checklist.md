# Neural Launch Ops Checklist

## Purpose
This is the concrete operator checklist for the remaining non-code launch gates:

1. Landing-page runtime wiring and deploy
2. Kalshi credentialed verification
3. PyPI package publish

Use this alongside:
- [Beta Release Runbook](/Users/hudson/Documents/GitHub/Neural/public/Neural/docs/notion/prds/2026-04-11-neural-beta-release-runbook.md)
- [Beta Release Checklist](/Users/hudson/Documents/GitHub/Neural/public/Neural/docs/notion/release-notes/2026-04-11-neural-beta-release-checklist.md)

## 1. Landing Page Runtime Wiring

### Goal
The live landing page loads, the waitlist API works, and signups insert into Neon with attribution fields.

### Required inputs
- `CLOUDFLARE_PAGES_PROJECT_NAME`
- `CLOUDFLARE_PAGES_PRODUCTION_BRANCH`
- `DATABASE_URL`
- optional if using Doppler:
  - `DOPPLER_TOKEN`
  - `DOPPLER_PROJECT_NAME`
  - `DOPPLER_RUNTIME_CONFIG`

### Files involved
- [deploy-pages.sh](/Users/hudson/Documents/GitHub/Neural/public/Neural-Landing-Page/scripts/deploy-pages.sh)
- [sync-pages-runtime-secrets.sh](/Users/hudson/Documents/GitHub/Neural/public/Neural-Landing-Page/scripts/sync-pages-runtime-secrets.sh)
- [waitlist.ts](/Users/hudson/Documents/GitHub/Neural/public/Neural-Landing-Page/src/pages/api/waitlist.ts)
- [schema.ts](/Users/hudson/Documents/GitHub/Neural/public/Neural-Landing-Page/src/db/schema.ts)
- [0001_sturdy_golden_guardian.sql](/Users/hudson/Documents/GitHub/Neural/public/Neural-Landing-Page/drizzle/0001_sturdy_golden_guardian.sql)

### Commands
From `/Users/hudson/Documents/GitHub/Neural/public/Neural-Landing-Page`:

```bash
bun test
bun run check
bun run build
```

If applying schema directly from the repo:

```bash
bun run db:push
```

If syncing runtime secrets from Doppler:

```bash
./scripts/sync-pages-runtime-secrets.sh
```

If deploying directly with Wrangler:

```bash
./scripts/deploy-pages.sh
```

### Pass criteria
- `bun test`, `bun run check`, and `bun run build` all pass
- live Pages project has `DATABASE_URL` available at runtime
- Neon schema includes the waitlist attribution columns
- a real `POST /api/waitlist` request returns `201` or `200`
- inserted row includes:
  - `landing_path`
  - `utm_source`
  - `utm_medium`
  - `utm_campaign`
  - `utm_content`
  - `utm_term`

### Fail criteria
- `DATABASE_URL` missing in Pages runtime
- migration not applied
- deploy succeeds but waitlist insert returns `500`
- insert works but attribution columns are absent or null when present in request

## 2. Kalshi Credentialed Verification

### Goal
Run the credentialed nightly Kalshi smoke once manually and confirm the supported live path is real.

### Required inputs
- `KALSHI_API_KEY_ID`
- one of:
  - `KALSHI_PRIVATE_KEY_BASE64`
  - `KALSHI_PRIVATE_KEY_PATH`
- optional:
  - `KALSHI_API_BASE`

### Files involved
- [nightly-kalshi-smoke.yml](/Users/hudson/Documents/GitHub/Neural/public/Neural/.github/workflows/nightly-kalshi-smoke.yml)
- [test_auth_verify.py](/Users/hudson/Documents/GitHub/Neural/public/Neural/tests/infrastructure/test_auth_verify.py)

### Setup
Add the required secrets to GitHub Actions for the SDK repo.

Recommended values:
- production base: `https://api.elections.kalshi.com`
- demo base if intentionally testing demo: `https://demo-api.kalshi.co`

### Commands
Local repo preflight from `/Users/hudson/Documents/GitHub/Neural/public/Neural`:

```bash
make release-dry-run
```

Then manually trigger the workflow in GitHub Actions:
- Workflow: `nightly-kalshi-smoke`
- Trigger: manual dispatch first, schedule later

### Pass criteria
- workflow authenticates successfully
- auth verification test passes
- workflow completes green without credential/config errors

### Fail criteria
- missing secret
- invalid private key format
- auth verification fails
- network/base URL mismatch

### Interpretation
- If the workflow fails due to credentials or account state, the repo is still launchable as a beta but cannot claim full live Kalshi verification.
- If the workflow fails due to SDK behavior, reopen product hardening work before publish.

## 3. PyPI Publish

### Goal
Publish the SDK package cleanly and verify it installs outside the repo.

### Required inputs
- `PYPI_API_TOKEN`
- optional if using TestPyPI separately:
  - `TESTPYPI_API_TOKEN`

### Files involved
- [pyproject.toml](/Users/hudson/Documents/GitHub/Neural/public/Neural/pyproject.toml)
- [publish.yml](/Users/hudson/Documents/GitHub/Neural/public/Neural/.github/workflows/publish.yml)
- [release-dry-run.yml](/Users/hudson/Documents/GitHub/Neural/public/Neural/.github/workflows/release-dry-run.yml)
- [package_smoke.py](/Users/hudson/Documents/GitHub/Neural/public/Neural/scripts/package_smoke.py)

### Commands
From `/Users/hudson/Documents/GitHub/Neural/public/Neural`:

```bash
make release-dry-run
```

Confirm package version:

```bash
python3 - <<'PY'
from pathlib import Path
import tomllib
data = tomllib.loads(Path("pyproject.toml").read_text())
print(data["project"]["version"])
PY
```

After dry run passes, push the matching release tag:

```bash
git tag v0.4.1
git push origin v0.4.1
```

Replace `v0.4.1` with the actual version in `pyproject.toml` if it changes.

### Pass criteria
- release dry run passes
- `twine check` passes inside the dry run
- wheel and sdist install smoke passes
- publish workflow completes green
- package installs in a clean environment after publish

### Fail criteria
- version/tag mismatch
- PyPI token missing or invalid
- artifact upload failure
- published package cannot install cleanly outside the repo

## 4. Recommended Execution Order

1. Land landing-page runtime wiring and live verification
2. Run Kalshi credentialed smoke
3. Run SDK release dry run
4. Push the production tag and publish
5. Deploy docs manually if required by launch timing

## 5. Decision Rule

You can truthfully say:

- `Beta launched`: after landing page is live and package is published
- `Live Kalshi verified`: only after the credentialed workflow passes
- `Growth phase starts`: after the above are done, or after package plus landing page are done if Kalshi verification is temporarily blocked by account access
