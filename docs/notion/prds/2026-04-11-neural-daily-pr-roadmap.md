# Neural Daily PR Roadmap

## Purpose

Turn the 30-day beta ship plan into a daily delivery model that keeps the public SDK honest, keeps the landing page shippable, and leaves enough follow-on work for one meaningful PR per day.

## Operating Constraints

- Keep the public beta Kalshi-first.
- Treat Polymarket as read and stream beta only until live support exists.
- Keep deployment, sentiment, and control-plane work out of the default install and onboarding story.
- Prefer one PR-sized Linear issue per branch.
- Avoid work that expands the public promise faster than tests, docs, and release gates.

## Daily PR Rule

Each day should close one of these units:

- one packaging or dependency-hardening PR
- one docs or example-truthfulness PR
- one trading or data reliability PR
- one landing-page or launch-surface PR
- one release-process or adoption PR
- one roadmap or premium-boundary PR

If a change cannot fit this size, split it before starting implementation.

## 30-Day Sequence

### Days 1-7: Ship Truth And Packaging Integrity

- close CLI and package-surface mismatches
- finish extras split and import-guard hardening
- add wheel and sdist smoke validation
- document branch and PR conventions for daily shipping
- rewrite install and getting-started docs around supported paths
- make landing-page build and deploy checks reproducible
- publish the module-status and support matrix

### Days 8-14: Kalshi-First Reliability

- add nightly credentialed Kalshi verification
- harden the nightly job preflight, secret expectations, and failure summaries
- improve supported market-discovery helpers
- expand paper-trading happy-path coverage
- replace silent runtime suppression with explicit failures or warnings
- tighten streaming retry and disconnect handling
- classify low-confidence modules into ship, defer, or remove decisions
- remove deployment and sentiment from the default beta story

### Days 15-21: Release Candidate Formation

- add publish dry-run validation before tags
- create the beta release checklist and first release-notes draft
- validate clean installs across supported Python versions
- align landing-page messaging with the shipping SDK scope
- write the developer adoption brief
- publish the first candidate release notes draft
- confirm issue-to-branch and PR linkage is working in practice

### Days 22-30: Open-Core And Next Backlog

- finalize the open-core feature matrix
- draft the post-beta premium roadmap
- define the first premium provider-extension contracts
- write the weekly update cadence and operator summary template
- convert launch learnings into the next 30 days of backlog
- keep at least one reliability or developer-adoption PR landing daily
- defer speculative TUI or platform work unless it directly unlocks supported beta flows

## Open-Core Roadmap

### Free Core

- Kalshi-first SDK
- auth and environment helpers
- market data collection
- paper trading
- supported live reads and selected live trade paths
- CLI health and capability checks
- public docs and release notes

### Experimental Visible Surface

- Polymarket read and stream support
- backtesting helpers with limited validation
- advanced strategy and execution helpers that stay behind clear beta warnings
- deployment scaffolding that is documented as non-core

### Premium Follow-On

- hosted runners and managed execution workflows
- premium data and enrichment packs
- private provider adapters and routing logic
- operator dashboards, control-plane workflows, and team-facing telemetry
- managed reliability, incident response, and workflow orchestration

## Exit Criteria For The First 30 Days

- package install and CLI smoke checks run in CI
- nightly credentialed Kalshi verification exists
- nightly secret expectations and triage guidance are documented
- landing page deploy path is documented and reproducible
- public docs match supported APIs and scope
- release checklist and release-notes artifacts exist for each beta drop
- open-core boundary is explicit across docs and Linear
- the next 30 days of PR-sized backlog already exist before the beta cycle ends
