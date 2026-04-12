# Low-Confidence Module Triage

## Decision

Classify the low-confidence Neural surfaces into three groups for the public beta:

- `Ship now`: Polymarket US reads and streaming only, with explicit read-only or read-and-stream-only positioning.
- `Defer`: deployment helpers and FIX streaming stay in the repo but remain outside the supported beta path.
- `Remove from story`: sentiment and social-data tooling, plus the broader research trading stack, stop contributing to the default beta narrative.

## Context

The Kalshi-first beta boundary is already established in the public docs and the open-core feature matrix, but several adjacent modules still create launch noise because they are visible in the repo and partially documented. The release process needs a durable decision record that maps those modules to a launch stance.

The highest-risk surfaces are not equally mature:

- Polymarket US live trading is intentionally blocked by `NotImplementedError`, but the read and streaming paths have meaningful adapter and source coverage.
- Deployment helpers are structurally useful, but they still contain explicit placeholders and shallow validation.
- FIX streaming remains partially implemented.
- Sentiment, aggregation, backtesting, strategy, and execution-research modules have thin or inconsistent test grounding relative to the public beta promise.

## Options Considered

- Keep all visible modules in the public story and rely on caveats alone.
- Hide or delete every incomplete surface before launch.
- Keep useful exploratory modules in the repo, but explicitly separate ship-now, defer, and remove-from-story decisions.

## Outcome

Choose explicit triage rather than silent drift.

### Ship Now

- Polymarket US read path
- Polymarket US streaming path

Conditions:

- keep all live order placement, cancel, and status paths clearly unsupported
- keep docs and examples aligned with the read-only or read-and-stream-only contract
- continue treating this as experimental relative to the Kalshi-first core

Follow-up issues:

- `INT-193` Mark Polymarket as read-only beta and hide unsupported live-trading claims

### Defer

- deployment helpers
- FIX streaming

Conditions:

- keep both out of the default onboarding and release story
- keep them behind extras and explicit warnings
- do not elevate them into the free-core promise until runtime behavior and tests improve

Follow-up issues:

- `INT-206` Add FIX streaming regression coverage and capability gating
- `INT-232` Move experimental deployment stack out of the default install and onboarding story
- `INT-226` Make the landing page build reproducible in a clean CI environment

### Remove From Story

- Twitter and social-data collection
- aggregation and enrichment sources that depend on fragile providers
- sentiment analysis and sentiment-driven strategies
- research trading stack: backtesting, strategy library, order-manager and auto-executor workflows that are not yet release-grade

Conditions:

- do not use these modules in README, landing-page copy, or promoted onboarding flows
- keep them available for internal iteration and future PR-sized hardening work
- only move them back toward the public story when tests, examples, and release validation support that promotion

Follow-up issues:

- `INT-194` Quarantine or remove the Twitter sentiment source from the default story
- `INT-200` Refresh historical-data and backtesting examples around supported APIs
- `INT-204` Raise backtesting engine coverage to a release-worthy baseline

## Consequences

- The public beta promise stays narrow enough to defend.
- Contributors can still iterate on non-core modules without pretending they are launch-ready.
- Release reviews now have a concrete decision log to audit against instead of relying on memory.
