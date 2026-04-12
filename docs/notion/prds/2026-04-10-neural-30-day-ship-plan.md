# Neural 30-Day Ship Plan

## Problem

Neural has real product substance, but the current public surface overstates what is truly production-ready. The SDK passes local quality gates, yet the ship surface is blurred across experimental modules, stale docs, incomplete exchange support, and packaging drift. The landing page is close to deployable, but the overall product story is not yet honest, narrow, and repeatable enough for a clean public beta.

## Users

- Quant and developer early adopters who want a reliable Kalshi-first SDK
- Internal operators shipping daily PRs against a clear execution plan
- Future paid users who will want premium data, managed infrastructure, and private execution surfaces

## Goals

- Ship an honest public beta in 30 days
- Narrow the public surface to what works now
- Turn the repo into a daily-PR machine with small, sequenced issues
- Establish a clean open-core boundary between free core and private premium surfaces

## Non-goals

- Full multi-exchange live trading in the next 30 days
- Managed deployment platform GA in the public repo
- Production claims for Twitter sentiment, advanced deployment automation, or unsupported Polymarket live trading

## Scope

### Public Core

- Kalshi-first Python SDK
- Auth, market data, paper trading, selected live trading flows
- Honest backtesting and risk tooling where APIs are proven
- Public docs, examples, release process, and landing page

### Private / Premium Follow-on

- Managed deployment provider
- Premium data and signal packs
- Advanced execution and operator surfaces
- TUI and private platform integrations when backed by stable SDK contracts

## Success Metrics

- `pip install neural-sdk` and base import path work reliably
- `neural` CLI exists and returns stable JSON for health and capability checks
- Public docs match the shipping APIs
- Landing page is deployed with a working waitlist path
- Nightly credentialed exchange smoke tests run for the supported live surface
- At least 20 PR-sized issues are execution-ready and at least 30 days of backlog exists

## Risks

- Scope creep toward platform and TUI work before the SDK beta is honest
- Docs and examples regressing faster than code changes
- Heavy base dependencies making install and packaging brittle
- Shipping broad product claims before unsupported exchange behavior is fenced off

## Open Questions

- Whether the first public beta should expose any Polymarket surface beyond read and streaming capabilities
- Whether the first release should ship a minimal CLI only or a richer operator workflow
- Which premium surface should be productized first after the public beta: deployment, premium data, or managed execution

## Worker Lanes

### Lane A: Ship Truth

- narrow public claims to tested capabilities
- remove or clearly mark unsupported and experimental paths

### Lane B: SDK Surface

- repair packaging, install paths, CLI, and dependency boundaries
- make the public interface small, stable, and scriptable

### Lane C: Trading Reliability

- harden Kalshi-first flows, paper trading, streaming, and live smoke coverage

### Lane D: Developer Adoption

- refresh docs, examples, release notes, and onboarding

### Lane E: Launch Surface

- deploy the landing page, waitlist, analytics, and release communication path

### Lane F: Open-Core Expansion

- define what stays free and what becomes private premium follow-on work

## Sprint 1: Ship Truth And Surface Slimming

### Sprint Goal

Make the public SDK installable, honest, and reproducible.

### Planned Stories

- Repair the missing `neural` CLI entrypoint and define a minimal JSON contract
- Add wheel and sdist smoke validation to CI
- Slim the base dependency set and move heavy modules behind extras
- Rewrite README and getting-started around the actual beta surface
- Remove or rewrite stale examples that rely on unsupported flows
- Make the landing-page build reproducible from a clean checkout
- Publish the initial module-status table for core, experimental, and private surfaces

## Sprint 2: Kalshi-First Production Beta

### Sprint Goal

Make the supported trading and data path trustworthy enough for public beta.

### Planned Stories

- Harden Kalshi market discovery helpers and supported filters
- Add nightly credentialed smoke tests for auth, market browse, and safe live reads
- Expand paper-trading validation and operator-friendly CLI commands
- Replace silent runtime failure paths with explicit logging and typed errors
- Improve streaming retry and disconnect recovery behavior
- Fence experimental modules behind warnings, docs, and extras
- Refresh historical-data and backtesting examples to use proven APIs

## Sprint 3: Developer Adoption And Launch

### Sprint Goal

Turn the SDK and landing page into a coherent release candidate.

### Planned Stories

- Validate install paths for Python 3.10, 3.11, and 3.12
- Refresh docs IA so public, experimental, and private surfaces are clearly separated
- Generate reference docs only from shipping APIs
- Publish the beta release checklist and first release-notes draft
- Release a candidate build to TestPyPI
- Add landing-page conversion instrumentation and source attribution
- Finalize public beta narrative, FAQ, and support runbook

## Sprint 4: Open-Core And Post-Launch Growth

### Sprint Goal

Preserve daily delivery after launch while establishing premium follow-on surfaces.

### Planned Stories

- Define and document the open-core boundary
- Add provider-extension contracts for private and premium modules
- Separate premium roadmap themes: advanced execution, private deployment, premium data, managed ops
- Create the future CLI-to-TUI contract without making TUI a current blocker
- Establish recurring weekly update and release-note cadence
- Build the next 30-day PR queue from learnings during the beta launch

## Open-Core Model

### Free Core

- Kalshi-first SDK
- auth and environment helpers
- market data collection
- paper trading
- selected live trading flows that are fully supported
- docs, examples, and release notes

### Source-Available But Experimental

- advanced strategies
- backtesting helpers with limited validation
- partial exchange adapters
- deployment scaffolding not yet sold as a product

### Private Premium

- managed deployment provider
- premium data and enrichment packs
- advanced execution and routing
- operator dashboards and TUI/private platform surfaces
- managed reliability and hosted workflows

## Execution Notes

- Linear is the execution system of record
- use the existing `Neural – Core Library.` project and its current milestones
- keep one issue per PR-sized unit of work when possible
- treat old TUI and platform issues as downstream work unless they directly unblock the 30-day public beta
