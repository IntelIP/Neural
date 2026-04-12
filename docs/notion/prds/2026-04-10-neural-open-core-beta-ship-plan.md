# Neural Open-Core Beta Ship Plan

## Problem

Neural has a meaningful SDK foundation, but the current product story is broader than the tested and production-ready surface. Packaging, docs, examples, and release workflows currently ask users to trust more than the software justifies.

## Users

- Independent quantitative developers who want a fast path to prediction-market automation.
- Sports-market traders who need market data, paper trading, and narrow live execution support.
- Technical early adopters willing to use a Kalshi-first beta if the boundaries are explicit.
- Internal maintainers who need a reliable daily-PR delivery cadence.

## Goals

- Ship an honest public beta in 30 days.
- Narrow the core story to a Kalshi-first SDK with strong paper-trading and market-data workflows.
- Make the landing page and waitlist production-ready enough for launch.
- Create a Linear backlog that supports daily PRs for at least the next 30 days.
- Establish the open-core boundary so future premium work does not bloat the public SDK.

## Non-goals

- Do not ship a full multi-exchange live trading platform in this phase.
- Do not make experimental deployment or control-plane surfaces part of the beta promise.
- Do not build a speculative UI or TUI as a launch blocker.
- Do not promise social-data or sentiment-provider support that is not proven.

## Scope

### Core Beta

- Kalshi-first authentication and market data access
- Historical collection and replay on supported paths
- Paper trading and narrow live trading support where implemented
- Truthful docs, examples, packaging, and CI/release workflows
- Landing page plus waitlist capture

### Experimental But Visible

- Polymarket read and stream support where implemented
- Backtesting and execution helpers that are kept only if they align with the real supported API
- Selected advanced modules that are explicitly marked experimental

### Future Premium Surfaces

- Hosted runners and managed deployment workflows
- Strategy orchestration and control-plane features
- Rich telemetry, reporting, and team workflows
- Private adapters, provider contracts, and operational tooling

## Success Metrics

- Beta scope is consistent across README, docs, examples, package metadata, and landing page.
- Landing page is deployable from a clean CI environment.
- Release dry-run and nightly credentialed checks exist.
- A minimum 30-day PR-sized backlog exists in Linear.
- Every shipped PR maps cleanly back to a Linear issue.

## Risks

- Scope creep from older platform ambitions or unfinished premium ideas
- Docs drift recreating trust gaps after cleanup
- Hidden integration regressions because the live suite is not yet continuous
- Bloat from retaining under-tested modules in the public story

## Open Questions

- Which Polymarket capabilities should remain publicly documented during the beta?
- How aggressively should low-confidence modules be deferred versus hidden behind extras?
- What is the first premium surface with the clearest revenue path after beta?
