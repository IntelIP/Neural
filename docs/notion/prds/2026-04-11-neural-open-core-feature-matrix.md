# Neural Open-Core Feature Matrix

## Purpose

Make the Neural beta boundary explicit so docs, release work, and future premium planning all refer to the same contract.

## Tier Definitions

### Free Core

This is the supported public beta surface.

### Experimental Visible Surface

This remains in the repo and can be documented, but it is not part of the stable beta promise and should stay behind warnings, extras, or explicit operator review.

### Premium Follow-On

This is roadmap-only for now. It informs product direction and future packaging, but it should not expand the current public promise.

## Feature Matrix

| Surface | Current Tier | Current State | Launch Guidance |
| --- | --- | --- | --- |
| Kalshi auth and environment helpers | Free Core | Supported | Promote publicly |
| Kalshi market discovery and market data reads | Free Core | Supported | Promote publicly |
| Kalshi paper trading | Free Core | Supported | Promote publicly |
| Kalshi selected live trading flows | Free Core | Narrow beta | Promote only where tests and examples exist |
| `neural` doctor and capability checks | Free Core | Supported | Promote publicly |
| Public install path and package extras | Free Core | Supported | Promote publicly |
| Public docs, examples, release notes | Free Core | Supported | Keep aligned with the support matrix |
| Polymarket US reads | Experimental Visible Surface | Implemented | Document as beta-only |
| Polymarket US streaming | Experimental Visible Surface | Implemented | Document as beta-only |
| Polymarket US live order placement | Experimental Visible Surface | Not supported | Do not promote |
| FIX helpers and FIX streaming | Experimental Visible Surface | Partial | Keep behind `trading` extra and explicit warnings |
| Backtesting helpers | Experimental Visible Surface | Partial confidence | Keep visible only with caveats |
| Strategy and execution helper library | Experimental Visible Surface | Mixed maturity | Keep in docs only where grounded by tests |
| Sentiment and social-data tooling | Experimental Visible Surface | Research-only | Keep out of default onboarding |
| Deployment helpers | Experimental Visible Surface | Experimental | Keep out of default onboarding and launch story |
| Hosted runners | Premium Follow-On | Roadmap | Do not expose as current product |
| Managed deployment workflows | Premium Follow-On | Roadmap | Position as post-beta premium work |
| Premium data and enrichment packs | Premium Follow-On | Roadmap | Position as post-beta premium work |
| Private provider adapters and advanced routing | Premium Follow-On | Roadmap | Position as post-beta premium work |
| Operator dashboards and control plane | Premium Follow-On | Roadmap | Keep in roadmap only |
| Managed telemetry and incident operations | Premium Follow-On | Roadmap | Keep in roadmap only |

## Packaging Guidance

- Base install should cover the free core import surface and CLI.
- `trading` should unlock Kalshi execution, streaming, and FIX-adjacent helpers without changing the public support matrix by itself.
- `sentiment` should remain research-only.
- `deployment` should remain explicitly experimental.

## Messaging Rules

- Do not describe experimental visible surfaces as launch-ready.
- Do not let premium follow-on work leak into the beta promise.
- If a feature is not covered by tests, examples, and release validation, it cannot be promoted as part of the free core.

## Follow-On Execution

- Move any surface that lacks a clear tier assignment into a ship, defer, or remove decision.
- Use this matrix to audit README, docs IA, landing-page copy, and release notes before each beta drop.
- Convert premium follow-on themes into PR-sized backlog only after the free-core beta path is stable.
