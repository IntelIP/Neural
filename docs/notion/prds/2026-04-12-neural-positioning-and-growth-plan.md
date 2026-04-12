# Neural Positioning And Growth Plan

## Problem

Neural now has a credible Kalshi-first beta surface, but the public story is still fragile. Without a benchmarked positioning layer, launch copy can drift toward broad platform claims, and growth work can become generic instead of targeted.

## Users

- Python-first developers who want to automate Kalshi workflows without starting from raw exchange primitives
- Prediction-market traders who want market data, paper trading, and a narrow live beta path
- Small teams evaluating whether Neural is a faster starting point than building directly on exchange-native SDKs

## Goals

- Define a benchmarked category position for Neural
- Sharpen launch messaging so it matches the actual beta surface
- Create a 30-day growth loop that fits the current product maturity
- Turn launch and growth work into small, daily PR-sized tasks

## Non-goals

- Compete head-on with full institutional quant platforms on breadth
- Promise multi-venue live execution before it exists
- Treat experimental research, deployment, or sentiment work as part of the main launch message

## Scope

### Category Position

Neural should position as a `Kalshi-first prediction market SDK`, not as a generalized trading platform and not as a full quant operating system.

### Benchmark Categories

| Category | Representative | What it does well | Why Neural should not copy it directly |
| --- | --- | --- | --- |
| Exchange-native SDK | Kalshi developer docs and official SDKs | First-party auth, direct API access, exchange correctness | Too low-level to be the whole product story |
| Exchange-native client | Polymarket CLOB client docs | Direct authenticated CLOB access and trading methods | Good for direct venue integration, not a narrow, honest beta product story |
| Full quant engine | QuantConnect LEAN | Broad multi-asset engine, research, portfolio modeling, cloud and local workflows | Too broad and infrastructure-heavy for Neural's current wedge |
| Open-source bot framework | Freqtrade | Dry-run, backtesting, optimization, monitoring, strong operator loop | Crypto-first and exchange-heavy; useful as an operations benchmark, not a direct category target |

### Positioning Implication

Neural should win on:

- faster time to first useful Kalshi workflow
- a smaller and more honest beta surface
- paper-trading and CLI workflows that are easier to operationalize than raw exchange clients
- open-core clarity instead of pretending the whole repo is production-ready

Neural should avoid competing on:

- venue breadth
- institutional portfolio infrastructure
- hosted quant research cloud
- fully managed deployment or control-plane operations

## Recommended Messaging

### Category

Kalshi-first prediction market SDK

### One-Liner

Neural is a Kalshi-first Python SDK for prediction market trading with market data, paper trading, selected live workflows, and a lightweight CLI.

### Home Page Headline

Kalshi-first SDK for prediction market trading.

### Home Page Subhead

Build with Kalshi auth, market data, paper trading, selected live workflows, and a lightweight CLI. Join the beta waitlist or inspect the SDK on GitHub.

### Proof Points

- clean install and package smoke validation
- supported CLI with environment and capability checks
- explicit beta support matrix
- paper-trading workflow
- release checklist, release notes, and open-core boundary already documented

### Claims To Avoid

- "Build and deploy trading algorithms for sports prediction markets"
- "Works with Kalshi and Polymarket" without qualification
- "platform" language that implies hosted deployment or broad venue parity
- language that implies research modules are stable product features

## Launch Channels

### GitHub

- README as the canonical source of truth
- release notes linked from the repo
- examples and quickstart as the activation path

### Landing Page

- narrow beta message
- waitlist capture
- GitHub as the secondary CTA for technical evaluators

### Direct Outreach

- design-partner outreach to Python-first traders and small quant teams
- short benchmark-led note explaining why Neural is narrower than full bot frameworks and easier than raw exchange clients

## 30-Day Growth Loop

### Week 1

- deploy the landing page live
- validate the waitlist CTA and runtime path
- publish the beta FAQ and adoption brief

### Week 2

- instrument source attribution and CTA conversion
- validate SEO, social preview, and canonical metadata
- ship one proof-oriented example or walkthrough update

### Week 3

- run direct outreach to design partners
- collect the first objection patterns from waitlist replies and GitHub traffic
- update landing-page and FAQ copy based on those objections

### Week 4

- publish a weekly launch update
- review the acquisition and conversion data
- turn the strongest learnings into the next 30 days of PR-sized backlog

## Success Metrics

- waitlist conversion from landing visits
- GitHub click-through from the landing page
- `neural doctor` and install-path activation from launch traffic
- number of qualified design-partner conversations started
- number of public-beta objections that are answered by docs instead of manual support

## Risks

- slipping back into a broad platform pitch
- attracting users who expect venue breadth or production guarantees the beta does not offer
- spending growth time on channels before waitlist attribution and FAQ coverage exist

## Open Questions

- Whether the first public-facing proof asset should be the paper-trading walkthrough or a market-discovery walkthrough
- Whether design-partner outreach should target discretionary traders first or engineering teams first
- When to promote selected live Kalshi workflows more aggressively after credentialed validation is complete

## Sources

- Kalshi developer docs: https://docs.kalshi.com/sdks/overview
- Kalshi authenticated quickstart: https://docs.kalshi.com/getting_started/quick_start_authenticated_requests
- Polymarket authenticated client methods: https://docs.polymarket.com/developers/CLOB/clients/methods-l2
- QuantConnect LEAN overview: https://www.quantconnect.com/docs/v2/lean-engine
- Freqtrade documentation: https://docs.freqtrade.io/en/2024.12/
