# Neural Competitive Landscape

## Purpose

Map the current competitive landscape around prediction-market developer tooling so Neural can position the public beta honestly and avoid fighting on the wrong axis.

## Market Context

The current market is splitting into two clear camps:

- exchange-native developer tooling
- unified or operator-level infrastructure on top of fragmented venues

That means a generic claim like "fastest way to trade prediction markets" is weak unless the product can prove one of two things:

- deepest integration with a single venue
- best operator workflow across multiple venues

Neural is currently stronger in the first category than the second.

## Competitor Categories

### 1. Exchange-Native APIs

#### Kalshi

Kalshi now ships official SDKs with full API coverage, RSA-PSS auth, retries, and type-safe models. Their docs explicitly recommend direct API integration or generated clients for production applications.

Implication for Neural:

- Neural should not position itself as "the Kalshi API"
- Neural should position itself as the opinionated workflow layer on top of the Kalshi API

Relevant sources:

- https://docs.kalshi.com/sdks/overview
- https://docs.kalshi.com/getting_started/api_keys
- https://docs.kalshi.com/getting_started/quick_start_create_order

#### Polymarket

Polymarket exposes public CLOB methods for market reads, prices, and order books without requiring a signer. Their developer tooling is already strong enough that a thin read wrapper is not a compelling wedge by itself.

Implication for Neural:

- read and streaming support is useful as a beta edge, but not enough to anchor the story
- Neural should avoid implying live Polymarket support until it actually exists

Relevant source:

- https://docs.polymarket.com/developers/CLOB/clients/methods-public

### 2. Unified Prediction-Market Infrastructure

#### Oddpool

Oddpool is currently a more direct unified-data competitor than most generic trading APIs. Its public docs and pricing describe cross-venue data, arbitrage tracking, search, and a single WebSocket across Kalshi and Polymarket.

Implication for Neural:

- do not lead with "cross-venue data layer"
- do not lead with arbitrage or venue-comparison claims until Neural has the execution depth to back them up
- keep Neural focused on workflow quality around the supported Kalshi-first path

Relevant sources:

- https://docs.oddpool.com/
- https://docs.oddpool.com/websocket/overview
- https://www.oddpool.com/pricing
- https://www.ycombinator.com/companies/oddpool

#### Dome

Dome's value proposition was a unified API across prediction venues. According to reporting on February 19, 2026, Polymarket acquired Dome, which suggests the unification layer is already being consolidated into exchange-owned infrastructure.

Implication for Neural:

- do not lead with "multi-venue routing" or "cross-exchange execution" right now
- if Neural eventually plays here, it needs real execution, reliability, and operator-grade routing proof

Relevant sources:

- https://www.theblock.co/post/390546/polymarket-buys-fresh-prediction-market-api-startup-dome-marking-second-official-acquisition
- https://www.banklesstimes.com/articles/2026/02/20/polymarket-acquires-dome-developer-of-unified-prediction-markets-api/

### 3. Horizontal Quant Platforms

#### QuantConnect

QuantConnect is not a direct prediction-market competitor, but it is the strongest benchmark for end-to-end quant platform positioning. Their public positioning is unified research, backtesting, and live trading infrastructure at institutional scale.

Implication for Neural:

- Neural should not present itself as a full quant platform yet
- when users compare categories, Neural wins by focus and faster prediction-market onboarding, not by matching QuantConnect breadth

Relevant source:

- https://www.quantconnect.com/

## What Neural Can Win Today

Neural should compete on:

- a narrow, honest Kalshi-first workflow
- faster operator onboarding than building directly on raw venue APIs
- one SDK surface for market data, paper trading, CLI health checks, and selected live workflows
- open-core credibility with explicit boundaries instead of vague platform claims

Neural should not compete on:

- full multi-venue live execution
- broad exchange coverage
- hosted deployment or control-plane capabilities as if they already exist
- sentiment or research automation as a launch wedge

## Positioning Consequences

### Strong Positioning

"Kalshi-first SDK for prediction-market data, paper trading, and selected live workflows."

This works because it is:

- concrete
- defensible
- differentiated enough from raw exchange docs
- aligned with the actual supported beta surface

### Weak Positioning

- "Fastest way to trade prediction markets"
- "Unified API for prediction markets"
- "Build and deploy cross-venue prediction-market algorithms"

These are weak because they either overclaim or put Neural head-to-head with stronger incumbent narratives.

## Recommended Competitive Posture

- Treat Kalshi as the primary integration partner, even if informally.
- Keep Polymarket in the story only as read and streaming beta.
- Use open-core clarity as a trust signal.
- Sell the workflow, not the fantasy platform.
- Ship reliability and docs faster than competitors ship narrative.
