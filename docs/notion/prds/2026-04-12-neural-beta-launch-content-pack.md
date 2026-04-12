# Neural Beta Launch Content Pack

## One-Liner

Kalshi-first open-core SDK for prediction-market developers.

## Short Product Summary

Neural is a Python SDK for building prediction-market trading workflows without starting from raw exchange clients. The current beta focuses on Kalshi auth, market data, paper trading, selected live workflows, and a lightweight CLI. Polymarket reads and streaming remain available in beta, while broader sentiment, FIX, and deployment surfaces stay experimental.

## Homepage Summary

Build prediction-market trading workflows with a Kalshi-first SDK for auth, market data, paper trading, selected live flows, and CLI-based readiness checks.

## GitHub Summary

Neural is a Kalshi-first SDK beta for prediction-market developers. It gives you a narrow, honest workflow layer for auth, market data, paper trading, and selected live execution, while keeping broader research and platform surfaces clearly experimental.

## External Launch Post Draft

### Headline

Introducing Neural: a Kalshi-first SDK for prediction-market trading workflows

### Body

We are opening Neural in public beta as a Kalshi-first SDK for prediction-market developers.

The goal is simple: make it easier to go from credentials and market data to paper trading and selected live workflows without rebuilding the same plumbing from raw exchange clients every time.

What the beta supports today:

- Kalshi auth and environment helpers
- market-data and historical-data workflows
- paper trading
- selected live trading flows through `TradingClient`
- a lightweight CLI for environment and capability checks

What is still outside the supported beta promise:

- Polymarket live order placement
- broader multi-exchange execution
- sentiment and social-data workflows
- FIX as a stable default path
- deployment and control-plane tooling

We are intentionally shipping the narrowest version we can defend. If you want to build on prediction markets and start with a Kalshi-first path, Neural is ready for that beta workflow now.

## Waitlist Confirmation Draft

Subject:

Neural beta waitlist confirmed

Body:

You are on the Neural beta waitlist.

Neural is a Kalshi-first SDK for prediction-market trading workflows. The current beta supports auth, market data, paper trading, selected live workflows, and a lightweight CLI. We will follow up with access details, docs, and onboarding steps as the beta expands.

## FAQ

### What does Neural support today?

Kalshi auth, market data, paper trading, selected live workflows, and a lightweight CLI.

### Does Neural support Polymarket?

Yes, but only in a limited beta capacity for reads and streaming. Live order placement is not part of the supported beta.

### Is Neural a full multi-exchange trading platform?

No. The current beta is intentionally narrow and Kalshi-first.

### Is Neural production-ready?

Neural is in public beta. The supported path is strong enough for developer adoption, but some operational gates, especially credentialed live verification, are still being hardened.

### Does Neural include deployment and hosted execution?

Not as part of the supported beta. Those remain experimental or roadmap work.

### What should I try first?

Start with the docs, run `neural doctor`, and use the paper-trading workflow before moving to live trading.

## Messaging Rules

- Lead with Kalshi-first.
- Lead with SDK, not platform.
- Lead with prediction-market workflows, not sports.
- Mention Polymarket only with explicit beta limits.
- Do not promote deployment, sentiment, or FIX as launch-ready.
