# Neural Beta FAQ

## What is Neural?

Neural is a Kalshi-first Python SDK for prediction market trading. The supported beta surface covers auth, market data, paper trading, selected live workflows, and a lightweight CLI.

## Who is the beta for?

The beta is for technical users who want to automate prediction-market workflows in Python and are comfortable working with a narrow, explicit beta boundary.

## What works today?

- Kalshi auth and environment helpers
- Kalshi market data reads
- paper trading
- selected Kalshi live workflows through `TradingClient`
- `neural doctor`, `neural --version`, and clean install paths

## Does Neural support Polymarket?

Yes, but only as a beta read and streaming surface right now. Polymarket live order placement, cancellation, and order-status flows are not part of the supported beta contract.

## Does Neural support backtesting, FIX, sentiment, or deployment?

Those modules remain experimental. They are visible in the repo, but they are not part of the stable public beta promise.

## Should I use Neural for live production trading today?

Treat the beta as a narrow, technical release. The supported live path is Kalshi-first and intentionally constrained. Use the paper-trading workflow first, and do not assume broader venue or infrastructure support is production-ready.

## What does open core mean for Neural?

The free core is the Kalshi-first SDK surface. Experimental modules stay visible with clear caveats, and premium follow-on work such as managed deployment, premium data, and control-plane features stays roadmap-only for now.

## How do I get started?

Install the SDK, run `neural doctor`, configure Kalshi credentials, and start with the paper-trading and market-data workflows documented in the repo.

## How do I follow launch progress?

Use the repo release notes, waitlist, and public docs as the source of truth. The public beta scope is kept intentionally narrow so changes in support level remain visible.
