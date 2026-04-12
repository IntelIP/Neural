# Neural Beta Launch FAQ

## What is Neural?

Neural is a Kalshi-first SDK for prediction-market data, paper trading, and selected live workflows.

## Is Neural production-ready?

Neural is in public beta. The supported path is intentionally narrow and centered on Kalshi-first workflows. Some adjacent surfaces remain experimental.

## What works today?

- Kalshi auth
- Kalshi market data
- paper trading
- selected live workflows through `TradingClient`
- CLI environment and capability checks

## Does Neural support Polymarket?

Partially. Polymarket support is currently limited to reads and streaming beta workflows. Live order placement, cancel, and order-status flows are not part of the supported beta surface.

## Does Neural support backtesting?

Backtesting helpers exist, but they remain experimental relative to the supported trading path.

## Does Neural include sentiment or social-data trading?

Experimental only. Sentiment and social-data tooling are not part of the supported beta story.

## Does Neural handle deployment and hosted execution?

Not as part of the public beta promise. Deployment helpers exist in the repo as experimental work, but the hosted or managed control-plane story remains future work.

## Why is the scope so narrow?

Because the launch promise should match the software that is actually reliable. Neural is choosing a smaller, more defensible beta instead of pretending the whole repo is production-ready.

## Who should join the beta?

- quant developers
- small trading teams
- builders who want a workflow layer on top of Kalshi

## What is the fastest way to evaluate Neural?

1. Install the package
2. Run `neural doctor`
3. Read one Kalshi market
4. Run one paper-trading workflow

## What happens after beta?

After the Kalshi-first beta is stable, the roadmap expands into better activation, broader validation, and eventually premium follow-on surfaces such as managed workflows, premium data, and operator tooling.
