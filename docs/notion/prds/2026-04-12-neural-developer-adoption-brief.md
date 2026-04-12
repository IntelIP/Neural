# Neural Developer Adoption Brief

## Problem

The repo now has strong technical scaffolding, but launch copy still needs a short narrative that explains why a developer should try Neural now instead of waiting for a broader platform.

## Users

- Python developers exploring Kalshi automation
- Prediction-market traders who want a paper-first path before live execution
- Small technical teams evaluating SDK foundations for a niche trading workflow

## Goals

- Explain the supported beta value proposition in plain language
- Make the activation path obvious
- Make the limitations explicit enough to prevent the wrong expectations

## Non-goals

- Sell hosted infrastructure
- Promise full multi-exchange live trading
- Convince non-technical users that Neural is a no-code product

## Scope

### Why Try Neural Now

- You want a narrower and more honest starting point than a full quant platform
- You want more structure than raw exchange SDK calls
- You want a paper-trading and CLI workflow before building your own operator stack

### Activation Path

1. Install the SDK
2. Run `neural doctor`
3. Configure Kalshi credentials
4. Pull live market data
5. Run the supported paper-trading workflow
6. Evaluate selected live workflows only after the paper path is working

### Current Limitations

- Kalshi is the primary supported path
- Polymarket is read and streaming only
- backtesting, FIX, sentiment, and deployment remain experimental
- live Kalshi verification still depends on final credentialed validation in CI

### Reusable Message

Neural is a Kalshi-first Python SDK for prediction market trading. The public beta is intentionally narrow: auth, market data, paper trading, selected live workflows, and a lightweight CLI. The broader repo surface stays visible, but the beta promise stays small.

## Success Metrics

- waitlist signups from qualified technical users
- developers reaching the `neural doctor` and paper-trading steps
- reduced confusion about Polymarket, deployment, and experimental modules

## Risks

- attracting users who expect a polished hosted platform
- promoting experimental modules by accident through examples or copy

## Open Questions

- Whether the first outreach asset should be a short install-and-paper-trading walkthrough
- Which early user segment converts faster: discretionary traders or technical teams
