# Neural Waitlist Onboarding Sequence

## Goal

Move new waitlist users from signup to the supported beta path without pushing them into unsupported advanced modules too early.

## Principle

The first supported workflow is:

- docs
- `neural doctor`
- paper trading
- selected live workflows later

Do not start with:

- Polymarket live execution
- FIX
- sentiment
- deployment helpers

## Sequence

### Step 1: Confirmation

Subject:

You are on the Neural beta waitlist

Body:

You are on the Neural beta waitlist.

Neural is a Kalshi-first SDK for prediction-market developers. The current beta supports auth, market data, paper trading, selected live workflows, and a lightweight CLI.

Start here:

- docs
- GitHub
- the supported beta workflow

## Step 2: Readiness Check

Subject:

Get your environment ready for Neural

Body:

Before you trade anything, verify your environment.

Recommended first steps:

```bash
pip install neural-sdk
neural doctor
neural doctor --json
```

If you plan to use trading workflows, add:

```bash
pip install "neural-sdk[trading]"
```

## Step 3: Paper-Trading First

Subject:

Run the paper-trading workflow first

Body:

The safest first workflow is paper trading.

Why:

- it exercises the supported beta path
- it keeps users away from unsupported advanced modules
- it validates market-data and order-flow assumptions before live use

Direct users to:

- `docs/trading/paper-trading`
- `docs/trading/quickstart`

## Step 4: Live Workflow Upgrade

Subject:

Move from paper to selected live workflows

Body:

Only after paper trading is working should users move to selected live workflows.

Rules:

- stay on Kalshi-first flows
- keep Polymarket limited to reads and streaming
- do not rely on FIX, sentiment, or deployment helpers as part of initial onboarding

## Step 5: Activation Prompt

Subject:

What are you trying to build with Neural?

Body:

Reply with the workflow you care about most:

- market-data monitoring
- paper trading
- selected live Kalshi workflow
- research workflow

This creates a clean signal for prioritizing roadmap and support work.

## Success Criteria

- users reach docs before asking support questions
- users run `neural doctor` before live setup
- users try paper trading before live trading
- unsupported modules stop showing up as the default onboarding path
