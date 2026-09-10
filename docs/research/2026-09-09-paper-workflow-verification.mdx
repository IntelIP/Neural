---
title: Local paper workflow verification
---

# Local paper workflow verification — 2026-09-09

Outcome: existing implementation passes the customer workflow at the installed Neural wheel / Vaticor bridge layer. No blocking defect found; no source changes needed.

## Candidate and release state

- Repository: [IntelIP/vaticor](https://github.com/IntelIP/vaticor)
- Branch: `main`; clean before and after verification.
- Exact HEAD: `8af4b3bd7e8f519adcfe274f93c643cac3a41449`.
- GitHub PR #40, “NRCL-97 Prove saved paper replay and restart scenarios”: live GitHub reports MERGED at `2026-09-08T02:11:35Z`, merge commit matches HEAD. https://github.com/IntelIP/vaticor/pull/40
- Installed Neural source commit: `9b0899d433eca225b586fb5d50d68e803b6aa22a`.
- Installed wheel SHA-256: `423a666bb44b5f39171b4295d84c9c11322233449273e64501ec1b124722718e`.
- Runtime ID: `4a06caf4f28c22acbfc1440d7642afd9f7aafbad546ef7d317d6d6459f958ef3`.

## Direct verification

Executed the existing integration checks:

```sh
bun test --timeout 60000 tests/unit/local-paper.test.ts tests/local-paper.integration.test.ts
```

Result: PASS; 15 tests; 159 assertions; zero failures; 42.94 seconds.

Verified using isolated temporary queues and the installed Neural wheel:

- Save and deduplicate immutable recordings; reject malformed, incomplete, sequence-gap and oversized input.
- Validate a Kalshi price-rule strategy; submit a saved recording; execute through Neural; reopen persisted strategy, assumptions and results.
- Clone original inputs with a strategy threshold change; preserve source results and fee/cash assumptions; compare two completed compatible experiments; reject incompatible comparison inputs.
- Recover queued work after a child worker exits after simulation and before commit; publish one result; charge fixture fees once (two fills at 0.02 each); preserve history and inspect result from subsequent bridge processes.
- Reconnect, expiry, insufficient depth and sequence-gap scenarios; origin/request validation and runtime identity rejection.

The synthetic round trip yields cash 10.56 from starting cash 10 and realized PnL 0.56. This is fixture behavior, not measured trading performance.

## Boundaries and next PM action

- NRCL-97 is already implemented and merged. Reconcile its Plane state rather than assign duplicate implementation.
- Frontend click-through and full Next.js server restart were not repeated this run. The direct check covers persistent workflow through separate bridge processes, including forced worker death, but is not a fresh browser usability acceptance.
- Existing docs explicitly limit the simulator: Kalshi only, local and loopback-only, synthetic fixtures, fill-or-kill depth handling; no partial-fill simulation, live feed compatibility, queue priority, slippage or certified exchange fee accuracy.
- The original verification performed no hosted deployment, live trading, authenticated venue call or paid experiment.
- Verification required no product code changes. This report preserves the original evidence.
