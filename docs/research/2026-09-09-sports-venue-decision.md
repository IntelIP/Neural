---
title: Sports venue decision brief
---

# Sports venue decision brief — Neural / Vaticor

Research date: September 9, 2026 America/New_York. Public API observations occurred September 10, 2026 UTC (still September 9 Eastern). Decision: choose the next bounded data/paper integration alongside the existing Kalshi work. MLB full-game pregame moneylines are a provisional research sample, not a committed product-sport decision.

## Recommendation

**Advance Polymarket US as the provisional second data/paper adapter. Keep Novig as an active candidate whose API access and commercial fit must be resolved before a live adapter commitment.** This is an integration-readiness recommendation, not a finding that Polymarket US offers better prices or profitability.

Reasons: Polymarket US offers working unauthenticated sports discovery and books; Neural already references Polymarket US normalization examples; its current docs explicitly describe an ISV route for hosted customer workflows. Novig has meaningful observed MLB activity and attractive documented pregame trading fees, but the current public evidence does not establish a Vaticor customer-account automation model. Final live venue selection remains open.

Local fit verified in `README.md`: stable product is a deterministic, dependency-free kernel beneath Vaticor; Kalshi data uses decimal prices and fractional depth; existing Polymarket US examples are listed. `docs/trading/recorded-paper.mdx` states recordings currently support Kalshi only. No implementation or live readiness inferred from those descriptions. 

## Facts that change the plan

| Dimension | Polymarket US | Novig | Implication |
|---|---|---|---|
| Read access | Public gateway exposes events, sports, books and display-price history without a key; direct discovery/book requests worked | Public daily market/trade CSVs require no authentication; NBX trading credentials must be requested | Polymarket US has the clearer immediate live-data path |
| Hosted customer model | Documented ISV/IB integration: firm acts for onboarded retail participants; signed agreements and RSA/JWT credentials; preproduction test funds | Affiliate documentation supports odds display and deeplinks; NBX credentials alone do not establish hosted customer-account rights | Ask which commercial model fits Vaticor; do not assume personal keys authorize a hosted service |
| Pregame straight-contract fee | Current taker formula: `0.06 × contracts × p × (1-p)`; $1.50 for 100 at $0.50 | Straight fees apply only to fills matched during `OPEN_INGAME`; no trading fee for a pregame fill | Compare executable cost, not fee schedule alone |
| History | Display-price endpoint plus public time-and-sales and end-of-day reports | Daily market census and executed-trade CSVs | Neither price history nor daily volume provides historical queue/depth replay |

Sources: [Polymarket US API](https://docs.polymarket.us/api-reference/introduction), [ISV eligibility](https://docs.polymarket.us/partners/partner-types/isvs), [partner responsibilities](https://docs.polymarket.us/partners/your-role), [partner onboarding](https://docs.polymarket.us/partners/get-connected/onboarding), [Novig authentication](https://docs.novig.com/api-reference/authentication), [Novig affiliate integration](https://docs.novig.com/affiliates/overview), [Polymarket US fees](https://docs.polymarket.us/fees), [Novig fees](https://docs.novig.com/fees).

Polymarket US's partner route is more concrete than our earlier blanket statement that hosted access was unknown for both candidates. It is **not** evidence that IntelIP is approved or that unattended customer strategy execution and connection to pre-existing accounts are included. Its documented partner flow also includes customer onboarding and funding integration, which may exceed our initial model. Separately, institutional market-data onboarding requires a Market Data Agreement; public reachability does not establish storage, redistribution, replay or resale rights. [Data onboarding](https://docs.polymarket.us/data-guide/onboarding)

## Contract equivalence has a demonstrated blocker

The current Novig **MLB Winner Series**, public Appendix A, differs from Polymarket US's sports guidance:

- **Forfeit without on-field result:** Novig settles to void; Polymarket US awards the winner of the forfeit.
- **Postponement:** Novig's ordinary window is 48 hours, or 45 days for postseason, subject to formal date updates and specified exceptions. Polymarket US generally uses rescheduling before contract expiration, described as typically two weeks.
- **Exceptional payout:** Novig voids use its defined fair-value process. Polymarket US uses last fair market price for specified contingencies. Neither should be modeled as an automatic refund or always-$0/$1 settlement.

Sources: [Novig contract directory — MLB Winner Series, Appendix A pages 2–6](https://support.novig.com/en/articles/16083642-contracts), [Polymarket US sports FAQ](https://docs.polymarket.us/faqs/sports-faqs).

**Matching teams, date and winner proposition is insufficient to claim economic equivalence.** First milestone should surface matched sporting propositions with explicit rule differences. Strict equivalence requires reviewing the exact listed contracts on both venues, including Kalshi; this study did not establish a strictly equivalent three-venue pair. Do not label resulting price gaps risk-free arbitrage.

## Bounded public-data evidence

**Novig:** Manifest lists both trade and market files for all seven dates September 2–8. September 8 `markets.csv` downloaded successfully (6,453,454 bytes). Filtering exact `reportTicker == MLB-MONEY` produced **33 listed market IDs, 31 with positive daily volume, and 4,591,596.11 contracts traded**. Sum uses decimal `dailyVolume`, one row per market. This includes pregame and live activity; the census contains markets beyond a single day's games and lacks event/team/start-time metadata. It demonstrates activity, not executable liquidity or pregame volume. The trade CSV exceeded the bounded 30 MB request cap, so no trade-level result is claimed. [Manifest](https://data.novig.com/reporting/trade-data/index.json), [September 8 market report](https://data.novig.com/reporting/trade-data/2026-09-08/markets.csv), [schema](https://docs.novig.com/api-reference/trade-data)

**Polymarket US:** Unauthenticated league discovery returned MLB markets, including Tampa Bay–Atlanta scheduled September 10 at 16:15 UTC. At **03:44:56 UTC**, its pregame full-game winner book reported **$0.475 bid / $0.480 ask**, top bid quantity **427**, top ask quantity **166,401.72**, with book timestamp **03:44:48.798724014 UTC**. This is one displayed snapshot, not a fill guarantee, sustained liquidity measurement, or comparison with Novig. Decimal depth appeared in the response and must be preserved. [Discovery request](https://gateway.polymarket.us/v2/leagues/mlb/events?limit=3), [sample book](https://gateway.polymarket.us/v1/markets/aec-mlb-tb-atl-2026-09-10/book)

History distinction: Polymarket US `longPrice`/`shortPrice` are book-derived display prices, may sum above one, and are not trade prints. Its time-and-sales report has price/size/time/symbol without aggressor side. Novig trades have maker and taker rows; count only taker rows for trade count/notional volume. None is sufficient alone for fill-quality backtests. [Price history](https://docs.polymarket.us/api-reference/price-history/get-price-history), [time-and-sales](https://docs.polymarket.us/faqs/execution-tape), [daily report](https://docs.polymarket.us/faqs/eod-reporting)

## Concrete work to pull next

1. **Sports proposition matching and settlement differences.** Customer can compare the same game/outcome while seeing material rule differences. Define versioned event/team identities, doubleheader game number, full-game versus segment, extra innings, listed rule source/version, and `compatible / different / unknown` result. Acceptance fixtures must cover ordinary winner, forfeit, postponed game, duplicate team matchup and partial-game exclusion. Unknown rules prevent an equivalence claim, not market discovery.
2. **Polymarket US recording adapter for the existing paper workflow.** Reuse existing discovery/normalization code where suitable. Preserve raw venue IDs, decimal prices/depth, observation/exchange timestamps, book quality, and schema version. Explicitly handle one-sided/missing/stale books. Keep historical display prices separate from trades/books. Acceptance: one sports recording runs through the existing strategy contract and Vaticor experiment comparison without strategy changes; unsupported fill assumptions remain visible.
3. **Second-venue commercial and matched-market trial decision.** Record answers for API eligibility, customer-owned/existing-account model, unattended automation, hosting/geography, data retention/display/replay rights, fees and credential scopes. Draft questions can proceed now; sending messages was not authorized in this stream. Once access is available, compare matched propositions at the same pregame observation times and fixed 10/100/1,000-contract sizes, reporting price plus fees, depth and freshness, with rule differences beside the result. Choose or defer the live adapter from those results.

Novig-specific adapter requirements if selected: OAuth renewal; separate QA configuration; quantity conversion (`100` wire units per contract versus contract units in CSVs); fees rounded to ledger precision; snapshot plus stream recovery; venue lifecycle transitions and cancellations; accepted order versus reconciled fill distinction. Preserve raw capability differences instead of expanding the stable kernel into a broad broker abstraction. [Fees and units](https://docs.novig.com/fees), [order-book semantics](https://docs.novig.com/api-reference/WSS/orderbook-channel), [order placement](https://docs.novig.com/api-reference/orders/place-order)

## Remaining decision evidence

Unknown: comparative executable prices across matched venues; stable pregame depth; exact contract equivalence with Kalshi; commercial permissions and onboarding cost for Vaticor; Novig credential eligibility; production adapter behavior. These affect future live selection. They do not block sports matching fixtures or a bounded Polymarket US paper/data milestone.

The research stream used public read-only sources; it made no trading or account changes.
