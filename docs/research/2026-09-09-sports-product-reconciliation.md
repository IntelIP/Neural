---
title: Sports product reconciliation
---

# Sports product reconciliation — September 9, 2026

## Outcome

The three authorized workstreams are complete: Plane reconciliation, local paper workflow verification, and second-venue research. No product code changes were necessary. This report records the PM decision and links evidence; it is not a release or deployment receipt.

## Product goal and boundaries

Help sports strategy developers compare matched propositions and settlement differences, test strategies reproducibly, and operate them across supported U.S. venues. Neural owns normalized contracts, recording, replay and execution; Vaticor owns user workflows and hosted operations. Initial user is provisionally the strategy developer. MLB pregame moneylines are a research sample, not a committed launch sport.

## Plane reconciliation

- NRCL-92–94 reconciled to Done after GitHub verified PR #38 merged, with existing local acceptance evidence.
- NRCL-95–96 reconciled to Done after GitHub verified PR #39 merged, with saved-recording and clone evidence.
- NRCL-97 reconciled to Done after PR #40 merge verification and current direct integration tests.
- NRCL-80–85 canceled as superseded paper-to-action and retirement work. Historical records retained; no infrastructure, repository or website removed.
- Project description updated to sports-first and conditional second-venue selection.
- Missing native dates/estimates on NRCL-92–96 and NRCL-84–85 were explicitly assigned to the September 9 reconciliation scope (1 point); they do not represent reconstructed historical build estimates or original delivery promises. Existing populated metadata was preserved.
- Confirmed final non-archived state: 18 Done, 7 Canceled, 3 Backlog, zero Started. All changed records were read back.

## Work now captured

| Story | State | Result or next outcome |
|---|---|---|
| [NRCL-98](https://app.plane.so/intelligent-intellectual-property/projects/7ffb1939-48c5-4a7a-a4be-6cd283fc8274/issues/64f56165-13e3-45bb-aad7-0cee0e1ff0aa) | Done | Second-venue feasibility brief and bounded public evidence |
| [NRCL-99](https://app.plane.so/intelligent-intellectual-property/projects/7ffb1939-48c5-4a7a-a4be-6cd283fc8274/issues/730de44e-7f65-4c6b-95f7-3501f4e00d71) | Backlog | Match sports propositions and expose settlement differences |
| [NRCL-100](https://app.plane.so/intelligent-intellectual-property/projects/7ffb1939-48c5-4a7a-a4be-6cd283fc8274/issues/8af2a8b1-5b4c-4ae2-8374-63ced7293206) | Backlog | Run an unchanged paper strategy on Polymarket US recordings |
| [NRCL-101](https://app.plane.so/intelligent-intellectual-property/projects/7ffb1939-48c5-4a7a-a4be-6cd283fc8274/issues/9b7fc695-f593-4c09-9cec-eccb09723fde) | Backlog | Resolve venue access and compare pregame execution costs |

NRCL-100 and NRCL-101 have native blocked-by links to NRCL-98 and NRCL-99. NRCL-98 is complete; rule matching remains the open dependency. Drafting access questions can proceed before the matched-price portion of NRCL-101. Future backlog was captured, not committed or implemented in this turn.

## Direct verification

[Paper workflow report](2026-09-09-paper-workflow-verification.md): clean Vaticor main 8af4b3bd7e8f519adcfe274f93c643cac3a41449, installed Neural source 9b0899d433eca225b586fb5d50d68e803b6aa22a. Existing local tests: 15 passed, 159 assertions, zero failures. Covers saved recording reuse, strategy configuration, execution, clone/compare, reopening and worker crash recovery. No fresh browser walkthrough or full application restart repeated. Kalshi synthetic paper only; no hosted/live readiness claimed.

## Venue decision

[Research brief](2026-09-09-sports-venue-decision.md): Polymarket US is the provisional second data/paper adapter on access and integration readiness. Novig remains an active candidate. Documented MLB forfeit and postponement differences mean equal teams/date do not establish equivalent economic contracts. No live venue winner or profitability claim.

## Next pull

Refine NRCL-99 for one agreed sport and exact contract sources. Once the rule contract is ready, NRCL-100 adapter work and NRCL-101 access/cost validation can run beside each other. Keep at most three started stories; keep hosted paper and live execution as subsequent bounded milestones.
