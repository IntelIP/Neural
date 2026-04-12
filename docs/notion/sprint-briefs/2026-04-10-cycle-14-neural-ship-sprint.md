# Sprint Brief: Cycle 14 Neural Ship Sprint

## Sprint Goal

Lock the public story to the real product, remove launch-breaking packaging and docs mismatches, and make the landing page and SDK release path reproducible enough to support daily PR shipping.

## Planned Issues

- `INT-179` Remove or implement the broken `neural` CLI entrypoint
- `INT-180` Narrow the default package surface to the honest open-core beta
- `INT-181` Replace silent runtime exception swallowing in core modules
- `INT-182` Prune distributable bloat from the repository and package build path
- `INT-202` Add clean-install and import smoke tests for package integrity
- `INT-203` Document issue-to-branch and PR conventions for daily shipping
- `INT-191` Audit and normalize Kalshi game-discovery helpers against real API shapes
- `INT-193` Mark Polymarket as read-only beta and hide unsupported live-trading claims
- `INT-194` Quarantine or remove the Twitter sentiment source from the default story
- `INT-217` Add end-to-end paper trading happy-path coverage
- `INT-208` Rewrite the getting-started flow around the honest Kalshi-first beta
- `INT-218` Align the landing page message to the real open-core beta scope
- `INT-226` Make the landing page build reproducible in a clean CI environment

## Dependencies

- Kalshi credential access for integration verification
- Stable package metadata for smoke tests and release dry runs
- Shared agreement on the beta scope before docs and landing updates merge

## Risks

- Older docs and examples may reintroduce unsupported claims after cleanup
- CI and release work may expose additional packaging regressions
- Landing page messaging can drift if the module-status matrix is not published quickly

## Definition Of Done

- The public beta scope is explicit and consistent.
- A clean checkout can validate the SDK and landing page paths.
- The first sprint leaves the project in a state where PRs can merge daily without re-litigating scope.
