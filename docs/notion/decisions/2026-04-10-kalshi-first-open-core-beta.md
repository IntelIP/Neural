# Kalshi-First Open-Core Beta

## Decision

Ship Neural first as a Kalshi-first open-core beta, and treat broader platform, deployment, and premium control-plane ambitions as explicitly post-beta work.

## Context

The codebase contains real value, but the tested and production-ready core is narrower than the total visible surface. A truthful beta needs a strong supported center, clean docs, and an execution model that supports daily PR delivery.

## Options Considered

- Ship the broader multi-exchange platform story now.
- Delay launch until every visible module is production-grade.
- Ship a narrow, honest beta with explicit boundaries and a post-beta premium roadmap.

## Outcome

Choose the narrow, honest beta. The supported story centers on the core SDK, Kalshi-first workflows, paper trading, selected live paths, and a deployable landing page. Experimental and premium-adjacent work remains visible only with clear labeling and roadmap separation.

## Consequences

- Launch risk decreases because the product promise matches the software.
- Some ambitious features move out of the immediate story, which may feel slower in the short term.
- The open-core roadmap becomes more credible because premium ideas no longer distort the public beta scope.
