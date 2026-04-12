# Neural Launch And Growth Plan

## Problem

Neural is close to finishing core beta development, but growth work will fail if the product promise stays broader than the actual supported SDK surface.

The launch plan needs to convert current engineering progress into:

- a clear public narrative
- a live waitlist funnel
- a repeatable beta launch motion
- a 30-day growth backlog after ship

## Launch Thesis

Ship Neural as a Kalshi-first developer SDK for prediction-market trading workflows.

Do not launch Neural as:

- a sports-only product
- a full multi-exchange execution platform
- a full hosted deployment product

## Ideal Users

### Primary User

- independent developer or quant who wants to build on Kalshi without starting from raw exchange clients

### Secondary User

- prediction-market trader who wants paper-trading, market-data, and lightweight automation scaffolding

### Tertiary User

- small team evaluating whether Neural can become the workflow layer before paying for premium data, hosting, or operator tooling later

## Core Positioning

### One-Liner

Kalshi-first open-core SDK for prediction-market developers.

### Expanded Message

Neural helps developers move from auth and market data to paper trading and selected live trading flows with a narrow, honest SDK beta. Polymarket reads and streaming remain available in beta, while broader sentiment, FIX, and deployment surfaces stay experimental.

## Proof Points

- clean package install and import smoke path
- supported CLI with `doctor` output
- Kalshi-first docs and examples
- explicit support matrix
- explicit low-confidence module triage
- release checklist and release-notes workflow

## Launch Risks

- landing page still drifting toward broader or older messaging
- no live Kalshi credential validation until account access returns
- experimental modules creating confusion if they leak into launch content

## What “Development Complete” Means For Beta

Neural can move into growth mode when these conditions hold:

- landing page is live and captures waitlist signups
- package release path is validated
- docs and landing-page copy match the Kalshi-first support matrix
- deferred modules are out of the default story
- live Kalshi verification is either done or explicitly called out as the only remaining operational gate

## Launch Assets

### Required

- landing page with working waitlist form
- README and docs hero aligned to the beta promise
- beta release notes
- release checklist
- open-core feature matrix
- low-confidence module decision log

### Next

- comparison page or doc: Neural vs official SDKs and data vendors
- concise external launch post
- beta FAQ
- onboarding email for waitlist users

## Growth User Stories

### Messaging And Conversion

- As a first-time visitor, I can understand what Neural supports in under 10 seconds.
- As a skeptical developer, I can tell what is supported beta versus experimental without reading the whole docs site.
- As an interested visitor, I can join the beta waitlist directly from the homepage.
- As a GitHub visitor, I can move from landing page to repo and docs without friction.

### Developer Adoption

- As a new developer, I can compare Neural with official exchange SDKs and understand why Neural exists.
- As a Kalshi-first builder, I can copy a short supported workflow and get to paper or live testing quickly.
- As a cautious user, I can see clear warnings around Polymarket, FIX, deployment, and sentiment surfaces.

### Growth Operations

- As a launch owner, I can attribute waitlist signups to source and campaign.
- As a product lead, I can publish one clear weekly update on traction, blockers, and next launch actions.
- As an operator, I can convert launch surprises into PR-sized backlog quickly.

## Immediate Growth Work

### Phase 1: Launch Surface

- align landing-page messaging to the support matrix
- make the waitlist API usable from the homepage
- validate source attribution for signups
- publish the beta release notes and landing-page copy together

### Phase 2: Developer Acquisition

- publish a comparison brief against Kalshi, Polymarket, Oddpool, and QuantConnect
- publish a short launch post focused on Kalshi-first developer workflows
- push one example-driven GitHub and docs update per week

### Phase 3: Conversion

- email waitlist signups with docs, GitHub, and beta access instructions
- instrument source and campaign tags on the waitlist
- create a short onboarding path for “paper trade first, then live”

### Phase 4: Post-Launch Expansion

- open-core upsell pages for premium data, hosted workflows, and operator tooling
- tighter comparison content once live Kalshi verification is green
- convert traction signals into the next 30-day backlog

## Recommended Growth Issues

- Align landing-page copy and CTA flow to the Kalshi-first beta
- Add homepage waitlist capture and basic source attribution
- Publish competitor-informed positioning and comparison content
- Create the external beta launch content pack
- Add waitlist analytics and campaign fields
- Publish beta FAQ and onboarding email draft

## Metrics

### Launch Metrics

- landing-page conversion rate to waitlist
- docs click-through rate from landing page
- GitHub click-through rate from landing page

### Activation Metrics

- number of users who run `neural doctor`
- docs visits to trading quickstart
- beta users who reach paper-trading workflow

### Narrative Health

- support requests caused by unsupported-feature confusion
- percentage of launch content that stays inside the support matrix
- number of growth assets that require rewording after engineering review

## Sources

- `README.md`
- `docs/notion/prds/2026-04-11-neural-open-core-feature-matrix.md`
- `docs/notion/decisions/2026-04-11-low-confidence-module-triage.md`
- `docs/notion/release-notes/2026-04-11-neural-beta-release-notes.md`
- `public/Neural-Landing-Page/src/pages/index.astro`
- `public/Neural-Landing-Page/src/pages/api/waitlist.ts`
