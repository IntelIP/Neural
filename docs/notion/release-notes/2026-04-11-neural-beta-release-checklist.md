# Neural Beta Release Checklist

## Release Gate

- The supported beta scope is unchanged or explicitly documented.
- README, docs, examples, release notes, and landing-page copy agree on the Kalshi-first beta boundary.
- The module-status matrix still separates free core, experimental surfaces, and premium follow-on work.
- The low-confidence module decision log still matches what the release promotes or defers.
- Any new blocker has a corresponding Linear issue.

## Content Checklist

- A release-notes draft exists and describes the supported Kalshi-first surface.
- A beta FAQ exists and answers the first-wave boundary and support questions.
- The external beta summary stays within the supported beta boundary.
- README and adjacent docs do not present unsupported live trading or premium surfaces as launch-ready.
- Experimental surfaces are clearly marked as experimental or read-only where applicable.

## Validation Checklist

- SDK lint and targeted tests pass.
- Wheel and sdist smoke checks pass.
- A clean install works from outside the repository checkout.
- CLI health checks and JSON output remain clean.
- Nightly Kalshi verification is green or explicitly triaged.
- Landing-page `bun run check` passes.
- Landing-page `bun run build` passes.

## Launch Checklist

- Publish the package artifact.
- Verify install and CLI behavior from a clean environment.
- Confirm landing-page messaging matches the shipped SDK contract.
- Share the release notes draft and the Linear release update.

## Post-Launch Checklist

- Convert launch surprises into new Linear issues the same day.
- Update weekly notes if the release changes scope, blockers, or the release boundary.
- Carry any unresolved beta mismatch into the next release-readiness pass.
