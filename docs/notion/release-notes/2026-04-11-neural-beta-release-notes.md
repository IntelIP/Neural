# Neural Beta Release Notes

## Summary

Neural's public beta is now explicitly Kalshi-first. The supported surface is limited to Kalshi auth, market data, paper trading, selected live trading flows, the CLI, and the documented install path. Experimental surfaces remain visible for context, but they are not part of the public beta promise.

## User-Visible Changes

- The public story now centers on Kalshi-first authentication and market-data workflows.
- Paper trading and the supported live trading flows are called out as the beta execution path.
- The CLI and supported install path are part of the public beta surface.
- Polymarket US is described as read and stream only in the beta story.
- Backtesting, FIX helpers, sentiment, and deployment remain documented as experimental rather than launch-ready.

## Operational Changes

- Release readiness now depends on docs, examples, packaging smoke checks, and landing-page alignment.
- Kalshi credentialed verification is part of the release motion.
- The release process now expects a release checklist, a release-notes draft, and a concise external beta summary.
- Landing-page copy must stay aligned with the supported SDK contract.

## Known Gaps

- Polymarket live order placement is not supported in the public beta.
- Backtesting helpers remain experimental.
- FIX helpers, sentiment tooling, and deployment helpers remain outside the supported beta surface.
- Premium and control-plane work stay roadmap-only.

## Follow-Up Work

- Convert any beta mismatch into a Linear issue before the next release.
- Refresh smoke checks if the supported beta surface changes.
- Revisit the open-core boundary only after the current beta path remains stable across docs, packaging, and launch surface.
