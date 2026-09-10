"""Bounded, offline price-rule simulation. Not an exchange execution engine."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any

from neural.kalshi import _wire
from neural.recordings import (
    MAX_SOURCE_AGE_SECONDS,
    RECORD_VERSION,
    read_recording_metadata,
    replay_book_recording,
)
from neural.strategy import StrategySpec, _decimal, _decimal_text

PAPER_MODEL = "neural-paper/2"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def simulate_recording(
    spec: StrategySpec,
    path: str | Path,
    *,
    initial_cash: str,
    fee_per_contract: str,
    max_order_age_seconds: int = 30,
    max_events: int = 10000,
) -> dict[str, Any]:
    """Run at most one full buy/exit cycle; return only after validating EOF.

    Signals execute against the next book, fill-or-kill, within the same session.
    Fees are a caller-supplied assumption on each side, not a venue fee schedule.
    """
    metadata = read_recording_metadata(path)
    if (spec.venue, spec.market_id) != (metadata["venue"], metadata["market_id"]):
        raise ValueError("recording venue/market does not match strategy")
    if metadata["outcome"] is not None and metadata["outcome"] != spec.outcome:
        raise ValueError("recording outcome does not match strategy")
    normalized = metadata["version"] == RECORD_VERSION
    for name, value in (
        ("max_order_age_seconds", max_order_age_seconds),
        ("max_events", max_events),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    cash = _decimal(initial_cash, "initial_cash")
    fee = _decimal(fee_per_contract, "fee_per_contract")
    with localcontext() as context:
        context.prec = 80
        if spec.quantity % Decimal("0.01"):
            raise ValueError("paper quantity must align to the model's 0.01 lot")
        position = cost = realized = Decimal(0)
        entered = False
        pending: tuple[str, datetime] | None = None
        previous: datetime | None = None
        trace: list[dict[str, Any]] = []
        digest = hashlib.sha256()
        if normalized:
            digest.update((_canonical(metadata) + "\n").encode())
        books = count = 0
        for count, event in enumerate(replay_book_recording(path, expected_metadata=metadata), 1):
            if count > max_events:
                raise ValueError("recording exceeds max_events")
            if previous is not None and event.received_at < previous:
                raise ValueError("recording receive timestamps must not regress")
            previous = event.received_at
            digest.update((_canonical(_wire(asdict(event))) + "\n").encode())
            row: dict[str, Any] = {"event": count, "received_at": event.received_at.isoformat()}
            if event.kind == "reset":
                row.update(action="cancel" if pending else "reset", reason=event.reason)
                pending = None
                trace.append(row)
                continue
            assert event.update is not None
            update = event.update
            book = update.book
            books += 1
            if book.ticker != spec.market_id:
                raise ValueError("recorded ticker does not match strategy")
            bids = book.yes_bids if spec.outcome == "yes" else book.no_bids
            asks = book.asks(spec.outcome)
            if bids and asks and bids[0].price > asks[0].price:
                raise ValueError("crossed recorded book")
            row.update(sid=update.sid, seq=update.seq)
            if normalized:
                row.update(
                    source_at=update.source_at.isoformat() if update.source_at else None,
                    quality="full_depth",
                )
            if pending is not None:
                side, created = pending
                pending = None
                row.update(action="cancel", side=side)
                if (event.received_at - created).total_seconds() > max_order_age_seconds:
                    row["reason"] = "order_expired"
                elif update.source_at is not None and update.source_at <= created:
                    row["reason"] = "source_not_after_intent"
                else:
                    remaining = spec.quantity
                    notional = Decimal(0)
                    levels = []
                    for level in asks if side == "buy" else bids:
                        if (side == "buy" and level.price > spec.entry_price) or (
                            side == "sell" and level.price < spec.exit_price
                        ):
                            break
                        quantity = min(remaining, level.quantity)
                        notional += quantity * level.price
                        remaining -= quantity
                        levels.append(
                            {
                                "price": _decimal_text(level.price),
                                "quantity": _decimal_text(quantity),
                            }
                        )
                        if not remaining:
                            break
                    fees = fee * spec.quantity
                    if remaining:
                        row["reason"] = "insufficient_executable_depth"
                    elif side == "sell" and cash + notional - fees < 0:
                        row["reason"] = "insufficient_cash_for_fees"
                    else:
                        if side == "buy":
                            cost = notional + fees
                            cash -= cost
                            position = spec.quantity
                            entered = True
                        else:
                            cash += notional - fees
                            realized += notional - fees - cost
                            position = cost = Decimal(0)
                        row.update(
                            action="fill",
                            levels=levels,
                            fees=_decimal_text(fees),
                            notional=_decimal_text(notional),
                        )
            elif not entered and asks and asks[0].price <= spec.entry_price:
                reserve = spec.quantity * (spec.entry_price + fee)
                reason = None
                if position + spec.quantity > spec.max_position:
                    reason = "max_position"
                elif cost + reserve > spec.max_exposure_usd:
                    reason = "max_exposure_usd"
                elif reserve > cash:
                    reason = "insufficient_cash"
                if reason:
                    row.update(action="reject", side="buy", reason=reason)
                else:
                    pending = ("buy", event.received_at)
                    row.update(action="intent", side="buy", reserved_cash=_decimal_text(reserve))
            elif position and bids and bids[0].price >= spec.exit_price:
                pending = ("sell", event.received_at)
                row.update(action="intent", side="sell")
            else:
                row["action"] = "hold"
            trace.append(row)
        if not books:
            raise ValueError("recording contains no books")
        report: dict[str, Any] = {
            "model": PAPER_MODEL,
            "strategy": spec.to_dict(),
            "strategy_id": spec.version_id,
            "assumptions": {
                "initial_cash": _decimal_text(_decimal(initial_cash, "initial_cash")),
                "fee_per_contract": _decimal_text(fee),
                "max_order_age_seconds": max_order_age_seconds,
                "lot": "0.01",
                "execution": "next-book fill-or-kill; one buy/exit cycle; no settlement",
            },
            "recording_digest": digest.hexdigest(),
            "event_count": count,
            "trace": trace,
            "cash": _decimal_text(cash),
            "position": _decimal_text(position),
            "acquisition_cost": _decimal_text(cost),
            "realized_pnl": _decimal_text(realized),
        }
        if normalized:
            report["sports_market"] = metadata["sports_market"]
            report["market_compatibility"] = {
                "status": "unknown",
                "reason": "comparison_requires_second_market",
            }
            report["recording"] = {
                key: value for key, value in metadata.items() if key != "sports_market"
            }
            report["assumptions"]["max_source_age_seconds"] = MAX_SOURCE_AGE_SECONDS
            report["assumptions"]["data_quality"] = "synthetic full-depth snapshots"
        report["result_id"] = hashlib.sha256(_canonical(report).encode()).hexdigest()
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", type=Path, required=True)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--cash", required=True)
    parser.add_argument("--fee-per-contract", required=True)
    parser.add_argument("--max-order-age-seconds", type=int, default=30)
    args = parser.parse_args()
    try:
        spec = StrategySpec.from_json(args.strategy.read_text())
        result = simulate_recording(
            spec,
            args.recording,
            initial_cash=args.cash,
            fee_per_contract=args.fee_per_contract,
            max_order_age_seconds=args.max_order_age_seconds,
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(_canonical(result))


if __name__ == "__main__":
    main()
