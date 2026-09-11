"""Read-only views of Neural's local paper journal; never run or migrate jobs."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any

from neural.paper_worker import APPLICATION_ID, MAX_INPUT_BYTES
from neural.sports import SportsMarket, compare_sports_markets
from neural.strategy import StrategySpec, _decimal_text


class PaperJournal:
    """Inspect an existing v1 journal without a writer lock or worker startup.

    Missing files raise FileNotFoundError; malformed/unsupported journals and
    inputs raise ValueError. Historical model results remain opaque on reads.
    The caller owns the journal path and selects the verified runtime archive.
    """

    def __init__(self, database: str | Path):
        """Select an existing local journal path without opening or creating it."""
        if str(database) in ("", ":memory:"):
            raise ValueError("paper journal requires an existing database file")
        self.database = Path(database).resolve()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Open a read snapshot and reject journals outside the supported schema."""
        if not self.database.exists():
            raise FileNotFoundError("paper job database not found")
        db = None
        try:
            db = sqlite3.connect(self.database.as_uri() + "?mode=ro", uri=True, timeout=5)
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA query_only=ON")
            db.execute("BEGIN")
            if (
                db.execute("PRAGMA application_id").fetchone()[0] != APPLICATION_ID
                or db.execute("PRAGMA user_version").fetchone()[0] != 1
            ):
                raise ValueError("unsupported paper job database")
            yield db
        except sqlite3.Error as exc:
            if isinstance(exc, sqlite3.OperationalError) and str(exc) in (
                "database is locked",
                "database table is locked",
            ):
                raise
            raise ValueError("cannot read paper job database") from exc
        finally:
            if db is not None:
                db.close()

    @staticmethod
    def _view(row: sqlite3.Row) -> dict[str, Any]:
        """Decode a saved row while checking configuration identity and state."""
        try:
            if not isinstance(row["config"], str):
                raise ValueError("stored job configuration is invalid")
            config = json.loads(row["config"])
            if (
                not isinstance(config, dict)
                or hashlib.sha256(row["config"].encode()).hexdigest() != row["id"]
                or row["status"] not in ("queued", "completed", "failed")
            ):
                raise ValueError("stored job input integrity mismatch")
            view = {key: row[key] for key in ("id", "status", "error")}
            view["config"] = config
            if "result" in row.keys():
                result = json.loads(row["result"]) if row["result"] is not None else None
                if (result is not None and not isinstance(result, dict)) or (
                    (row["status"] == "completed") != (result is not None)
                ):
                    raise ValueError("stored job result is invalid")
                view["result"] = result
            return view
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("stored job data is invalid") from exc

    @classmethod
    def _job(cls, db: sqlite3.Connection, identity: str) -> dict[str, Any]:
        """Read one complete saved job from the caller's snapshot."""
        if not isinstance(identity, str):
            raise ValueError("job identity must be a string")
        row = db.execute(
            "SELECT id,status,error,config,result FROM jobs WHERE id=?", (identity,)
        ).fetchone()
        if row is None:
            raise ValueError("unknown paper job")
        return cls._view(row)

    def history(self, *, offset: int = 0, limit: int = 25) -> dict[str, Any]:
        """Newest inserted jobs first, without result traces or invented timestamps."""
        if type(offset) is not int or not 0 <= offset <= 100000:
            raise ValueError("offset must be an integer from 0 to 100000")
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer from 1 to 100")
        with self._connection() as db:
            rows = db.execute(
                "SELECT id,status,error,config FROM jobs ORDER BY rowid DESC LIMIT ? OFFSET ?",
                (limit + 1, offset),
            ).fetchall()
            return {
                "jobs": [self._view(row) for row in rows[:limit]],
                "has_more": len(rows) > limit,
            }

    def inspect(self, identity: str) -> dict[str, Any]:
        """Return saved configuration and result; do not execute or reinterpret it."""
        with self._connection() as db:
            return self._job(db, identity)

    def recording(self, identity: str) -> bytes:
        """Return original input bytes after checking size and the saved digest."""
        with self._connection() as db:
            job = self._job(db, identity)
            raw = db.execute(
                "SELECT CASE WHEN length(recording) BETWEEN 1 AND ? THEN recording END "
                "FROM jobs WHERE id=?",
                (MAX_INPUT_BYTES, identity),
            ).fetchone()[0]
            if not isinstance(raw, bytes) or hashlib.sha256(raw).hexdigest() != job["config"].get(
                "recording_sha256"
            ):
                raise ValueError("stored job recording integrity mismatch")
            return raw

    def _pair(self, first: str, second: str) -> list[dict[str, Any]]:
        """Read two distinct completed jobs from a single consistent snapshot."""
        if first == second:
            raise ValueError("choose two distinct completed experiments")
        with self._connection() as db:
            jobs = [self._job(db, identity) for identity in (first, second)]
        if any(job["status"] != "completed" for job in jobs):
            raise ValueError("choose two distinct completed experiments")
        return jobs

    @staticmethod
    def _assumptions(job: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Separate fixed replay inputs from tunable strategy thresholds and caps."""
        config = dict(job["config"])
        strategy = config.pop("strategy", None)
        StrategySpec.from_dict(strategy)
        return config, {
            key: strategy[key]
            for key in ("schema_version", "kind", "venue", "market_id", "outcome", "quantity")
        }

    def compare(self, first: str, second: str) -> list[dict[str, Any]]:
        """Compare strategy variants only when their recorded inputs/assumptions match."""
        jobs = self._pair(first, second)
        if self._assumptions(jobs[0]) != self._assumptions(jobs[1]):
            raise ValueError(
                "comparison requires the same recording, model, market, outcome, "
                "quantity, cash, fees and simulation limits"
            )
        for job in jobs:
            trace = job["result"].get("trace")
            if not isinstance(trace, list) or any(
                not isinstance(event, dict) or not isinstance(event.get("action"), str)
                for event in trace
            ):
                raise ValueError("saved comparison trace unavailable or invalid")
            with localcontext() as context:
                context.prec = 100
                total_fees = sum((self._trace_fee(event) for event in trace), Decimal(0))
            job["summary"] = {
                "total_fees": _decimal_text(total_fees),
                "fill_count": sum(event["action"] == "fill" for event in trace),
                "rejection_count": sum(event["action"] == "reject" for event in trace),
            }
        return jobs

    @staticmethod
    def _trace_fee(event: dict[str, Any]) -> Decimal:
        """Parse products of two 18-digit inputs without truncating their scale."""
        value = event.get("fees", "0")
        if (
            not isinstance(value, str)
            or re.fullmatch(r"(0|[1-9][0-9]{0,35})(\.[0-9]{1,36})?", value) is None
        ):
            raise ValueError("saved trace fees must be a finite nonnegative decimal product")
        return Decimal(value)

    def compare_markets(self, first: str, second: str) -> dict[str, Any]:
        """Inspect sports terms separately; this never compares execution costs/PnL."""
        jobs = self._pair(first, second)
        markets = [job["result"].get("sports_market") for job in jobs]
        if any(market is None for market in markets):
            raise ValueError(
                "market terms unavailable: both recordings need sports contract metadata"
            )
        comparison = compare_sports_markets(*(SportsMarket.from_dict(market) for market in markets))
        return {"comparison": comparison.to_dict(), "ids": [first, second], "markets": markets}
