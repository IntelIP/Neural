"""Local durable offline paper jobs. No venue execution or hosted scheduling."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from neural.paper import PAPER_MODEL, _canonical, simulate_recording
from neural.strategy import StrategySpec, _decimal, _decimal_text

MAX_INPUT_BYTES = 32 * 1024 * 1024
APPLICATION_ID = 0x4E505731


class PaperJobs:
    """Single-host SQLite queue; use a private local filesystem, not NFS.

    Process death rolls back the active transaction. Pure simulation may run
    again after a crash, but one terminal result is committed per input identity.
    """

    def __init__(self, database: str | Path):
        self.database = str(database)
        if self.database in ("", ":memory:"):
            raise ValueError("durable jobs require a database file")
        with self._transaction() as db:
            app = db.execute("PRAGMA application_id").fetchone()[0]
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if (
                app == 0
                and version == 0
                and not db.execute(
                    "SELECT 1 FROM sqlite_master WHERE name NOT GLOB 'sqlite_*' LIMIT 1"
                ).fetchone()
            ):
                db.execute("""CREATE TABLE jobs (
                    id TEXT PRIMARY KEY, config TEXT NOT NULL, recording BLOB NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('queued','completed','failed')),
                    result TEXT, error TEXT,
                    CHECK ((status='queued' AND result IS NULL AND error IS NULL)
                        OR (status='completed' AND result IS NOT NULL AND error IS NULL)
                        OR (status='failed' AND result IS NULL AND error IS NOT NULL))
                )""")
                db.execute(f"PRAGMA application_id={APPLICATION_ID}")
                db.execute("PRAGMA user_version=1")
            elif app != APPLICATION_ID or version != 1:
                raise ValueError("unsupported paper job database")

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.database, timeout=5, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def submit(
        self,
        spec: StrategySpec,
        recording: str | Path,
        *,
        initial_cash: str,
        fee_per_contract: str,
        max_order_age_seconds: int = 30,
        max_events: int = 10000,
    ) -> str:
        """Snapshot input bytes and enqueue once; identical submissions share ID."""
        if spec.venue != "kalshi":
            raise ValueError("paper jobs support kalshi only")
        for name, value, ceiling in (
            ("max_events", max_events, 10000),
            ("max_order_age_seconds", max_order_age_seconds, 86400),
        ):
            if type(value) is not int or not 0 < value <= ceiling:
                raise ValueError(f"{name} must be an integer from 1 to {ceiling}")
        with open(recording, "rb") as source:
            data = source.read(MAX_INPUT_BYTES + 1)
        if not data or len(data) > MAX_INPUT_BYTES:
            raise ValueError("recording must be nonempty and at most 32 MiB")
        config = _canonical(
            {
                "model": PAPER_MODEL,
                "strategy": spec.to_dict(),
                "initial_cash": _decimal_text(_decimal(initial_cash, "initial_cash")),
                "fee_per_contract": _decimal_text(_decimal(fee_per_contract, "fee_per_contract")),
                "max_order_age_seconds": max_order_age_seconds,
                "max_events": max_events,
                "recording_sha256": hashlib.sha256(data).hexdigest(),
            }
        )
        identity = hashlib.sha256(config.encode()).hexdigest()
        with self._transaction() as db:
            db.execute(
                "INSERT INTO jobs(id,config,recording,status) VALUES(?,?,?,'queued') "
                "ON CONFLICT(id) DO NOTHING",
                (identity, config, data),
            )
        return identity

    def inspect(self, identity: str) -> dict[str, Any]:
        """Return persisted state; never starts a job."""
        with self._transaction() as db:
            row = db.execute(
                "SELECT id,status,result,error FROM jobs WHERE id=?", (identity,)
            ).fetchone()
            if row is None:
                raise ValueError("unknown paper job")
            return self._view(row)

    @staticmethod
    def _view(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "status": row["status"],
            "error": row["error"],
            "result": json.loads(row["result"]) if row["result"] else None,
        }

    def run_next(self) -> dict[str, Any] | None:
        """Commit one queued result, or return None. A crash leaves it queued."""
        # ponytail: one writer holds the lock during replay; shard databases or
        # introduce leases only when bounded single-host throughput is inadequate.
        with self._transaction() as db:
            row = db.execute(
                "SELECT * FROM jobs WHERE status='queued' ORDER BY rowid LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            try:
                config = json.loads(row["config"])
                if (
                    hashlib.sha256(row["config"].encode()).hexdigest() != row["id"]
                    or hashlib.sha256(row["recording"]).hexdigest() != config["recording_sha256"]
                ):
                    raise ValueError("stored job input integrity mismatch")
                if config["model"] != PAPER_MODEL:
                    raise ValueError("stored job model is unsupported")
                spec = StrategySpec.from_dict(config["strategy"])
                with tempfile.TemporaryDirectory(prefix="neural-paper-") as directory:
                    path = Path(directory) / "recording.jsonl"
                    path.write_bytes(row["recording"])
                    result = simulate_recording(
                        spec,
                        path,
                        initial_cash=config["initial_cash"],
                        fee_per_contract=config["fee_per_contract"],
                        max_order_age_seconds=config["max_order_age_seconds"],
                        max_events=config["max_events"],
                    )
                db.execute(
                    "UPDATE jobs SET status='completed',result=? WHERE id=?",
                    (_canonical(result), row["id"]),
                )
            except (ValueError, OverflowError) as exc:
                db.execute(
                    "UPDATE jobs SET status='failed',error=? WHERE id=?", (str(exc), row["id"])
                )
            finished = db.execute(
                "SELECT id,status,result,error FROM jobs WHERE id=?", (row["id"],)
            ).fetchone()
            return self._view(finished)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    submit = commands.add_parser("submit")
    submit.add_argument("--strategy", type=Path, required=True)
    submit.add_argument("--recording", type=Path, required=True)
    submit.add_argument("--cash", required=True)
    submit.add_argument("--fee-per-contract", required=True)
    commands.add_parser("run-next")
    commands.add_parser("inspect").add_argument("id")
    args = parser.parse_args()
    try:
        jobs = PaperJobs(args.database)
        if args.command == "submit":
            identity = jobs.submit(
                StrategySpec.from_json(args.strategy.read_text()),
                args.recording,
                initial_cash=args.cash,
                fee_per_contract=args.fee_per_contract,
            )
            result = jobs.inspect(identity)
        elif args.command == "inspect":
            result = jobs.inspect(args.id)
        else:
            result = jobs.run_next()
    except (ValueError, OSError, sqlite3.Error) as exc:
        parser.error(str(exc))
    print(_canonical(result))


if __name__ == "__main__":
    main()
