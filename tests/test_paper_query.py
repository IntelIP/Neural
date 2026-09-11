"""Read-only consumer behavior over current, historical and damaged journals."""

import hashlib
import json
import sqlite3
from dataclasses import replace
from decimal import localcontext

import pytest

import neural.paper_worker as worker
from neural.paper_query import PaperJournal
from tests.test_recordings import inputs


def enqueue(jobs, *, venue="polymarket-us", entry="0.45", quantity=None, **assumptions):
    spec, path = inputs(venue)
    identity = jobs.submit(
        replace(spec, entry_price=entry, quantity=quantity or spec.quantity),
        path,
        **{"initial_cash": "10", "fee_per_contract": "0.01", **assumptions},
    )
    return identity, path


def test_reopen_compare_and_source_bytes_across_synthetic_venues(tmp_path):
    database = tmp_path / "journal.sqlite3"
    jobs = worker.PaperJobs(database)
    identities = []
    for venue in ("kalshi", "polymarket-us"):
        for entry in ("0.45", "0.46"):
            identity, source = enqueue(jobs, venue=venue, entry=entry)
            identities.append(identity)
            assert jobs.run_next()["status"] == "completed"
    before = database.read_bytes()
    journal = PaperJournal(database)
    page = journal.history(limit=2)
    assert [job["id"] for job in page["jobs"]] == list(reversed(identities[2:]))
    assert page["has_more"] is True
    assert "result" not in page["jobs"][0]
    assert journal.history(offset=2, limit=2)["has_more"] is False
    assert journal.history(offset=4) == {"jobs": [], "has_more": False}
    first, second = identities[2:]
    saved = journal.inspect(first)
    assert saved["config"]["strategy"]["venue"] == "polymarket_us"
    assert saved["result"] == jobs.inspect(first)["result"]
    assert journal.recording(first) == source.read_bytes()
    with localcontext() as context:
        context.prec = 2
        compared = journal.compare(first, second)
    assert [job["id"] for job in compared] == [first, second]
    for job in compared:
        assert job["result"]["cash"] == "10.52"
        assert job["result"]["realized_pnl"] == "0.52"
        assert job["summary"]["total_fees"] == "0.04"
        assert job["summary"]["fill_count"] == 2
        assert job["summary"]["rejection_count"] == 0
        assert job["result"] == jobs.inspect(job["id"])["result"]
        report = dict(job["result"])
        result_id = report.pop("result_id")
        encoded = json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        assert hashlib.sha256(encoded.encode()).hexdigest() == result_id
    assert "total_fees" not in journal.inspect(first)["result"]
    with pytest.raises(ValueError, match="same recording"):
        journal.compare(identities[0], first)
    terms = journal.compare_markets(identities[0], first)
    assert terms["comparison"]["status"] == "compatible"  # Supplied synthetic rules only.
    assert terms["ids"] == [identities[0], first]
    assert database.read_bytes() == before


def test_reads_do_not_acquire_writer_lock_or_initialize_journal(tmp_path):
    database = tmp_path / "read only?#.sqlite3"
    jobs = worker.PaperJobs(database)
    identity, _ = enqueue(jobs)
    journal = PaperJournal(database)
    before = database.read_bytes()
    with jobs._transaction():
        assert journal.inspect(identity)["status"] == "queued"
        assert journal.history()["jobs"][0]["id"] == identity
        assert journal.recording(identity)
    assert database.read_bytes() == before
    missing = tmp_path / "missing" / "never-created.sqlite3"
    with pytest.raises(FileNotFoundError, match="not found"):
        PaperJournal(missing).history()
    assert not missing.parent.exists()


def test_historical_models_stay_opaque_and_read_only(tmp_path, monkeypatch):
    database = tmp_path / "old.sqlite3"
    jobs = worker.PaperJobs(database)
    monkeypatch.setattr(worker, "PAPER_MODEL", "neural-paper/1")
    first, source = enqueue(jobs)
    second, _ = enqueue(jobs, entry="0.46")
    historical = {"model": "neural-paper/1", "cash": "7.123456789012345678"}
    with jobs._transaction() as db:
        db.execute("UPDATE jobs SET status='completed',result=?", (json.dumps(historical),))
    before = database.read_bytes()
    journal = PaperJournal(database)
    assert journal.inspect(first)["result"] == historical
    assert journal.recording(first) == source.read_bytes()
    with pytest.raises(ValueError, match="trace unavailable"):
        journal.compare(first, second)
    with pytest.raises(ValueError, match="terms unavailable"):
        journal.compare_markets(first, second)
    assert database.read_bytes() == before


def test_derived_fees_preserve_product_scale(tmp_path):
    database = tmp_path / "tiny-fees.sqlite3"
    jobs = worker.PaperJobs(database)
    spec, recording = inputs()
    ids = []
    for entry in ("0.45", "0.46"):
        ids.append(
            jobs.submit(
                replace(spec, entry_price=entry, quantity="0.01"),
                recording,
                initial_cash="10",
                fee_per_contract="0.000000000000000001",
            )
        )
        assert jobs.run_next()["status"] == "completed"
    before = database.read_bytes()
    for job in PaperJournal(jobs.database).compare(*ids):
        assert job["summary"]["total_fees"] == "0.00000000000000000002"
        assert job["summary"]["fill_count"] == 2
    assert database.read_bytes() == before


@pytest.mark.parametrize("fee", ["NaN", "Infinity", "-0.01", 0.01, "0." + "0" * 36 + "1"])
def test_invalid_saved_fees_reject_comparison(tmp_path, fee):
    jobs = worker.PaperJobs(tmp_path / "invalid-fees.sqlite3")
    first, _ = enqueue(jobs)
    second, _ = enqueue(jobs, entry="0.46")
    jobs.run_next()
    jobs.run_next()
    result = jobs.inspect(first)["result"]
    result["trace"][0]["fees"] = fee
    with sqlite3.connect(jobs.database) as db:
        db.execute("UPDATE jobs SET result=? WHERE id=?", (json.dumps(result), first))
    with pytest.raises(ValueError, match="saved trace fees"):
        PaperJournal(jobs.database).compare(first, second)


def test_exclusive_lock_stays_distinguishable_from_invalid_data(tmp_path, monkeypatch):
    database = tmp_path / "busy.sqlite3"
    worker.PaperJobs(database)
    before = database.read_bytes()
    connect = sqlite3.connect
    monkeypatch.setattr(
        sqlite3, "connect", lambda *args, **kwargs: connect(*args, **{**kwargs, "timeout": 0})
    )
    with connect(database) as writer:
        writer.execute("BEGIN EXCLUSIVE")
        with pytest.raises(sqlite3.OperationalError, match="database is locked"):
            PaperJournal(database).history()
    assert database.read_bytes() == before


@pytest.mark.parametrize(
    "change",
    [{"initial_cash": "11"}, {"fee_per_contract": "0.02"}, {"max_events": 1000}, {"quantity": "1"}],
)
def test_incompatible_assumptions_and_unfinished_jobs_reject(tmp_path, change):
    jobs = worker.PaperJobs(tmp_path / "jobs.sqlite3")
    first, _ = enqueue(jobs)
    second, _ = enqueue(jobs, **change)
    journal = PaperJournal(jobs.database)
    with pytest.raises(ValueError, match="completed"):
        journal.compare(first, second)
    jobs.run_next()
    jobs.run_next()
    with pytest.raises(ValueError, match="same recording"):
        journal.compare(first, second)
    with pytest.raises(ValueError, match="distinct"):
        journal.compare(first, first)
    with pytest.raises(ValueError, match="unknown"):
        journal.inspect("missing")


@pytest.mark.parametrize("offset,limit", [(-1, 25), (True, 25), (100001, 25), (0, 0), (0, 101)])
def test_pagination_is_bounded_before_opening_storage(tmp_path, offset, limit):
    database = tmp_path / "absent.sqlite3"
    with pytest.raises(ValueError):
        PaperJournal(database).history(offset=offset, limit=limit)
    assert not database.exists()


@pytest.mark.parametrize("damage", ["config", "recording", "result", "version", "not_sqlite"])
def test_damaged_journals_fail_without_repairing_them(tmp_path, damage):
    database = tmp_path / "jobs.sqlite3"
    jobs = worker.PaperJobs(database)
    identity, _ = enqueue(jobs)
    jobs.run_next()
    if damage == "not_sqlite":
        database.write_bytes(b"not a SQLite journal")
    else:
        with sqlite3.connect(database) as db:
            if damage == "config":
                db.execute("UPDATE jobs SET config=?", ('{"model":"changed"}',))
            elif damage == "recording":
                db.execute("UPDATE jobs SET recording=?", (b"changed",))
            elif damage == "result":
                db.execute("UPDATE jobs SET result=?", ("[]",))
            else:
                db.execute("PRAGMA user_version=99")
    before = database.read_bytes()
    journal = PaperJournal(database)
    with pytest.raises(ValueError):
        if damage == "recording":
            journal.recording(identity)
        else:
            journal.inspect(identity)
    assert database.read_bytes() == before
