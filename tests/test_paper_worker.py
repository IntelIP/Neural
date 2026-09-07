"""Durable queue workflow, process-death and concurrent worker checks."""

import json
import sqlite3
import subprocess
import sys

import pytest

import neural.paper_worker as worker
from neural.strategy import StrategySpec
from tests.test_paper_recording import recording


def enqueue(tmp_path):
    spec = StrategySpec("kalshi", "KX-EXAMPLE", "yes", "0.45", "0.65", "2", "2", "1")
    path = recording(tmp_path)
    database = tmp_path / "jobs.sqlite3"
    jobs = worker.PaperJobs(database)
    identity = jobs.submit(spec, path, initial_cash="10", fee_per_contract="0.01")
    return jobs, identity, spec, path, database


def test_snapshot_dedup_and_restart(tmp_path, monkeypatch):
    jobs, identity, spec, path, database = enqueue(tmp_path)
    assert jobs.submit(spec, path, initial_cash="10.0", fee_per_contract="0.010") == identity
    path.write_text("original input replaced")
    result = worker.PaperJobs(database).run_next()
    assert result["status"] == "completed"
    assert result["result"]["cash"] == "10.56"

    def forbidden(*args, **kwargs):
        pytest.fail("completed job was executed again")

    monkeypatch.setattr(worker, "simulate_recording", forbidden)
    restarted = worker.PaperJobs(database)
    assert restarted.run_next() is None
    assert restarted.inspect(identity) == result


def test_bad_recording_is_terminal_failure(tmp_path):
    jobs, _, spec, path, database = enqueue(tmp_path)
    path.write_text("bad\n")
    identity = jobs.submit(spec, path, initial_cash="10", fee_per_contract="0.01")
    jobs.run_next()
    result = jobs.run_next()
    assert result["id"] == identity
    assert result["status"] == "failed"
    assert result["error"] and result["result"] is None
    assert worker.PaperJobs(database).inspect(identity) == result
    assert jobs.run_next() is None


def test_process_death_after_simulation_rolls_back(tmp_path):
    jobs, identity, _, _, database = enqueue(tmp_path)
    code = """
import os, sys
import neural.paper_worker as worker
original = worker.simulate_recording
def die(*args, **kwargs):
    original(*args, **kwargs)
    os._exit(73)
worker.simulate_recording = die
worker.PaperJobs(sys.argv[1]).run_next()
"""
    process = subprocess.run([sys.executable, "-c", code, str(database)], timeout=20)
    assert process.returncode == 73
    assert jobs.inspect(identity)["status"] == "queued"
    assert worker.PaperJobs(database).run_next()["result"]["cash"] == "10.56"


def test_concurrent_processes_publish_one_result(tmp_path):
    jobs, identity, _, _, database = enqueue(tmp_path)
    command = [sys.executable, "-m", "neural.paper_worker", "--database", str(database), "run-next"]
    processes = [
        subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for _ in range(2)
    ]
    results = []
    for process in processes:
        stdout, stderr = process.communicate(timeout=20)
        assert process.returncode == 0, stderr
        results.append(json.loads(stdout))
    assert sum(result is None for result in results) == 1
    assert jobs.inspect(identity)["status"] == "completed"


def test_storage_failure_leaves_job_queued(tmp_path, monkeypatch):
    jobs, identity, _, _, _ = enqueue(tmp_path)

    def fail(*args, **kwargs):
        raise OSError("disk unavailable")

    monkeypatch.setattr(worker, "simulate_recording", fail)
    with pytest.raises(OSError):
        jobs.run_next()
    assert jobs.inspect(identity)["status"] == "queued"


def test_input_integrity_and_unknown_job(tmp_path):
    jobs, identity, _, _, database = enqueue(tmp_path)
    with sqlite3.connect(database) as db:
        db.execute("UPDATE jobs SET recording=?", (b"corrupt",))
    assert jobs.run_next()["error"] == "stored job input integrity mismatch"
    with pytest.raises(ValueError, match="unknown"):
        jobs.inspect("missing")


@pytest.mark.parametrize(
    "schema", ["CREATE TABLE important (value TEXT)", "CREATE VIEW important AS SELECT 1"]
)
def test_foreign_database_is_not_repurposed(tmp_path, schema):
    database = tmp_path / "foreign.sqlite3"
    with sqlite3.connect(database) as db:
        db.execute(schema)
        before = db.execute("SELECT type,name,sql FROM sqlite_master").fetchall()
    with pytest.raises(ValueError, match="unsupported"):
        worker.PaperJobs(database)
    with sqlite3.connect(database) as db:
        assert db.execute("SELECT type,name,sql FROM sqlite_master").fetchall() == before
        assert db.execute("PRAGMA application_id").fetchone()[0] == 0
        assert db.execute("PRAGMA user_version").fetchone()[0] == 0


def test_cli_submit_run_inspect(tmp_path):
    _, _, spec, path, _ = enqueue(tmp_path)
    strategy = tmp_path / "strategy.json"
    strategy.write_text(spec.to_json())
    command = [
        sys.executable,
        "-m",
        "neural.paper_worker",
        "--database",
        str(tmp_path / "cli.sqlite3"),
    ]

    def invoke(*args):
        result = subprocess.run(command + list(args), capture_output=True, text=True, timeout=20)
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)

    queued = invoke(
        "submit",
        "--strategy",
        str(strategy),
        "--recording",
        str(path),
        "--cash",
        "10",
        "--fee-per-contract",
        "0.01",
    )
    assert queued["status"] == "queued"
    completed = invoke("run-next")
    assert completed["result"]["cash"] == "10.56"
    assert invoke("inspect", queued["id"]) == completed


def test_submission_limits(tmp_path, monkeypatch):
    jobs, _, spec, path, _ = enqueue(tmp_path)
    with pytest.raises(ValueError):
        jobs.submit(spec, path, initial_cash="10", fee_per_contract="0.01", max_events=True)
    monkeypatch.setattr(worker, "MAX_INPUT_BYTES", 1)
    with pytest.raises(ValueError, match="32 MiB"):
        jobs.submit(spec, path, initial_cash="10", fee_per_contract="0.01")


@pytest.mark.parametrize("database", ["", ":memory:"])
def test_temporary_database_rejected(database):
    with pytest.raises(ValueError, match="database file"):
        worker.PaperJobs(database)
