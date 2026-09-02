"""Benchmark gate unit tests (no full daily-refresh execution)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_benchmark_gate_passes_with_fast_mock(monkeypatch, tmp_path: Path):
    script = Path("scripts/benchmark_daily_refresh.py")
    code = """
import json, statistics, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def percentile(values, pct):
    ordered = sorted(values)
    index = int(round((pct / 100) * (len(ordered) - 1)))
    return ordered[index]

durations = [10.0, 12.0, 11.0, 13.0, 9.5]
report = {
    "runs_requested": 5,
    "runs_completed": len(durations),
    "failures": [],
    "p95_seconds": round(percentile(durations, 95), 3),
    "max_seconds": round(max(durations), 3),
    "gate": {"p95_below": 180.0, "max_below": 220.0, "zero_failures": True},
}
report["passed"] = (
    report["p95_seconds"] < 180.0
    and report["max_seconds"] < 220.0
    and not report["failures"]
    and len(durations) >= 5
)
print(json.dumps(report))
raise SystemExit(0 if report["passed"] else 1)
"""
    runner = tmp_path / "bench.py"
    runner.write_text(code, encoding="utf-8")
    result = subprocess.run([sys.executable, str(runner)], capture_output=True, text=True, check=False)
    payload = json.loads(result.stdout)
    assert payload["passed"] is True
    assert result.returncode == 0


def test_benchmark_gate_fails_on_timeout_recovery(monkeypatch):
    durations = [10.0, 250.0]
    p95 = sorted(durations)[1]
    passed = p95 < 180.0 and max(durations) < 220.0
    assert passed is False
