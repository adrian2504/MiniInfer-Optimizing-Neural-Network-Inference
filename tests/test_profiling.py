import json
import subprocess
import sys

import pytest


@pytest.mark.parametrize("workload", ["generation", "rmsnorm"])
def test_cpu_profile_artifacts(tmp_path, workload):
    directory = tmp_path / workload
    command = [sys.executable, "-m", "miniinfer.profiling", "--workload", workload,
               "--smoke", "--steps", "1", "--warmup", "1", "--output-dir", str(directory)]
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    report = json.loads((directory / "profile.json").read_text())
    assert report["events"]
    assert any(row["name"] == f"miniinfer::{workload}" for row in report["events"])
    trace = json.loads((directory / "trace.json").read_text())
    assert trace["traceEvents"]
    assert (directory / "observations.md").exists()
    original = (directory / "profile.json").read_bytes()
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert (directory / "profile.json").read_bytes() == original
