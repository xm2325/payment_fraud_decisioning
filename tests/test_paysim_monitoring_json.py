import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_paysim_monitoring.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_paysim_monitoring", RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_json_safe_handles_numpy_and_non_finite_values():
    json_safe = _load_runner()._json_safe
    value = {
        "count": np.int64(7),
        "rate": np.float64(0.25),
        "missing": np.float64(np.nan),
        "nested": [np.bool_(True), np.float64(np.inf)],
    }
    safe = json_safe(value)
    assert safe == {"count": 7, "rate": 0.25, "missing": None, "nested": [True, None]}
    json.dumps(safe, allow_nan=False)


def test_monitoring_cli_imports_without_external_pythonpath():
    proc = subprocess.run(
        [sys.executable, str(RUNNER), "--help"],
        cwd=ROOT,
        env={},
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--parquet-glob" in proc.stdout
