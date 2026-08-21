import importlib.util
import json
from pathlib import Path

import numpy as np


def _load_runner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_paysim_monitoring.py"
    spec = importlib.util.spec_from_file_location("run_paysim_monitoring", path)
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
