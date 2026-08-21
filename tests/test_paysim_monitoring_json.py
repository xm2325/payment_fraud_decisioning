import json

import numpy as np

from scripts.run_paysim_monitoring import _json_safe


def test_json_safe_handles_numpy_and_non_finite_values():
    value = {
        "count": np.int64(7),
        "rate": np.float64(0.25),
        "missing": np.float64(np.nan),
        "nested": [np.bool_(True), np.float64(np.inf)],
    }
    safe = _json_safe(value)
    assert safe == {"count": 7, "rate": 0.25, "missing": None, "nested": [True, None]}
    json.dumps(safe, allow_nan=False)
