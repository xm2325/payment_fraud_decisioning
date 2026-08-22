from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss


def nested_policy_cut(train_cut: int, validation_cut: int) -> int:
    """Pre-specify the midpoint separating calibration from policy selection."""
    if validation_cut - train_cut < 2:
        raise ValueError("validation period must contain at least two steps")
    cut = (int(train_cut) + int(validation_cut)) // 2
    if not train_cut < cut < validation_cut:
        raise ValueError("nested policy cut must lie strictly inside validation")
    return cut


def calibration_metrics(
    y: Sequence[int], probability: Sequence[float], *, n_bins: int = 10
) -> dict:
    """Return simple calibration diagnostics without changing the operating policy.

    ECE uses fixed probability bins. These metrics are descriptive only; future
    labels are never used to refit calibration or choose the routing policy.
    """
    y_arr = np.asarray(y, dtype=int)
    p_arr = np.asarray(probability, dtype=float)
    if len(y_arr) != len(p_arr) or len(y_arr) == 0:
        raise ValueError("y and probability must have equal non-zero length")
    if np.any(~np.isfinite(p_arr)) or np.any((p_arr < 0) | (p_arr > 1)):
        raise ValueError("probability must be finite and in [0, 1]")
    if n_bins < 1:
        raise ValueError("n_bins must be positive")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_id = np.minimum(np.digitize(p_arr, edges[1:-1], right=False), n_bins - 1)
    ece = 0.0
    for idx in range(n_bins):
        mask = bin_id == idx
        if not np.any(mask):
            continue
        ece += float(mask.mean()) * abs(float(p_arr[mask].mean()) - float(y_arr[mask].mean()))

    return {
        "n": int(len(y_arr)),
        "fraud_n": int(y_arr.sum()),
        "fraud_rate": float(y_arr.mean()),
        "mean_predicted_risk": float(p_arr.mean()),
        "brier": float(brier_score_loss(y_arr, p_arr)),
        "ece_10bin": float(ece),
        "pr_auc": float(average_precision_score(y_arr, p_arr)),
    }
