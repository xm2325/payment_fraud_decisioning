from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss


def psi(expected, actual, bins=10):
    expected = np.asarray(expected); actual = np.asarray(actual)
    cuts = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(cuts) < 3:
        return 0.0
    cuts[0] = -np.inf; cuts[-1] = np.inf
    e = np.histogram(expected, bins=cuts)[0] / max(1, len(expected))
    a = np.histogram(actual, bins=cuts)[0] / max(1, len(actual))
    e = np.clip(e, 1e-6, None); a = np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def weekly_monitor(df, prob, baseline_prob, model_alert_threshold, tail_scores=None, tail_threshold=None, baseline_tail_alert_rate=None):
    z = df[["timestamp", "is_fraud", "amount", "fraud_type"]].copy()
    z["prob"] = np.asarray(prob)
    if tail_scores is not None:
        z["tail_score"] = np.asarray(tail_scores)
    z["week"] = z["timestamp"].dt.tz_localize(None).dt.to_period("W").astype(str)
    rows = []
    for week, g in z.groupby("week", sort=True):
        novel = g["fraud_type"].eq("novel_shared_device_microburst")
        row = {
            "week": week,
            "n": len(g),
            "fraud_rate": g["is_fraud"].mean(),
            "fraud_value": (g["amount"] * g["is_fraud"]).sum(),
            "model_alert_rate": (g["prob"] >= model_alert_threshold).mean(),
            "mean_model_score": g["prob"].mean(),
            "brier": brier_score_loss(g["is_fraud"], g["prob"]),
            "score_psi_vs_validation": psi(baseline_prob, g["prob"]),
            "novel_fraud_rate": novel.mean(),
        }
        if tail_scores is not None and tail_threshold is not None:
            row["tail_alert_rate"] = (g["tail_score"] > tail_threshold).mean()
        rows.append(row)
    out = pd.DataFrame(rows)
    if "tail_alert_rate" in out and baseline_tail_alert_rate is not None:
        tail_alarm = out["tail_alert_rate"] >= max(0.005, 1.5 * baseline_tail_alert_rate)
    else:
        tail_alarm = pd.Series(np.zeros(len(out), dtype=bool))
    out["status"] = np.select(
        [tail_alarm | (out["score_psi_vs_validation"] >= 0.25), out["score_psi_vs_validation"] >= 0.10],
        ["INVESTIGATE", "WATCH"], default="NORMAL"
    )
    return out
