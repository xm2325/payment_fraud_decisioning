from __future__ import annotations
import numpy as np
import pandas as pd


def percentile_rank(x):
    return pd.Series(np.asarray(x)).rank(method="average", pct=True).to_numpy()


def _select_two_lane(model_priority, anomaly_priority, k, exploration_share=0.20):
    """Reserve a fixed capacity share for label-free anomaly exploration.

    The allocation is governance-defined, not tuned on unseen-fraud labels.
    """
    k_explore = max(1, int(round(k * exploration_share))) if k > 1 else 0
    k_exploit = max(0, k - k_explore)
    selected = np.zeros(len(model_priority), dtype=bool)
    if k_exploit:
        exploit_idx = np.argsort(-model_priority)[:k_exploit]
        selected[exploit_idx] = True
    remaining = np.where(~selected)[0]
    if k_explore and len(remaining):
        explore_order = remaining[np.argsort(-anomaly_priority[remaining])[:k_explore]]
        selected[explore_order] = True
    return selected


def queue_metrics(df, model_score, anomaly_score, capacities=(25, 50, 100, 200)):
    """Evaluate fixed analyst budgets per 10k transactions.

    `two_lane_80_20` reserves 20% of review capacity for label-free anomaly
    discovery, avoiding a hidden optimisation on future novel-fraud labels.
    """
    z = df[["is_fraud", "amount", "fraud_type"]].reset_index(drop=True).copy()
    mr = percentile_rank(model_score)
    ar = percentile_rank(anomaly_score)
    amount_rank = percentile_rank(np.log1p(z["amount"].to_numpy(float)))
    hybrid = np.maximum(mr, ar) + 1e-4 * amount_rank

    y = z["is_fraud"].to_numpy(int)
    amt = z["amount"].to_numpy(float)
    novel = z["fraud_type"].eq("novel_shared_device_microburst").to_numpy()
    total_fraud_value = amt[y == 1].sum()
    rows = []
    for per10k in capacities:
        k = max(1, int(np.ceil(len(z) * per10k / 10_000)))
        selections = {}
        for method, priority in [("model", mr), ("hybrid", hybrid)]:
            selected = np.zeros(len(z), dtype=bool)
            selected[np.argsort(-priority)[:k]] = True
            selections[method] = selected
        selections["two_lane_80_20"] = _select_two_lane(mr, ar, k, exploration_share=0.20)

        for method, selected in selections.items():
            rows.append({
                "queue": method,
                "capacity_per_10k": per10k,
                "review_n": int(selected.sum()),
                "precision": float(y[selected].mean()) if selected.any() else np.nan,
                "fraud_recall": float(selected[y == 1].mean()) if (y == 1).any() else np.nan,
                "fraud_value_recall": float(amt[(y == 1) & selected].sum() / total_fraud_value) if total_fraud_value else np.nan,
                "novel_fraud_recall": float(selected[novel].mean()) if novel.any() else np.nan,
            })
    return pd.DataFrame(rows)


def investigation_queue(df, model_score, anomaly_score, n=100):
    """Create an analyst-facing queue with simple, auditable reason codes."""
    z = df.reset_index(drop=True).copy()
    mr = percentile_rank(model_score)
    ar = percentile_rank(anomaly_score)
    priority = np.maximum(mr, ar)
    top = np.argsort(-priority)[:min(n, len(z))]
    out = z.loc[top, [
        "transaction_id", "timestamp", "sender", "recipient", "type", "amount",
        "sender_tx_1h", "sender_tx_24h", "recipient_fanin_24h",
        "device_activity_24h", "device_unique_senders_24h", "is_fraud", "fraud_type"
    ]].copy()
    out["model_score"] = np.asarray(model_score)[top]
    out["anomaly_score"] = np.asarray(anomaly_score)[top]
    out["priority_percentile"] = priority[top]

    def reason(row):
        reasons = []
        if row.model_score >= np.quantile(np.asarray(model_score), 0.99):
            reasons.append("HIGH_MODEL_RISK")
        if row.anomaly_score >= np.quantile(np.asarray(anomaly_score), 0.99):
            reasons.append("BEHAVIOURAL_TAIL")
        if row.device_unique_senders_24h >= 3:
            reasons.append("SHARED_DEVICE")
        if row.sender_tx_1h >= 3:
            reasons.append("RAPID_SENDER_VELOCITY")
        if row.recipient_fanin_24h >= 5:
            reasons.append("RECIPIENT_CONCENTRATION")
        return "|".join(reasons) if reasons else "RISK_RANK"

    out["reason_codes"] = out.apply(reason, axis=1)
    return out.sort_values(["priority_percentile", "amount"], ascending=[False, False])


def exploration_sensitivity(df, model_score, anomaly_score, capacities=(100, 200), shares=(0.0, 0.1, 0.2, 0.3, 0.4)):
    """Sensitivity analysis for governance-set exploration capacity."""
    z = df[["is_fraud", "amount", "fraud_type"]].reset_index(drop=True).copy()
    mr = percentile_rank(model_score)
    ar = percentile_rank(anomaly_score)
    y = z["is_fraud"].to_numpy(int)
    amt = z["amount"].to_numpy(float)
    novel = z["fraud_type"].eq("novel_shared_device_microburst").to_numpy()
    total_fraud_value = amt[y == 1].sum()
    rows = []
    for per10k in capacities:
        k = max(1, int(np.ceil(len(z) * per10k / 10_000)))
        for share in shares:
            if share <= 0:
                selected = np.zeros(len(z), dtype=bool)
                selected[np.argsort(-mr)[:k]] = True
            else:
                selected = _select_two_lane(mr, ar, k, exploration_share=float(share))
            rows.append({
                "capacity_per_10k": per10k,
                "exploration_share": float(share),
                "fraud_value_recall": float(amt[(y == 1) & selected].sum() / total_fraud_value),
                "novel_fraud_recall": float(selected[novel].mean()) if novel.any() else np.nan,
                "precision": float(y[selected].mean()) if selected.any() else np.nan,
            })
    return pd.DataFrame(rows)
