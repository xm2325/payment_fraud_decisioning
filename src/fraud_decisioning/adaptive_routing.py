from __future__ import annotations

import numpy as np
import pandas as pd


def _take_top(indices: np.ndarray, score: np.ndarray, k: int) -> np.ndarray:
    if k <= 0 or len(indices) == 0:
        return np.empty(0, dtype=int)
    k = min(int(k), len(indices))
    order = np.argsort(-score[indices], kind="mergesort")[:k]
    return indices[order]


def adaptive_capacity_routing(
    df: pd.DataFrame,
    model_prob,
    anomaly_score,
    review_threshold: float,
    block_threshold: float,
    anomaly_threshold: float,
    analyst_capacity_per_hour: float = 6.0,
    traffic_multipliers=(1.0, 1.5, 2.0, 4.0),
    exploration_share: float = 0.20,
) -> pd.DataFrame:
    """Backlog-aware admission control for a two-lane fraud review queue.

    The fixed policy first defines *candidate* exploit/explore cases. When the
    candidate stream would exceed analyst capacity, this controller admits a
    capacity-feasible subset each hour, reserving a governance-defined share
    for label-free exploration and spilling unused slots to the other lane.

    Traffic multipliers are represented as an equivalent reduction in the
    number of unique source-stream cases that can be admitted per hour
    (capacity / multiplier). This keeps the score/class mix fixed and makes the
    stress test deterministic; it is a scaling sensitivity, not a staffing
    forecast.
    """
    if analyst_capacity_per_hour <= 0:
        raise ValueError("analyst_capacity_per_hour must be positive")
    if not 0 <= exploration_share <= 1:
        raise ValueError("exploration_share must be in [0, 1]")

    z = df.reset_index(drop=True).copy()
    p = np.asarray(model_prob, dtype=float)
    a = np.asarray(anomaly_score, dtype=float)
    if len(z) != len(p) or len(z) != len(a):
        raise ValueError("df and score arrays must have the same length")

    block = p >= float(block_threshold)
    exploit = (p >= float(review_threshold)) & ~block
    explore = (a > float(anomaly_threshold)) & ~block & ~exploit
    candidate = exploit | explore
    z["hour"] = pd.to_datetime(z["timestamp"]).dt.floor("h")

    y = z["is_fraud"].to_numpy(int)
    amount = z["amount"].to_numpy(float)
    novel = z["fraud_type"].eq("novel_shared_device_microburst").to_numpy()
    legit = y == 0
    total_fraud_value = amount[y == 1].sum()
    candidate_fraud_value = amount[(y == 1) & candidate].sum()
    candidate_novel_n = int((novel & candidate).sum())

    hours = pd.date_range(z.hour.min(), z.hour.max(), freq="h")
    hour_to_idx = {h: np.flatnonzero(z.hour.to_numpy() == h) for h in hours}

    rows = []
    for mult in traffic_multipliers:
        mult = float(mult)
        if mult <= 0:
            raise ValueError("traffic multipliers must be positive")
        effective_capacity = float(analyst_capacity_per_hour) / mult
        token_carry = 0.0
        exploration_token_carry = 0.0
        selected = np.zeros(len(z), dtype=bool)
        hourly_selected = []
        hourly_candidates = []
        exploit_cutoffs = []
        explore_cutoffs = []

        for h in hours:
            idx = hour_to_idx[h]
            e_idx = idx[exploit[idx]]
            x_idx = idx[explore[idx]]
            hourly_candidates.append(len(e_idx) + len(x_idx))

            token_carry += effective_capacity
            slots = int(np.floor(token_carry + 1e-12))
            token_carry -= slots
            if slots <= 0:
                hourly_selected.append(0)
                continue

            # Carry fractional exploration tokens across hours so a 20%
            # reservation remains meaningful even when only 1-2 review slots
            # are available in an hour. This avoids starving discovery during
            # high-traffic periods through integer rounding.
            exploration_token_carry += slots * exploration_share
            x_slots = min(int(np.floor(exploration_token_carry + 1e-12)), slots)
            exploration_token_carry -= x_slots
            e_slots = slots - x_slots

            e_sel = _take_top(e_idx, p, e_slots)
            x_sel = _take_top(x_idx, a, x_slots)
            used = len(e_sel) + len(x_sel)
            spare = slots - used

            # Spill unused capacity to the lane with remaining candidates.
            if spare > 0:
                e_rem = np.setdiff1d(e_idx, e_sel, assume_unique=False)
                x_rem = np.setdiff1d(x_idx, x_sel, assume_unique=False)
                if len(e_rem):
                    add = _take_top(e_rem, p, spare)
                    e_sel = np.concatenate([e_sel, add])
                    spare -= len(add)
                if spare > 0 and len(x_rem):
                    add = _take_top(x_rem, a, spare)
                    x_sel = np.concatenate([x_sel, add])
                    spare -= len(add)

            selected[e_sel] = True
            selected[x_sel] = True
            hourly_selected.append(len(e_sel) + len(x_sel))
            if len(e_sel):
                exploit_cutoffs.append(float(np.min(p[e_sel])))
            if len(x_sel):
                explore_cutoffs.append(float(np.min(a[x_sel])))

        selected_fraud_value = amount[(y == 1) & selected].sum()
        selected_novel_n = int((novel & selected).sum())
        system_covered = selected | block
        system_fraud_value = amount[(y == 1) & system_covered].sum()
        selected_n = int(selected.sum())
        candidate_n = int(candidate.sum())
        scaled_reviews_per_hour = selected_n * mult / max(len(hours), 1)
        rows.append({
            "traffic_multiplier": mult,
            "analyst_capacity_per_hour": float(analyst_capacity_per_hour),
            "exploration_share": float(exploration_share),
            "candidate_n": candidate_n,
            "admitted_n_source_stream": selected_n,
            "candidate_acceptance_rate": selected_n / max(candidate_n, 1),
            "scaled_reviews_per_hour": scaled_reviews_per_hour,
            "capacity_utilisation_after_admission": scaled_reviews_per_hour / float(analyst_capacity_per_hour),
            "analyst_reviewed_fraud_value_share": selected_fraud_value / max(total_fraud_value, 1e-12),
            "candidate_fraud_value_retention": selected_fraud_value / max(candidate_fraud_value, 1e-12),
            "system_fraud_value_coverage_with_blocks": system_fraud_value / max(total_fraud_value, 1e-12),
            "analyst_reviewed_novel_recall": selected_novel_n / max(int(novel.sum()), 1),
            "candidate_novel_retention": selected_novel_n / max(candidate_novel_n, 1),
            "system_novel_recall_with_blocks": float(system_covered[novel].mean()) if novel.any() else np.nan,
            "legitimate_review_rate": float(selected[legit].mean()) if legit.any() else np.nan,
            "mean_hourly_candidates_source_stream": float(np.mean(hourly_candidates)),
            "max_hourly_candidates_source_stream": int(np.max(hourly_candidates)) if hourly_candidates else 0,
            "max_hourly_admitted_source_stream": int(np.max(hourly_selected)) if hourly_selected else 0,
            "median_dynamic_exploit_cutoff": float(np.median(exploit_cutoffs)) if exploit_cutoffs else np.nan,
            "median_dynamic_explore_cutoff": float(np.median(explore_cutoffs)) if explore_cutoffs else np.nan,
        })
    return pd.DataFrame(rows)
