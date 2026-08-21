from __future__ import annotations

import numpy as np
import pandas as pd


def _serve_lane(backlog: float, capacity: float):
    served = min(backlog, capacity)
    return backlog - served, served


def queue_sla_stress(df, model_prob, anomaly_score, review_threshold: float,
                     block_threshold: float, anomaly_threshold: float,
                     analyst_capacities_per_hour=(4.0, 6.0, 8.0),
                     volume_multipliers=(0.5, 1.0, 1.5, 2.0, 4.0),
                     exploration_share=0.20):
    """Stress-test two-lane analyst queues under traffic growth.

    Review-band transactions form the exploitation lane. Label-free anomaly
    alerts outside the review/block set form the exploration lane. Traffic
    multipliers scale arrival counts while keeping score mix fixed. Analysts
    have fixed hourly capacity with a governance reservation for exploration;
    unused capacity can spill to the other lane.
    """
    z = df[["timestamp"]].copy().reset_index(drop=True)
    p = np.asarray(model_prob, dtype=float)
    a = np.asarray(anomaly_score, dtype=float)
    block = p >= float(block_threshold)
    exploit = (p >= float(review_threshold)) & ~block
    explore = (a > float(anomaly_threshold)) & ~block & ~exploit
    z["hour"] = pd.to_datetime(z["timestamp"]).dt.floor("h")
    z["exploit"] = exploit.astype(int)
    z["explore"] = explore.astype(int)
    hourly = z.groupby("hour", as_index=False)[["exploit", "explore"]].sum()
    if hourly.empty:
        return pd.DataFrame()
    hours = pd.date_range(hourly.hour.min(), hourly.hour.max(), freq="h")
    hourly = hourly.set_index("hour").reindex(hours, fill_value=0).rename_axis("hour").reset_index()
    observed_arrival_rate = float((hourly.exploit + hourly.explore).mean())

    rows = []
    for capacity in analyst_capacities_per_hour:
        total_cap = float(capacity)
        explore_cap = total_cap * float(exploration_share)
        exploit_cap = total_cap - explore_cap
        for mult in volume_multipliers:
            b_exploit = 0.0
            b_explore = 0.0
            max_backlog = 0.0
            backlog_hours = 0
            served_total = 0.0
            arrivals_total = 0.0
            wait_proxy = []
            for _, r in hourly.iterrows():
                arr_e = float(r.exploit) * float(mult)
                arr_x = float(r.explore) * float(mult)
                b_exploit += arr_e
                b_explore += arr_x
                arrivals_total += arr_e + arr_x

                b_exploit, s_e = _serve_lane(b_exploit, exploit_cap)
                b_explore, s_x = _serve_lane(b_explore, explore_cap)
                spare = total_cap - s_e - s_x
                if spare > 0 and (b_exploit > 0 or b_explore > 0):
                    if b_exploit >= b_explore:
                        b_exploit, s1 = _serve_lane(b_exploit, spare)
                        spare -= s1; s_e += s1
                        if spare > 0:
                            b_explore, s2 = _serve_lane(b_explore, spare); s_x += s2
                    else:
                        b_explore, s1 = _serve_lane(b_explore, spare)
                        spare -= s1; s_x += s1
                        if spare > 0:
                            b_exploit, s2 = _serve_lane(b_exploit, spare); s_e += s2
                served_total += s_e + s_x
                backlog = b_exploit + b_explore
                max_backlog = max(max_backlog, backlog)
                backlog_hours += int(backlog > 1e-9)
                wait_proxy.append(backlog / max(total_cap, 1e-9))

            final_backlog = b_exploit + b_explore
            service_capacity = total_cap * len(hourly)
            rows.append({
                "traffic_multiplier": float(mult),
                "analyst_capacity_per_hour": total_cap,
                "exploration_share": float(exploration_share),
                "hours": int(len(hourly)),
                "observed_candidate_arrival_rate_per_hour": observed_arrival_rate,
                "scaled_arrival_rate_per_hour": observed_arrival_rate * float(mult),
                "arrivals": arrivals_total,
                "served": served_total,
                "capacity_utilisation": arrivals_total / max(service_capacity, 1e-9),
                "max_backlog_cases": max_backlog,
                "final_backlog_cases": final_backlog,
                "hours_with_backlog_rate": backlog_hours / len(hourly),
                "max_wait_proxy_hours": float(max(wait_proxy) if wait_proxy else 0.0),
                "stable_by_end": bool(final_backlog < 1e-9),
                "meets_4h_sla_proxy": bool(
                    (arrivals_total / max(service_capacity, 1e-9) < 1.0)
                    and (max(wait_proxy) if wait_proxy else 0.0) <= 4.0
                ),
            })
    return pd.DataFrame(rows)
