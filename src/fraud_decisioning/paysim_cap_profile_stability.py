from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


PROFILE_GRID: tuple[str, ...] = ("case_first", "balanced", "value_first")


def profile_stability_summary(
    rows: pd.DataFrame,
    *,
    profiles: Sequence[str] = PROFILE_GRID,
) -> pd.DataFrame:
    """Summarise which capacity stress points preserve each profile's review path."""
    required = {"profile", "capacity_alerts_per_10k", "exact_review_path"}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    output = []
    for profile in profiles:
        subset = rows.loc[rows.profile == profile].sort_values("capacity_alerts_per_10k")
        if subset.empty:
            raise ValueError(f"missing rows for profile {profile}")
        capacities = subset.capacity_alerts_per_10k.astype(int).tolist()
        preserved = subset.loc[
            subset.exact_review_path.astype(bool), "capacity_alerts_per_10k"
        ].astype(int).tolist()
        divergent = subset.loc[
            ~subset.exact_review_path.astype(bool), "capacity_alerts_per_10k"
        ].astype(int).tolist()
        output.append(
            {
                "profile": profile,
                "tested_capacities": ",".join(str(value) for value in capacities),
                "preserved_capacities": ",".join(str(value) for value in preserved),
                "divergent_capacities": ",".join(str(value) for value in divergent),
                "all_tested_capacities_pathwise_preserved": len(divergent) == 0,
                "first_divergent_capacity": divergent[0] if divergent else None,
            }
        )
    return pd.DataFrame(output)
