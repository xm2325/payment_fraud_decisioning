import pandas as pd

from fraud_decisioning.paysim_cap_profile_stability import profile_stability_summary


def test_profile_summary_all_preserved():
    rows = pd.DataFrame(
        [
            {"profile": profile, "capacity_alerts_per_10k": capacity, "exact_review_path": True}
            for profile in ("case_first", "balanced", "value_first")
            for capacity in (10, 25, 50, 100)
        ]
    )
    summary = profile_stability_summary(rows)
    assert summary.all_tested_capacities_pathwise_preserved.all()
    assert summary.first_divergent_capacity.isna().all()


def test_profile_summary_localises_divergence():
    rows = pd.DataFrame(
        [
            {"profile": "case_first", "capacity_alerts_per_10k": 10, "exact_review_path": True},
            {"profile": "case_first", "capacity_alerts_per_10k": 25, "exact_review_path": True},
            {"profile": "balanced", "capacity_alerts_per_10k": 10, "exact_review_path": True},
            {"profile": "balanced", "capacity_alerts_per_10k": 25, "exact_review_path": True},
            {"profile": "value_first", "capacity_alerts_per_10k": 10, "exact_review_path": True},
            {"profile": "value_first", "capacity_alerts_per_10k": 25, "exact_review_path": False},
        ]
    )
    summary = profile_stability_summary(rows)
    value = summary.loc[summary.profile == "value_first"].iloc[0]
    assert not bool(value.all_tested_capacities_pathwise_preserved)
    assert int(value.first_divergent_capacity) == 25
    assert value.divergent_capacities == "25"
