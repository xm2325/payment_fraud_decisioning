from __future__ import annotations
import numpy as np
import pandas as pd
from .evaluation import policy_grid

SCENARIOS = {
    "reference": dict(block_efficacy=0.95, review_efficacy=0.65, review_case_cost=0.75, false_block_cost_unit=5.0, false_review_cost_unit=1.5),
    "high_customer_friction": dict(block_efficacy=0.95, review_efficacy=0.65, review_case_cost=1.0, false_block_cost_unit=20.0, false_review_cost_unit=5.0),
    "conservative_intervention": dict(block_efficacy=0.85, review_efficacy=0.45, review_case_cost=1.5, false_block_cost_unit=10.0, false_review_cost_unit=3.0),
    "strong_intervention_low_friction": dict(block_efficacy=0.98, review_efficacy=0.80, review_case_cost=0.5, false_block_cost_unit=2.0, false_review_cost_unit=0.5),
}


def policy_sensitivity(y_val, p_val, amount_val, y_test, p_test, amount_test, scenarios=None):
    """Select policy on validation for each business-cost scenario and audit on test.

    Test-optimal cost is used only as a retrospective comparator for regret; it
    is never used to select the deployed threshold pair.
    """
    scenarios = SCENARIOS if scenarios is None else scenarios
    rows = []
    for name, kw in scenarios.items():
        gv = policy_grid(y_val, p_val, amount_val, **kw)
        chosen = gv.loc[gv.total_policy_cost.idxmin()]
        gt = policy_grid(y_test, p_test, amount_test, **kw)
        match = gt[np.isclose(gt.review_threshold, chosen.review_threshold) & np.isclose(gt.block_threshold, chosen.block_threshold)]
        if len(match) != 1:
            raise RuntimeError(f"Could not map validation policy to test grid for {name}")
        deployed = match.iloc[0]
        oracle = gt.loc[gt.total_policy_cost.idxmin()]
        regret = float((deployed.total_policy_cost - oracle.total_policy_cost) / max(oracle.total_policy_cost, 1e-9))
        rows.append({
            "scenario": name,
            **kw,
            "selected_review_threshold": float(chosen.review_threshold),
            "selected_block_threshold": float(chosen.block_threshold),
            "validation_total_cost": float(chosen.total_policy_cost),
            "test_fraud_value_prevented_rate": float(deployed.fraud_value_prevented_rate),
            "test_legit_friction_rate": float(deployed.legit_friction_rate),
            "test_total_cost": float(deployed.total_policy_cost),
            "test_oracle_total_cost": float(oracle.total_policy_cost),
            "test_policy_regret": regret,
        })
    return pd.DataFrame(rows)
