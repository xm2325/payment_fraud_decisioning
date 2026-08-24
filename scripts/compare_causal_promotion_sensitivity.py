from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--causal", required=True)
    parser.add_argument("--retrospective", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    causal = pd.read_csv(args.causal)
    retrospective = pd.read_csv(args.retrospective)
    keys = ["cycle", "profile"]
    if len(causal) != 6 or len(retrospective) != 6:
        raise ValueError("Expected six cycle-2/3 profile sensitivity rows in each input")

    retrospective = retrospective.rename(
        columns={
            column: f"retrospective_{column}"
            for column in retrospective.columns
            if column not in keys
        }
    )
    causal = causal.rename(
        columns={
            column: f"causal_{column}"
            for column in causal.columns
            if column not in keys
        }
    )
    merged = retrospective.merge(causal, on=keys, how="inner", validate="one_to_one")
    if len(merged) != 6:
        raise ValueError("Sensitivity comparison did not produce six matched rows")
    merged["sensitivity_class_changed"] = (
        merged.retrospective_sensitivity_class != merged.causal_sensitivity_class
    )
    merged["decision_sequence_changed"] = (
        merged.retrospective_decisions != merged.causal_decisions
    )
    merged.to_csv(args.out, index=False)

    summary_path = Path(args.summary)
    with summary_path.open() as handle:
        summary = json.load(handle)
    changed = merged.loc[merged.sensitivity_class_changed | merged.decision_sequence_changed]
    summary["retrospective_vs_causal_sensitivity_changes"] = changed.to_dict("records")

    key = merged.loc[(merged.cycle == 3) & (merged.profile == "value_first")]
    if len(key) != 1:
        raise ValueError("Expected exactly one cycle-3 value-first sensitivity comparison")
    row = key.iloc[0]
    summary["cycle_3_value_first_sensitivity_change"] = {
        "retrospective_class": str(row.retrospective_sensitivity_class),
        "retrospective_decisions": str(row.retrospective_decisions),
        "causal_class": str(row.causal_sensitivity_class),
        "causal_decisions": str(row.causal_decisions),
        "retrospective_min_primary_lower_bound": float(row.retrospective_min_primary_lower_bound),
        "causal_min_primary_lower_bound": float(row.causal_min_primary_lower_bound),
        "retrospective_max_primary_lower_bound": float(row.retrospective_max_primary_lower_bound),
        "causal_max_primary_lower_bound": float(row.causal_max_primary_lower_bound),
    }
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
