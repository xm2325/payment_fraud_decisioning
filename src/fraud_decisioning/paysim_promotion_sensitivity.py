from __future__ import annotations

from collections.abc import Sequence


DEFAULT_BLOCK_STEPS = (1, 3, 5, 10)


def classify_dependence_sensitivity(decisions: Sequence[str]) -> str:
    """Classify whether a promotion conclusion is stable across block lengths.

    This is a robustness label only. It does not override the pre-declared v1.9
    promotion decision made with the 5-step block length.
    """
    values = tuple(str(value) for value in decisions)
    if not values:
        raise ValueError("decisions must not be empty")
    allowed = {"PROMOTE", "KEEP_INCUMBENT"}
    unknown = set(values).difference(allowed)
    if unknown:
        raise ValueError(f"Unsupported promotion decisions: {sorted(unknown)}")
    if all(value == "KEEP_INCUMBENT" for value in values):
        return "ROBUST_KEEP_INCUMBENT"
    if all(value == "PROMOTE" for value in values):
        return "ROBUST_PROMOTE"
    return "DEPENDENCE_SENSITIVE"


def sensitivity_summary(rows: Sequence[dict]) -> list[dict]:
    """Summarise block-length decision stability by cycle/profile."""
    grouped: dict[tuple[int, str], list[dict]] = {}
    for row in rows:
        key = (int(row["cycle"]), str(row["profile"]))
        grouped.setdefault(key, []).append(dict(row))

    output: list[dict] = []
    for (cycle, profile), group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda row: int(row["block_steps"]))
        decisions = [str(row["decision"]) for row in ordered]
        output.append(
            {
                "cycle": cycle,
                "profile": profile,
                "sensitivity_class": classify_dependence_sensitivity(decisions),
                "block_steps": ",".join(str(int(row["block_steps"])) for row in ordered),
                "decisions": ",".join(decisions),
                "min_primary_lower_bound": min(float(row["primary_lower_bound"]) for row in ordered),
                "max_primary_lower_bound": max(float(row["primary_lower_bound"]) for row in ordered),
                "min_fraud_recall_lcb": min(float(row["lcb_fraud_recall"]) for row in ordered),
                "max_fraud_recall_lcb": max(float(row["lcb_fraud_recall"]) for row in ordered),
            }
        )
    return output
