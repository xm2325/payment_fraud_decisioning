from __future__ import annotations

from collections.abc import Sequence

import numpy as np


CAPACITY_GRID: tuple[int, ...] = (10, 25, 50, 100)
TARGET_CAP = 5_000


def pathwise_queue_comparison(
    step: Sequence[int],
    reference_selected: Sequence[bool],
    reference_review_step: Sequence[int],
    candidate_selected: Sequence[bool],
    candidate_review_step: Sequence[int],
) -> dict[str, float | int | bool | None]:
    """Compare two realised review queues over the full causal time path.

    This is stricter than comparing aggregate metrics or the final selected set.
    It asks whether the same cases were reviewed at the same PaySim steps and
    reports the first time the realised review path diverged.
    """
    step_arr = np.asarray(step, dtype=int)
    ref_selected = np.asarray(reference_selected, dtype=bool)
    ref_review = np.asarray(reference_review_step, dtype=int)
    cand_selected = np.asarray(candidate_selected, dtype=bool)
    cand_review = np.asarray(candidate_review_step, dtype=int)

    n = len(step_arr)
    if n == 0 or any(len(value) != n for value in (ref_selected, ref_review, cand_selected, cand_review)):
        raise ValueError("pathwise queue inputs must have equal non-zero length")
    if np.any(ref_review[ref_selected] < step_arr[ref_selected]):
        raise ValueError("reference review cannot precede arrival")
    if np.any(cand_review[cand_selected] < step_arr[cand_selected]):
        raise ValueError("candidate review cannot precede arrival")

    ref_n = int(ref_selected.sum())
    cand_n = int(cand_selected.sum())
    overlap = int((ref_selected & cand_selected).sum())
    union = int((ref_selected | cand_selected).sum())
    exact_final_queue = bool(np.array_equal(ref_selected, cand_selected))

    common = ref_selected & cand_selected
    common_review_step_equal = bool(
        np.array_equal(ref_review[common], cand_review[common])
    )
    exact_review_path = bool(
        exact_final_queue
        and np.array_equal(ref_review[ref_selected], cand_review[cand_selected])
    )

    review_steps = sorted(
        set(int(value) for value in ref_review[ref_selected])
        | set(int(value) for value in cand_review[cand_selected])
    )
    divergent_steps: list[int] = []
    min_cumulative_jaccard = 1.0
    for value in review_steps:
        ref_at = ref_selected & (ref_review == value)
        cand_at = cand_selected & (cand_review == value)
        if not np.array_equal(ref_at, cand_at):
            divergent_steps.append(value)

        ref_prefix = ref_selected & (ref_review <= value)
        cand_prefix = cand_selected & (cand_review <= value)
        prefix_overlap = int((ref_prefix & cand_prefix).sum())
        prefix_union = int((ref_prefix | cand_prefix).sum())
        jaccard = float(prefix_overlap / prefix_union) if prefix_union else 1.0
        min_cumulative_jaccard = min(min_cumulative_jaccard, jaccard)

    return {
        "reference_alerts": ref_n,
        "candidate_alerts": cand_n,
        "final_overlap": overlap,
        "final_jaccard": float(overlap / union) if union else 1.0,
        "reference_only": int((ref_selected & ~cand_selected).sum()),
        "candidate_only": int((cand_selected & ~ref_selected).sum()),
        "exact_final_queue": exact_final_queue,
        "common_review_step_equal": common_review_step_equal,
        "exact_review_path": exact_review_path,
        "divergent_review_steps": int(len(divergent_steps)),
        "first_divergence_step": int(divergent_steps[0]) if divergent_steps else None,
        "min_cumulative_jaccard": float(min_cumulative_jaccard),
    }


def cohort_queue_comparison(
    arrival_step: Sequence[int],
    reference_selected: Sequence[bool],
    candidate_selected: Sequence[bool],
    *,
    step_min: int,
    step_max: int,
) -> dict[str, float | int | bool]:
    """Compare final selected sets for one arrival cohort."""
    step_arr = np.asarray(arrival_step, dtype=int)
    ref = np.asarray(reference_selected, dtype=bool)
    cand = np.asarray(candidate_selected, dtype=bool)
    if len(step_arr) == 0 or len(ref) != len(step_arr) or len(cand) != len(step_arr):
        raise ValueError("cohort queue inputs must have equal non-zero length")
    if step_min > step_max:
        raise ValueError("step_min must not exceed step_max")
    cohort = (step_arr >= int(step_min)) & (step_arr <= int(step_max))
    ref = ref & cohort
    cand = cand & cohort
    overlap = int((ref & cand).sum())
    union = int((ref | cand).sum())
    return {
        "reference_alerts": int(ref.sum()),
        "candidate_alerts": int(cand.sum()),
        "overlap": overlap,
        "jaccard": float(overlap / union) if union else 1.0,
        "reference_only": int((ref & ~cand).sum()),
        "candidate_only": int((cand & ~ref).sum()),
        "exact_cohort_queue": bool(np.array_equal(ref, cand)),
    }
