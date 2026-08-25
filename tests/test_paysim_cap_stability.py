import numpy as np

from fraud_decisioning.paysim_cap_stability import (
    cohort_queue_comparison,
    pathwise_queue_comparison,
)


def test_pathwise_exact_preservation():
    step = np.array([1, 1, 2, 2, 3, 3])
    selected = np.array([True, False, False, True, True, False])
    review_step = np.array([1, -1, -1, 2, 3, -1])

    result = pathwise_queue_comparison(
        step, selected, review_step, selected.copy(), review_step.copy()
    )

    assert result["exact_final_queue"] is True
    assert result["common_review_step_equal"] is True
    assert result["exact_review_path"] is True
    assert result["divergent_review_steps"] == 0
    assert result["first_divergence_step"] is None
    assert result["min_cumulative_jaccard"] == 1.0


def test_pathwise_detects_same_final_queue_but_different_review_timing():
    step = np.array([1, 1, 2, 2])
    selected = np.array([True, False, True, False])
    reference_review = np.array([1, -1, 2, -1])
    candidate_review = np.array([2, -1, 2, -1])

    result = pathwise_queue_comparison(
        step, selected, reference_review, selected, candidate_review
    )

    assert result["exact_final_queue"] is True
    assert result["common_review_step_equal"] is False
    assert result["exact_review_path"] is False
    assert result["divergent_review_steps"] == 2
    assert result["first_divergence_step"] == 1
    assert result["min_cumulative_jaccard"] < 1.0


def test_pathwise_detects_final_queue_replacement():
    step = np.array([1, 1, 2, 2])
    reference_selected = np.array([True, False, True, False])
    candidate_selected = np.array([True, False, False, True])
    reference_review = np.array([1, -1, 2, -1])
    candidate_review = np.array([1, -1, -1, 2])

    result = pathwise_queue_comparison(
        step,
        reference_selected,
        reference_review,
        candidate_selected,
        candidate_review,
    )

    assert result["exact_final_queue"] is False
    assert result["exact_review_path"] is False
    assert result["reference_only"] == 1
    assert result["candidate_only"] == 1
    assert result["final_overlap"] == 1


def test_cohort_comparison_is_arrival_scoped():
    step = np.array([1, 1, 2, 2, 3, 3])
    reference = np.array([True, False, True, False, True, False])
    candidate = np.array([True, False, False, True, True, False])

    first = cohort_queue_comparison(
        step, reference, candidate, step_min=1, step_max=1
    )
    second = cohort_queue_comparison(
        step, reference, candidate, step_min=2, step_max=2
    )

    assert first["exact_cohort_queue"] is True
    assert second["exact_cohort_queue"] is False
    assert second["reference_only"] == 1
    assert second["candidate_only"] == 1
