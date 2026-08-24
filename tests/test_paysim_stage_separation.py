import numpy as np

from fraud_decisioning.paysim_stage_separation import (
    probability_stage_metrics,
    split_summary_frame,
    split_validation_stages,
    stage_masks,
)


def test_validation_stage_split_is_temporal_disjoint_and_exhaustive():
    step = np.repeat(np.arange(446, 595), 2)
    split = split_validation_stages(step, calibration_fraction=0.5)
    calibration, policy = stage_masks(step, split)

    assert split.policy_cut == 520
    assert split.calibration_step_min == 446
    assert split.calibration_step_max == 519
    assert split.policy_step_min == 520
    assert split.policy_step_max == 594
    assert split.calibration_n_steps == 74
    assert split.policy_n_steps == 75
    assert not np.any(calibration & policy)
    assert np.all(calibration | policy)
    assert step[calibration].max() < step[policy].min()


def test_validation_stage_split_does_not_use_labels():
    step = np.repeat(np.arange(10, 20), 3)
    split_a = split_validation_stages(step)
    # Changing labels cannot change a function that receives only time steps.
    split_b = split_validation_stages(step.copy())
    assert split_a == split_b


def test_split_summary_reports_stage_ranges_and_counts():
    step = np.repeat(np.arange(1, 9), 2)
    y = np.tile(np.array([0, 1]), 8)
    split = split_validation_stages(step)
    frame = split_summary_frame(step, y, split)

    assert frame.stage.tolist() == ["calibration", "policy_selection"]
    assert frame.n.sum() == len(step)
    assert frame.fraud_n.sum() == int(y.sum())
    assert frame.loc[0, "step_max"] < frame.loc[1, "step_min"]


def test_probability_stage_metrics_are_bounded_and_labelled():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.05, 0.2, 0.7, 0.9])
    row = probability_stage_metrics("policy_selection", y, p)

    assert row["stage"] == "policy_selection"
    assert row["n"] == 4
    assert row["fraud_n"] == 2
    assert 0 <= row["brier"] <= 1
    assert 0 <= row["pr_auc"] <= 1


def test_invalid_stage_fraction_is_rejected():
    step = np.arange(10)
    for fraction in (0.0, 1.0, -0.1, 1.1):
        try:
            split_validation_stages(step, calibration_fraction=fraction)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid calibration fraction should fail")
