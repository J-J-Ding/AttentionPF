import numpy as np
import pytest

from metrics import compute_metrics
from metrics.interval_metrics import interval_metrics
from metrics.point_metrics import point_metrics


def test_point_metrics_match_hand_calculation():
    point = np.array([0.0, 2.0, 5.0])
    target = np.array([1.0, 2.0, 3.0])
    result = point_metrics(point, target)

    assert result["mae"] == pytest.approx(1.0)
    assert result["mse"] == pytest.approx(5.0 / 3.0)
    assert result["rmse"] == pytest.approx(np.sqrt(5.0 / 3.0))


def test_interval_metrics_use_closed_coverage_and_match_hand_values():
    lower = np.array([-1.0, 2.0])
    upper = np.array([0.0, 3.0])
    target = np.array([-1.0, 3.0])
    result = interval_metrics(lower, upper, target, coverage=0.9, target_range=4.0)

    assert result["picp"] == 1.0
    assert result["mpiw"] == 1.0
    assert result["pinaw"] == 0.25
    assert result["coverage_error"] == pytest.approx(0.1)
    assert result["interval_score"] == 1.0


def test_interval_score_penalizes_values_outside_interval():
    result = interval_metrics(
        lower=np.array([-1.0]),
        upper=np.array([1.0]),
        target=np.array([-2.0]),
        coverage=0.9,
        target_range=1.0,
    )
    assert result["interval_score"] == pytest.approx(22.0)
    assert result["picp"] == 0.0


def test_compute_metrics_returns_overall_and_every_horizon():
    target = np.array([[[0.0], [2.0]], [[1.0], [3.0]]], dtype=np.float32)
    prediction = np.empty((2, 2, 1, 3), dtype=np.float32)
    prediction[..., 0] = target - 1.0
    prediction[..., 1] = target + 0.5
    prediction[..., 2] = target + 1.0

    result = compute_metrics(prediction, target, coverage=0.9)

    assert result["overall"]["mae"] == pytest.approx(0.5)
    assert result["overall"]["picp"] == 1.0
    assert len(result["by_horizon"]) == 2
    assert [item["horizon"] for item in result["by_horizon"]] == [1, 2]


def test_metrics_weight_every_element_not_every_batch():
    point = np.array([0.0, 0.0, 10.0])
    target = np.zeros(3)
    assert point_metrics(point, target)["mae"] == pytest.approx(10.0 / 3.0)


def test_pinaw_is_none_when_target_range_is_zero():
    result = interval_metrics(
        lower=np.array([0.0]),
        upper=np.array([1.0]),
        target=np.array([0.5]),
        coverage=0.9,
        target_range=0.0,
    )
    assert result["pinaw"] is None


@pytest.mark.parametrize(
    ("prediction", "target"),
    [
        (np.zeros((2, 3, 1)), np.zeros((2, 3, 1))),
        (np.zeros((2, 3, 1, 3)), np.zeros((2, 3, 2))),
    ],
)
def test_compute_metrics_rejects_incompatible_shapes(prediction, target):
    with pytest.raises(ValueError, match="shape"):
        compute_metrics(prediction, target, coverage=0.9)
