import pytest
import torch

from losses.interval_loss import interval_score_loss
from losses.quantile_loss import QuantileLoss


def test_quantile_loss_matches_hand_calculation():
    prediction = torch.tensor([[[[-1.0, 0.0, 1.0]]]])
    target = torch.tensor([[[0.0]]])

    loss = QuantileLoss((0.05, 0.5, 0.95))(prediction, target)

    assert loss.item() == pytest.approx((0.05 + 0.0 + 0.05) / 3.0)


def test_quantile_loss_is_zero_for_exact_standalone_predictions():
    prediction = torch.full((2, 3, 1, 3), 2.5)
    target = torch.full((2, 3, 1), 2.5)
    assert QuantileLoss((0.05, 0.5, 0.95))(prediction, target).item() == 0.0


@pytest.mark.parametrize(
    ("prediction", "target"),
    [
        (torch.zeros(2, 3, 1), torch.zeros(2, 3, 1)),
        (torch.zeros(2, 3, 1, 3), torch.zeros(2, 3, 2)),
        (torch.zeros(2, 3, 1, 2), torch.zeros(2, 3, 1)),
    ],
)
def test_quantile_loss_rejects_shapes_that_would_broadcast(prediction, target):
    with pytest.raises(ValueError, match="shape"):
        QuantileLoss((0.05, 0.5, 0.95))(prediction, target)


def test_quantile_loss_backpropagates_finite_gradients():
    prediction = torch.randn(2, 4, 1, 3, requires_grad=True)
    target = torch.randn(2, 4, 1)
    QuantileLoss((0.05, 0.5, 0.95))(prediction, target).backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_interval_score_loss_matches_inside_and_outside_cases():
    lower = torch.tensor([[[-1.0], [-1.0]]])
    upper = torch.tensor([[[1.0], [1.0]]])
    target = torch.tensor([[[0.0], [-2.0]]])

    loss = interval_score_loss(lower, upper, target, coverage=0.9)

    assert loss.item() == pytest.approx((2.0 + 22.0) / 2.0)

