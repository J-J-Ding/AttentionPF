"""Training losses for point and interval forecasts."""

from losses.interval_loss import interval_score_loss
from losses.quantile_loss import QuantileLoss

__all__ = ["QuantileLoss", "interval_score_loss"]

