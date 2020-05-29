from ..mixins import LinearRegressorMixin
from ...bases import RegressorBase


@RegressorBase.register("linear_regression")
class LinearRegression(LinearRegressorMixin, RegressorBase):
    def __init__(self, *,
                 lr: float = 0.1,
                 loss: str = "l1",
                 fit_intersect: bool = True,
                 normalize_labels: bool = True):
        self._lr = lr
        self._loss = loss
        self._w = self._b = None
        self._fit_intersect = fit_intersect
        self._normalize_labels = normalize_labels


__all__ = ["LinearRegression"]
