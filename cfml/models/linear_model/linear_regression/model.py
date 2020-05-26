from ..mixin import LinearMixin
from ...bases import RegressorBase


@RegressorBase.register("linear_regression")
class LinearRegression(LinearMixin, RegressorBase):
    def __init__(self, *,
                 fit_intersect: bool = True,
                 normalize_labels: bool = True):
        self._w = self._b = None
        self._fit_intersect = fit_intersect
        self._normalize_labels = normalize_labels


__all__ = ["LinearRegression"]
