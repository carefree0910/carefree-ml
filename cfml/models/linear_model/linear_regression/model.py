import numpy as np

from typing import *

from ..mixin import LinearMixin
from ...bases import RegressorBase


@RegressorBase.register("linear_regression")
class LinearRegression(RegressorBase, LinearMixin):
    def __init__(self, *,
                 fit_intersect: bool = True,
                 normalize_labels: bool = True):
        self._w = self._b = None
        self._fit_intersect = fit_intersect
        self._normalize_labels = normalize_labels

    def fit(self,
            x: np.ndarray,
            y: np.ndarray) -> "LinearRegression":
        self._fit_linear(x, y)
        return self

    def predict(self,
                x: np.ndarray) -> np.ndarray:
        return LinearMixin.predict(self, x)


__all__ = ["LinearRegression"]
