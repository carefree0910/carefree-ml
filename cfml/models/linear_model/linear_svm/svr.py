import numpy as np

from typing import *

from .mixin import LinearSVMMixin
from ...bases import RegressorBase


@RegressorBase.register("linear_svr")
class LinearSVR(RegressorBase, LinearSVMMixin):
    def __init__(self, *,
                 lb: float = 1.,
                 eps: float = 0.,
                 fit_intersect: bool = True,
                 normalize_labels: bool = True):
        self._lb = lb
        self._eps = eps
        self._w = self._b = None
        self._fit_intersect = fit_intersect
        self._normalize_labels = normalize_labels

    def get_diffs(self,
                  y_batch: np.ndarray,
                  predictions: np.ndarray) -> Dict[str, np.ndarray]:
        raw_diff = predictions - y_batch
        l1_diff = np.abs(raw_diff)
        if self._eps <= 0.:
            tube_diff = l1_diff
        else:
            tube_diff = l1_diff - self._eps
        return {"diff": tube_diff, "delta_coeff": np.sign(raw_diff)}

    def fit(self,
            x: np.ndarray,
            y: np.ndarray) -> "LinearSVR":
        self._fit_linear(x, y)
        return self

    def predict(self,
                x: np.ndarray) -> np.ndarray:
        return LinearSVMMixin.predict(self, x)


__all__ = ["LinearSVR"]
