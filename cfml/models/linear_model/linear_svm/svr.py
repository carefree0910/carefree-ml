import numpy as np

from typing import *

from .mixin import LinearSVMMixin
from ..base import LinearMixin
from ...bases import RegressorBase


@RegressorBase.register("linear_svr")
class LinearSVR(RegressorBase, LinearMixin, LinearSVMMixin):
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

    def loss_function(self,
                      x_batch: np.ndarray,
                      y_batch: np.ndarray,
                      batch_indices: np.ndarray) -> Dict[str, Any]:
        predictions = self._predict_normalized(x_batch)
        raw_diff = predictions - y_batch
        l1_diff = np.abs(raw_diff)
        if self._eps <= 0.:
            tube_diff = l1_diff
        else:
            tube_diff = l1_diff - self._eps
        loss_dict = self.loss_core(tube_diff)
        loss_dict["raw_diff"] = raw_diff
        return loss_dict

    def gradient_function(self,
                          x_batch: np.ndarray,
                          y_batch: np.ndarray,
                          batch_indices: np.ndarray,
                          loss_dict: Dict[str, Any]) -> Dict[str, np.ndarray]:
        raw_diff, has_critical, critical_mask = map(loss_dict.get, ["raw_diff", "has_critical", "critical_mask"])
        if not has_critical:
            delta = None
            gradient_dict = {"_w": self._w}
        else:
            x_critical, raw_diff_critical = x_batch[critical_mask], raw_diff[critical_mask]
            delta = self._lb * np.sign(raw_diff_critical)
            gradient_dict = {"_w": self._w + (x_critical * delta).mean(0).reshape([-1, 1])}
        if self.fit_intersect:
            gb = np.zeros([1, 1], np.float32) if delta is None else delta.mean(0).reshape([1, 1])
            gradient_dict["_b"] = gb
        return gradient_dict

    def fit(self,
            x: np.ndarray,
            y: np.ndarray) -> "LinearSVR":
        self._fit_linear(x, y)
        return self

    def predict(self,
                x: np.ndarray) -> np.ndarray:
        return LinearMixin.predict(self, x)


__all__ = ["LinearSVR"]
