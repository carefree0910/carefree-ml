import numpy as np

from typing import *

from ..base import LinearMixin
from ...bases import ClassifierBase
from ....misc.toolkit import Activations


@ClassifierBase.register("linear_svm")
class LinearSVM(ClassifierBase, LinearMixin):
    def __init__(self, *,
                 lb: float = 1.,
                 fit_intersect: bool = True):
        self._lb = lb
        self._w = self._b = None
        self._x_mean = self._x_std = None
        self._fit_intersect = fit_intersect

    def loss_function(self,
                      x_batch: np.ndarray,
                      y_batch: np.ndarray) -> Dict[str, Any]:
        w_norm = np.linalg.norm(self._w)
        predictions = self._predict_normalized(x_batch)
        diff = 1. - y_batch * predictions
        critical_mask = (diff > 0).ravel()
        if not np.any(critical_mask):
            loss = 0.
        else:
            loss = 0.5 * w_norm + self._lb * diff[critical_mask].mean()
        return {"loss": loss, "critical_mask": critical_mask}

    def gradient_function(self,
                          x_batch: np.ndarray,
                          y_batch: np.ndarray,
                          loss_dict: Dict[str, Any]) -> Dict[str, np.ndarray]:
        if loss_dict["loss"] <= 0.:
            zero = np.zeros(1, np.float32)
            return {"_w": zero, "_b": zero}
        critical_mask = loss_dict["critical_mask"]
        x_critical, y_critical = x_batch[critical_mask], y_batch[critical_mask]
        delta = -self._lb * y_critical
        gradient_dict = {"_w": self._w + (x_critical * delta).sum(0).reshape([-1, 1])}
        if self.fit_intersect:
            gradient_dict["_b"] = delta.sum(0).reshape([1, 1])
        return gradient_dict

    def fit(self,
            x: np.ndarray,
            y: np.ndarray) -> "LinearSVM":
        self.check_binary_classification(y)
        y = y.copy()
        y[y == 0] = -1
        self._fit_linear(x, y)
        return self

    def predict_prob(self,
                     x: np.ndarray) -> np.ndarray:
        affine = LinearMixin.predict(self, x)
        sigmoid = Activations.sigmoid(affine * 5.)
        return np.hstack([1. - sigmoid, sigmoid])


__all__ = ["LinearSVM"]
