import numpy as np

from typing import *

from ..base import LinearMixin
from ...bases import ClassifierBase
from ...mixins import BinaryClassifierMixin
from ....misc.toolkit import Activations


@ClassifierBase.register("linear_svc")
class LinearSVC(ClassifierBase, LinearMixin, BinaryClassifierMixin):
    def __init__(self, *,
                 lb: float = 1.,
                 fit_intersect: bool = True):
        self._lb = lb
        self._w = self._b = None
        self._x_mean = self._x_std = None
        self._fit_intersect = fit_intersect

    def loss_function(self,
                      x_batch: np.ndarray,
                      y_batch: np.ndarray,
                      batch_indices: np.ndarray) -> Dict[str, Any]:
        predictions = self._predict_normalized(x_batch)
        diff = 1. - y_batch * predictions
        critical_mask = (diff > 0).ravel()
        loss = 0.5 * np.linalg.norm(self._w)
        has_critical = np.any(critical_mask)
        if has_critical:
            loss += self._lb * diff[critical_mask].mean()
        return {"loss": loss, "has_critical": has_critical, "critical_mask": critical_mask}

    def gradient_function(self,
                          x_batch: np.ndarray,
                          y_batch: np.ndarray,
                          batch_indices: np.ndarray,
                          loss_dict: Dict[str, Any]) -> Dict[str, np.ndarray]:
        has_critical, critical_mask = map(loss_dict.get, ["has_critical", "critical_mask"])
        if not has_critical:
            delta = None
            gradient_dict = {"_w": self._w}
        else:
            x_critical, y_critical = x_batch[critical_mask], y_batch[critical_mask]
            delta = -self._lb * y_critical
            gradient_dict = {"_w": self._w + (x_critical * delta).sum(0).reshape([-1, 1])}
        if self.fit_intersect:
            gb = np.zeros([1, 1], np.float32) if delta is None else delta.sum(0).reshape([1, 1])
            gradient_dict["_b"] = gb
        return gradient_dict

    def fit(self,
            x: np.ndarray,
            y: np.ndarray) -> "LinearSVC":
        self.check_binary_classification(y)
        y_svm = y.copy()
        y_svm[y_svm == 0] = -1
        self._fit_linear(x, y_svm)
        self._generate_binary_threshold(x, y)
        return self

    def predict(self,
                x: np.ndarray) -> np.ndarray:
        return BinaryClassifierMixin.predict(self, x)

    def predict_prob(self,
                     x: np.ndarray) -> np.ndarray:
        affine = LinearMixin.predict(self, x)
        sigmoid = Activations.sigmoid(affine * 5.)
        return np.hstack([1. - sigmoid, sigmoid])


__all__ = ["LinearSVC"]
