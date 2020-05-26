import numpy as np

from abc import ABCMeta
from typing import *

from .kernel import Kernel
from ..bases import Base
from ..mixins import NormalizeMixin
from ...misc.optim import GradientDescentMixin
from ...misc.toolkit import Activations, Metrics


class SVMMixin(NormalizeMixin, GradientDescentMixin, metaclass=ABCMeta):
    @property
    def lb(self):
        return getattr(self, "_lb", 1.)

    @property
    def kernel(self):
        kernel = getattr(self, "_kernel", None)
        if kernel is None:
            kernel = Kernel()
            setattr(self, "_kernel", kernel)
        return kernel

    def parameter_names(self) -> List[str]:
        return ["_alpha", "_b"]

    def loss_function(self,
                      x_batch: np.ndarray,
                      y_batch: np.ndarray,
                      batch_indices: np.ndarray) -> Dict[str, Any]:
        ak = self._alpha.dot(self._k_mat)
        predictions = ak.T[batch_indices] + self._b
        diff = 1. - y_batch * predictions
        critical_mask = (diff > 0).ravel()
        loss = 0.5 * ak.dot(self._alpha.T).item()
        has_critical = np.any(critical_mask)
        if has_critical:
            loss += self.lb * diff[critical_mask].mean().item()
        return {"loss": loss, "ak": ak, "has_critical": has_critical, "critical_mask": critical_mask}

    def gradient_function(self,
                          x_batch: np.ndarray,
                          y_batch: np.ndarray,
                          batch_indices: np.ndarray,
                          loss_dict: Dict[str, Any]) -> Dict[str, np.ndarray]:
        ak, has_critical, critical_mask = map(loss_dict.get, ["ak", "has_critical", "critical_mask"])
        if not has_critical:
            delta = None
            gradient_dict = {"_alpha": ak}
        else:
            k_critical, y_critical = self._k_mat[batch_indices][critical_mask], y_batch[critical_mask]
            delta = -self.lb * y_critical
            gradient_dict = {"_alpha": ak + (k_critical * delta).mean(0).reshape([1, -1])}
        gb = np.zeros([1, 1], np.float32) if delta is None else delta.mean(0).reshape([1, 1])
        gradient_dict["_b"] = gb
        return gradient_dict

    def _fit_svm(self,
                 x: np.ndarray,
                 y: np.ndarray):
        self._initialize(x, y)
        self._k_mat = self.kernel.project(self._x_normalized, self._x_normalized)
        self._alpha = np.zeros([1, x.shape[0]], np.float32)
        self._b = np.zeros([1, 1], np.float32)
        self._gradient_descent(self._x_normalized, self._y_normalized)
        probabilities = self.predict_prob(x)
        self.threshold = Metrics.get_binary_threshold(y, probabilities, "acc")

    def _predict_normalized(self,
                            x_normalized: np.ndarray) -> np.ndarray:
        if self._alpha is None:
            Base.raise_not_fit(self)
        projection = self.kernel.project(self._x_normalized, x_normalized)
        affine = self._alpha.dot(projection) + self._b
        return affine.T

    def predict_prob(self,
                     x: np.ndarray) -> np.ndarray:
        affine = NormalizeMixin.predict(self, x)
        sigmoid = Activations.sigmoid(np.clip(affine, -2., 2.) * 5.)
        return np.hstack([1. - sigmoid, sigmoid])


__all__ = ["SVMMixin"]
