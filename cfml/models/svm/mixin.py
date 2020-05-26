import numpy as np

from abc import ABCMeta, abstractmethod
from typing import *

from .kernel import Kernel
from ..bases import Base
from ..mixins import NormalizeMixin
from ...misc.optim import GradientDescentMixin


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

    @abstractmethod
    def get_diffs(self,
                  y_batch: np.ndarray,
                  predictions: np.ndarray) -> Dict[str, np.ndarray]:
        pass

    def loss_function(self,
                      x_batch: np.ndarray,
                      y_batch: np.ndarray,
                      batch_indices: np.ndarray) -> Dict[str, Any]:
        ak = self._alpha.dot(self._k_mat)
        predictions = ak.T[batch_indices] + self._b
        diffs = self.get_diffs(y_batch, predictions)
        diff = diffs["diff"]
        critical_mask = (diff > 0).ravel()
        loss = 0.5 * ak.dot(self._alpha.T).item()
        has_critical = np.any(critical_mask)
        if has_critical:
            loss += self.lb * diff[critical_mask].mean().item()
        diffs.update({"loss": loss, "ak": ak, "has_critical": has_critical, "critical_mask": critical_mask})
        return diffs

    def gradient_function(self,
                          x_batch: np.ndarray,
                          y_batch: np.ndarray,
                          batch_indices: np.ndarray,
                          loss_dict: Dict[str, Any]) -> Dict[str, np.ndarray]:
        delta_coeff = loss_dict["delta_coeff"]
        ak, has_critical, critical_mask = map(loss_dict.get, ["ak", "has_critical", "critical_mask"])
        if not has_critical:
            delta = None
            gradient_dict = {"_alpha": ak}
        else:
            k_critical = self._k_mat[batch_indices][critical_mask]
            coeff_critical = delta_coeff[critical_mask]
            delta = self.lb * coeff_critical
            gradient_dict = {"_alpha": ak + (k_critical * delta).mean(0).reshape([1, -1])}
        gb = np.zeros([1, 1], np.float32) if delta is None else delta.mean(0).reshape([1, 1])
        gradient_dict["_b"] = gb
        return gradient_dict

    def _fit_svm(self,
                 x: np.ndarray,
                 y: np.ndarray):
        self._initialize_statistics(x, y)
        self._k_mat = self.kernel.project(self._x_normalized, self._x_normalized)
        self._alpha = np.zeros([1, x.shape[0]], np.float32)
        self._b = np.zeros([1, 1], np.float32)
        self._gradient_descent(self._x_normalized, self._y_normalized)

    def _predict_normalized(self,
                            x_normalized: np.ndarray) -> np.ndarray:
        if self._alpha is None:
            Base.raise_not_fit(self)
        projection = self.kernel.project(self._x_normalized, x_normalized)
        affine = self._alpha.dot(projection) + self._b
        return affine.T


__all__ = ["SVMMixin"]
