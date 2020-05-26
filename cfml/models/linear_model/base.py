import numpy as np

from abc import ABCMeta
from typing import *

from ..bases import Base
from ..mixins import NormalizeMixin
from ...misc.optim import GradientDescentMixin


class LinearMixin(NormalizeMixin, GradientDescentMixin, metaclass=ABCMeta):
    @property
    def fit_intersect(self):
        return getattr(self, "_fit_intersect", True)

    def parameter_names(self) -> List[str]:
        parameters = ["_w"]
        if self.fit_intersect:
            parameters.append("_b")
        return parameters

    def loss_function(self,
                      x_batch: np.ndarray,
                      y_batch: np.ndarray,
                      batch_indices: np.ndarray) -> Dict[str, Any]:
        predictions = self._predict_normalized(x_batch)
        diff = predictions - y_batch
        return {"diff": diff, "loss": np.linalg.norm(diff).item()}

    def gradient_function(self,
                          x_batch: np.ndarray,
                          y_batch: np.ndarray,
                          batch_indices: np.ndarray,
                          loss_dict: Dict[str, Any]) -> Dict[str, np.ndarray]:
        diff = loss_dict["diff"]
        gradient_dict = {"_w": (diff * x_batch).mean(0).reshape([-1, 1])}
        if self.fit_intersect:
            gradient_dict["_b"] = diff.mean(0).reshape([1, 1])
        return gradient_dict

    def _fit_linear(self,
                    x: np.ndarray,
                    y: np.ndarray):
        self._initialize_statistics(x, y)
        self._w = np.random.random([x.shape[1], 1])
        if self.fit_intersect:
            self._b = np.random.random([1, 1])
        self._gradient_descent(self._x_normalized, self._y_normalized)

    def _predict_normalized(self,
                            x_normalized: np.ndarray) -> np.ndarray:
        if self._w is None:
            Base.raise_not_fit(self)
        affine = x_normalized.dot(self._w)
        if self.fit_intersect:
            affine += self._b
        return affine


__all__ = ["LinearMixin"]
