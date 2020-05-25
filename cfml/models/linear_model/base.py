import numpy as np

from abc import ABCMeta
from typing import *

from ...misc.optim import GradientDescentMixin


class LinearMixin(GradientDescentMixin, metaclass=ABCMeta):
    @property
    def fit_intersect(self):
        return getattr(self, "_fit_intersect", True)

    @property
    def normalize_labels(self):
        return getattr(self, "_normalize_labels", False)

    def parameter_names(self) -> List[str]:
        parameters = ["_w"]
        if self.fit_intersect:
            parameters.append("_b")
        return parameters

    def loss_function(self,
                      x_batch: np.ndarray,
                      y_batch: np.ndarray) -> Dict[str, Any]:
        predictions = self._predict_normalized(x_batch)
        diff = predictions - y_batch
        return {"diff": diff, "loss": np.linalg.norm(diff).item()}

    def gradient_function(self,
                          x_batch: np.ndarray,
                          y_batch: np.ndarray,
                          loss_dict: Dict[str, Any]) -> Dict[str, np.ndarray]:
        diff = loss_dict["diff"]
        gradients = {"_w": (diff * x_batch).mean(0).reshape([-1, 1])}
        if self.fit_intersect:
            gradients["_b"] = diff.mean(0).reshape([1, 1])
        return gradients

    def _fit_linear(self,
                    x: np.ndarray,
                    y: np.ndarray):
        self._x_mean, self._x_std = x.mean(0), x.std(0)
        self._w = np.random.random([x.shape[1], 1])
        if self.fit_intersect:
            self._b = np.random.random([1, 1])
        x_normalized = (x - self._x_mean) / self._x_std
        if not self.normalize_labels:
            y_normalized = y
        else:
            self._y_mean, self._y_std = y.mean(0), y.std(0)
            y_normalized = (y - self._y_mean) / self._y_std
        self._gradient_descent(x_normalized, y_normalized)

    def _predict_normalized(self,
                            x_normalized: np.ndarray) -> np.ndarray:
        if self._w is None:
            raise ValueError("LinearRegression need to be fit before predict")
        affine = x_normalized.dot(self._w)
        if self.fit_intersect:
            affine += self._b
        return affine

    def predict(self,
                x: np.ndarray) -> np.ndarray:
        predictions = self._predict_normalized((x - self._x_mean) / self._x_std)
        if self.normalize_labels:
            predictions *= self._y_std
            predictions += self._y_mean
        return predictions


__all__ = ["LinearMixin"]
