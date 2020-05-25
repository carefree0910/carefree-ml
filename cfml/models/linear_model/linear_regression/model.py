import numpy as np

from typing import *

from ..base import LinearMixin
from ...bases import RegressorBase


@RegressorBase.register("linear_regression")
class LinearRegression(RegressorBase, LinearMixin):
    def __init__(self, *,
                 fit_intersect: bool = True,
                 normalize_labels: bool = True):
        self._w = self._b = None
        self._x_mean = self._x_std = None
        self._y_mean = self._y_std = None
        self._fit_intersect = fit_intersect
        self._normalize_labels = normalize_labels

    def parameter_names(self) -> List[str]:
        parameters = ["_w"]
        if self._fit_intersect:
            parameters.append("_b")
        return parameters

    def loss_function(self,
                      x_batch: np.ndarray,
                      y_batch: np.ndarray) -> Dict[str, Any]:
        prediction = self._predict_normalized(x_batch)
        diff = prediction - y_batch
        return {"diff": diff, "loss": np.linalg.norm(diff).item()}

    def gradient_function(self,
                          x_batch: np.ndarray,
                          y_batch: np.ndarray,
                          loss_dict: Dict[str, Any]) -> Dict[str, np.ndarray]:
        diff = loss_dict["diff"]
        gradients = {"_w": (diff * x_batch).mean(0).reshape([-1, 1])}
        if self._fit_intersect:
            gradients["_b"] = diff.mean(0).reshape([1, 1])
        return gradients

    def fit(self,
            x: np.ndarray,
            y: np.ndarray) -> "LinearRegression":
        self._fit_linear(x, y)
        return self

    def predict(self,
                x: np.ndarray) -> np.ndarray:
        return LinearMixin.predict(self, x)


__all__ = ["LinearRegression"]
