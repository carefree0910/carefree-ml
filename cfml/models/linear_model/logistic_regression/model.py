import numpy as np

from typing import *

from ..base import LinearMixin
from ...bases import ClassifierBase
from ....misc.toolkit import Activations


@ClassifierBase.register("logistic_regression")
class LogisticRegression(ClassifierBase, LinearMixin):
    def __init__(self, *,
                 lb: float = 0.,
                 fit_intersect: bool = True):
        self._lb = lb
        self._fit_intersect = fit_intersect
        self._w = self._b = None
        self._x_mean = self._x_std = None
        self._sigmoid = Activations("sigmoid")

    def gradient_function(self,
                          x_batch: np.ndarray,
                          y_batch: np.ndarray,
                          loss_dict: Dict[str, Any]) -> Dict[str, np.ndarray]:
        gradient_dict = super().gradient_function(x_batch, y_batch, loss_dict)
        if self._lb > 0.:
            gradient_dict["_w"] += self._lb * self._w
        return gradient_dict

    def fit(self,
            x: np.ndarray,
            y: np.ndarray) -> "LogisticRegression":
        num_classes = self.get_num_classes(y)
        if num_classes > 2:
            raise ValueError(
                "LogisticRegression only supports num_classes=2.\n"
                "For multi-class problems, please use NeuralNetwork instead"
            )
        self._fit_linear(x, y)
        return self

    def _predict_normalized(self,
                            x_normalized: np.ndarray) -> np.ndarray:
        affine = super()._predict_normalized(x_normalized)
        return self._sigmoid(affine)

    def predict_prob(self,
                     x: np.ndarray) -> np.ndarray:
        sigmoid = LinearMixin.predict(self, x)
        return np.hstack([1. - sigmoid, sigmoid])


__all__ = ["LogisticRegression"]
