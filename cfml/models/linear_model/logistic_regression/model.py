import numpy as np

from typing import *

from ..mixin import LinearMixin
from ...bases import ClassifierBase
from ...mixins import BinaryClassifierMixin
from ....misc.toolkit import Activations


@ClassifierBase.register("logistic_regression")
class LogisticRegression(ClassifierBase, LinearMixin, BinaryClassifierMixin):
    def __init__(self, *,
                 lb: float = 0.,
                 fit_intersect: bool = True):
        self._lb = lb
        self._fit_intersect = fit_intersect
        self._w = self._b = None
        self._sigmoid = Activations("sigmoid")

    def fit(self,
            x: np.ndarray,
            y: np.ndarray) -> "LogisticRegression":
        self.check_binary_classification(y)
        self._fit_linear(x, y)
        self._generate_binary_threshold(x, y)
        return self

    def _predict_normalized(self,
                            x_normalized: np.ndarray) -> np.ndarray:
        affine = super()._predict_normalized(x_normalized)
        return self._sigmoid(affine)

    def predict(self,
                x: np.ndarray) -> np.ndarray:
        return BinaryClassifierMixin.predict(self, x)

    def predict_prob(self,
                     x: np.ndarray) -> np.ndarray:
        sigmoid = LinearMixin.predict(self, x)
        return np.hstack([1. - sigmoid, sigmoid])


__all__ = ["LogisticRegression"]
