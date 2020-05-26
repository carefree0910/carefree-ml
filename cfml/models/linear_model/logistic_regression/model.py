import numpy as np

from ..mixins import LinearBinaryClassifierMixin
from ...bases import ClassifierBase
from ....misc.toolkit import Activations


@ClassifierBase.register("logistic_regression")
class LogisticRegression(LinearBinaryClassifierMixin, ClassifierBase):
    def __init__(self, *,
                 lb: float = 0.,
                 lr: float = 0.1,
                 fit_intersect: bool = True):
        self._lb = lb
        self._lr = lr
        self._fit_intersect = fit_intersect
        self._w = self._b = None
        self._sigmoid = Activations("sigmoid")

    def _predict_normalized(self,
                            x_normalized: np.ndarray) -> np.ndarray:
        affine = super()._predict_normalized(x_normalized)
        return self._sigmoid(affine)

    def predict_prob(self,
                     x: np.ndarray) -> np.ndarray:
        sigmoid = self.predict_raw(x)
        return np.hstack([1. - sigmoid, sigmoid])


__all__ = ["LogisticRegression"]
