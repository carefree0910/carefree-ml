import numpy as np

from .mixin import LinearSVMMixin
from ..mixin import LinearMixin
from ...bases import ClassifierBase
from ...mixins import BinaryClassifierMixin
from ....misc.toolkit import Activations
from ...svm.mixins import SVCMixin


@ClassifierBase.register("linear_svc")
class LinearSVC(ClassifierBase, SVCMixin, LinearSVMMixin, BinaryClassifierMixin):
    def __init__(self, *,
                 lb: float = 1.,
                 fit_intersect: bool = True):
        self._lb = lb
        self._w = self._b = None
        self._fit_intersect = fit_intersect

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
