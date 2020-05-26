import numpy as np

from typing import *

from .base import SVMMixin
from .kernel import Kernel
from ..bases import ClassifierBase
from ..mixins import BinaryClassifierMixin
from ...misc.toolkit import Activations


@ClassifierBase.register("svc")
class SVC(ClassifierBase, SVMMixin, BinaryClassifierMixin):
    def __init__(self, *,
                 lb: float = 1.,
                 kernel: str = "rbf",
                 kernel_config: Dict[str, Any] = None):
        self._lb = lb
        if kernel_config is None:
            kernel_config = {}
        self._kernel = Kernel(kernel, **kernel_config)
        self._x_mean = self._x_std = None

    def fit(self,
            x: np.ndarray,
            y: np.ndarray) -> "SVC":
        self.check_binary_classification(y)
        y_svm = y.copy()
        y_svm[y_svm == 0] = -1
        self._fit_svm(x, y_svm)
        self._generate_binary_threshold(x, y)
        return self

    def predict(self,
                x: np.ndarray) -> np.ndarray:
        return BinaryClassifierMixin.predict(self, x)

    def predict_prob(self,
                     x: np.ndarray) -> np.ndarray:
        return SVMMixin.predict_prob(self, x)


__all__ = ["SVC"]
