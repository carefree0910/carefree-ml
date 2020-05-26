import numpy as np

from typing import *

from .base import SVMMixin
from .kernel import Kernel
from ..bases import ClassifierBase


@ClassifierBase.register("svc")
class SVC(ClassifierBase, SVMMixin):
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
        y = y.copy()
        y[y == 0] = -1
        self._fit_svm(x, y)
        return self

    def predict(self,
                x: np.ndarray) -> np.ndarray:
        return (self.predict_prob(x)[..., 1] >= self.threshold).astype(np.int).reshape([-1, 1])

    def predict_prob(self,
                     x: np.ndarray) -> np.ndarray:
        return SVMMixin.predict_prob(self, x)


__all__ = ["SVC"]
