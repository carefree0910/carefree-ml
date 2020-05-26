import numpy as np

from typing import *

from .mixins import *
from .kernel import Kernel
from ..bases import RegressorBase


@RegressorBase.register("svr")
class SVR(RegressorBase, SVRMixin, SVMMixin):
    def __init__(self, *,
                 eps: float = 0.,
                 lb: Union[str, float] = "auto",
                 kernel: str = "rbf",
                 kernel_config: Dict[str, Any] = None):
        self._eps = eps
        self._lb = self._raw_lb = lb
        if kernel_config is None:
            kernel_config = {}
        self._kernel = Kernel(kernel, **kernel_config)
        self._normalize_labels = True

    def fit(self,
            x: np.ndarray,
            y: np.ndarray) -> "SVR":
        self._fit_svm(x, y)
        return self

    def predict(self,
                x: np.ndarray) -> np.ndarray:
        return SVMMixin.predict(self, x)


__all__ = ["SVR"]
