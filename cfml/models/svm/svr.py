import numpy as np

from typing import *

from .mixin import SVMMixin
from .kernel import Kernel
from ..bases import RegressorBase
from ..mixins import NormalizeMixin


@RegressorBase.register("svr")
class SVR(RegressorBase, SVMMixin, NormalizeMixin):
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

    def get_diffs(self,
                  y_batch: np.ndarray,
                  predictions: np.ndarray) -> Dict[str, np.ndarray]:
        if self._raw_lb == "auto":
            self._lb = float(len(y_batch))
        raw_diff = predictions - y_batch
        l1_diff = np.abs(raw_diff)
        if self._eps <= 0.:
            tube_diff = l1_diff
        else:
            tube_diff = l1_diff - self._eps
        return {"diff": tube_diff, "delta_coeff": np.sign(raw_diff)}

    def fit(self,
            x: np.ndarray,
            y: np.ndarray) -> "SVR":
        self._fit_svm(x, y)
        return self

    def predict(self,
                x: np.ndarray) -> np.ndarray:
        return NormalizeMixin.predict(self, x)


__all__ = ["SVR"]
