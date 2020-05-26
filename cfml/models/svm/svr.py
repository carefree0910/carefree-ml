from typing import *

from .mixins import *
from .kernel import Kernel
from ..bases import RegressorBase


@RegressorBase.register("svr")
class SVR(CoreSVRMixin, SVRMixin, RegressorBase):
    def __init__(self, *,
                 eps: float = 0.,
                 kernel: str = "rbf",
                 optimizer: str = "rmsprop",
                 lb: Union[str, float] = "auto",
                 kernel_config: Dict[str, Any] = None):
        self._eps = eps
        self._opt = optimizer
        self._lb = self._raw_lb = lb
        if kernel_config is None:
            kernel_config = {}
        self._kernel = Kernel(kernel, **kernel_config)
        self._normalize_labels = True


__all__ = ["SVR"]
