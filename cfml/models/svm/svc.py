from typing import *

from .mixins import *
from .kernel import Kernel
from ..bases import ClassifierBase


@ClassifierBase.register("svc")
class SVC(CoreSVCMixin, SVCMixin, ClassifierBase):
    def __init__(self, *,
                 lb: float = 1.,
                 kernel: str = "rbf",
                 kernel_config: Dict[str, Any] = None):
        self._lb = lb
        if kernel_config is None:
            kernel_config = {}
        self._kernel = Kernel(kernel, **kernel_config)


__all__ = ["SVC"]
