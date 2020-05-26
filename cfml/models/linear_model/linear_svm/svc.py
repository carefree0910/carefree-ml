from .mixin import LinearSVCMixin
from ...bases import ClassifierBase
from ...svm.mixins import CoreSVCMixin


@ClassifierBase.register("linear_svc")
class LinearSVC(CoreSVCMixin, LinearSVCMixin, ClassifierBase):
    def __init__(self, *,
                 lb: float = 1.,
                 fit_intersect: bool = True):
        self._lb = lb
        self._w = self._b = None
        self._fit_intersect = fit_intersect


__all__ = ["LinearSVC"]
