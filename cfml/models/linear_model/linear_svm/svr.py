from .mixin import LinearSVMMixin
from ...bases import RegressorBase
from ...svm.mixins import SVRMixin


@RegressorBase.register("linear_svr")
class LinearSVR(SVRMixin, LinearSVMMixin, RegressorBase):
    def __init__(self, *,
                 lb: float = 1.,
                 eps: float = 0.,
                 fit_intersect: bool = True,
                 normalize_labels: bool = True):
        self._lb = lb
        self._eps = eps
        self._w = self._b = None
        self._fit_intersect = fit_intersect
        self._normalize_labels = normalize_labels


__all__ = ["LinearSVR"]
