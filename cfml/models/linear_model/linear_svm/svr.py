from .mixins import LinearSVRMixin
from ...bases import RegressorBase
from ...svm.mixins import CoreSVRMixin


@RegressorBase.register("linear_svr")
class LinearSVR(CoreSVRMixin, LinearSVRMixin, RegressorBase):
    def __init__(self, *,
                 lb: float = 1.,
                 eps: float = 0.,
                 optimizer: str = "rmsprop",
                 fit_intersect: bool = True,
                 normalize_labels: bool = True):
        self._lb = lb
        self._eps = eps
        self._opt = optimizer
        self._w = self._b = None
        self._fit_intersect = fit_intersect
        self._normalize_labels = normalize_labels


__all__ = ["LinearSVR"]
