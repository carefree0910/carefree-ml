import numpy as np

from ..mixins import StrMixin


class Kernel(StrMixin):
    def __init__(self, kernel: str = None, **kwargs):
        self._name = kernel
        self._kernel, self._kwargs = kernel, kwargs

    def project(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        if self._kernel is None:
            return self.identical(x1, x2)
        return getattr(self, self._kernel)(x1, x2)

    @staticmethod
    def identical(x1, x2):
        return x1.dot(x2.T)

    def poly(self, x1, x2):
        p = self._kwargs.get("p", 3)
        return (x1.dot(x2.T) + 1.) ** p

    def rbf(self, x1, x2):
        gamma = self._kwargs.get("gamma", 1.)
        return np.exp(-gamma * np.sum((x1[..., None, :] - x2) ** 2, axis=2))


__all__ = ["Kernel"]
