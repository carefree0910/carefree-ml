import numpy as np

from typing import *

from ..bases import Optimizer, GradientDescentMixin


@Optimizer.register("adam")
class Adam(Optimizer):
    def __init__(self, lr, **kwargs):
        super().__init__(lr)
        self._eps = kwargs.get("eps", 1e-8)
        self._beta1 = kwargs.get("beta1", 0.9)
        self._beta2 = kwargs.get("beta2", 0.999)

    def step(self, model: GradientDescentMixin, gradient_dict: Dict[str, np.ndarray]):
        for key, value in gradient_dict.items():
            key1, key2 = f"{key}_1", f"{key}_2"
            cache1 = self._caches.setdefault(key1, np.zeros_like(value))
            cache2 = self._caches.setdefault(key2, np.zeros_like(value))
            cache1 = cache1 * self._beta1 + (1.0 - self._beta1) * value
            cache2 = cache2 * self._beta2 + (1.0 - self._beta2) * value**2
            self._caches[key1] = cache1
            self._caches[key2] = cache2
            gradient = cache1 / (np.sqrt(cache2 + self._eps))
            self.apply(key, gradient, model)


__all__ = ["Adam"]
