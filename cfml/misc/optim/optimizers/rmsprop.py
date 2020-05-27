import numpy as np

from typing import *

from ..bases import Optimizer, GradientDescentMixin


@Optimizer.register("rmsprop")
class RMSProp(Optimizer):
    def __init__(self, lr, **kwargs):
        super().__init__(lr)
        self._eps = kwargs.get("eps", 1e-8)
        self._rate = kwargs.get("decay_rate", 0.9)

    def step(self,
             model: GradientDescentMixin,
             gradient_dict: Dict[str, np.ndarray]):
        for key, value in gradient_dict.items():
            cache = self._caches.setdefault(key, np.zeros_like(value))
            cache = self._caches[key] = self._rate * cache + (1 - self._rate) * (value ** 2)
            gradient = value / (np.sqrt(cache + self._eps))
            self.apply(key, gradient, model)


__all__ = ["RMSProp"]
