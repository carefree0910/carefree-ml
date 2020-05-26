import numpy as np

from typing import *

from ..bases import Optimizer, GradientDescentMixin


@Optimizer.register("sgd")
class SGD(Optimizer):
    def __init__(self, lr, **kwargs):
        super().__init__(lr)
        self._caches = {}
        self._momentum = kwargs.get("momentum", 0.)
        self._nesterov = kwargs.get("nesterov", False)

    def step(self,
             model: GradientDescentMixin,
             gradient_dict: Dict[str, np.ndarray]):
        for key, value in gradient_dict.items():
            if self._momentum <= 0.:
                gradient = value
            else:
                velocity = self._caches.setdefault(key, np.zeros_like(value))
                velocity = self._momentum * velocity + value
                if self._nesterov:
                    velocity = self._momentum * velocity + value
                gradient = self._caches[key] = velocity
            self.apply(key, gradient, model)


__all__ = ["SGD"]
