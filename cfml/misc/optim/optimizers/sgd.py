import numpy as np

from typing import *

from ..bases import Optimizer, GradientDescentMixin


@Optimizer.register("sgd")
class SGD(Optimizer):
    def __init__(self, lr, **kwargs):
        super().__init__(lr)
        self._momentum = kwargs.get("momentum", 0.)

    def step(self,
             model: GradientDescentMixin,
             gradient_dict: Dict[str, np.ndarray]):
        for key, value in gradient_dict.items():
            self.apply(key, value, model)


__all__ = ["SGD"]
