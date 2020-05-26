import numpy as np

from typing import Union

from ...misc.toolkit import Activations


def raise_no_gradient():
    raise ValueError("gradient information is cleared / not gathered")


class Layer:
    def __init__(self,
                 num_units: int,
                 activation: Union[str, None]):
        self.forward = None
        self.num_units = num_units
        self.activation = None if activation is None else Activations(activation)

    def __call__(self,
                 x: np.ndarray,
                 w: np.ndarray,
                 b: np.ndarray) -> np.ndarray:
        affine = x.dot(w) + b
        if self.activation is None:
            self.forward = affine
        else:
            self.forward = self.activation(affine)
        return self.forward

    def backward(self,
                 w: np.ndarray,
                 prev_delta: np.ndarray) -> np.ndarray:
        if self.forward is None:
            raise_no_gradient()
        delta = prev_delta.dot(w.T)
        if self.activation is not None:
            delta *= self.activation.grad(self.forward)
        return delta


class Loss:
    def __init__(self,
                 loss: str):
        self._loss = loss
        self._caches = {}

    def __call__(self,
                 output: np.ndarray,
                 target: np.ndarray) -> float:
        return getattr(self, self._loss)(output, target)

    def backward(self) -> np.ndarray:
        if not self._caches:
            raise_no_gradient()
        return getattr(self, f"{self._loss}_backward")()

    def l1(self,
           output: np.ndarray,
           target: np.ndarray) -> float:
        diff = output - target
        self._caches["diff"] = diff
        return np.abs(diff).mean()

    def l1_backward(self) -> np.ndarray:
        diff = self._caches["diff"]
        return np.sign(diff)


class Initializer:
    def __init__(self,
                 initializer: str = "uniform",
                 **kwargs):
        self._initializer = initializer
        self._kwargs = kwargs

    def initialize(self,
                   input_dim: int,
                   output_dim: int) -> np.ndarray:
        return getattr(self, self._initializer)(input_dim, output_dim)

    def uniform(self,
                input_dim: int,
                output_dim: int) -> np.ndarray:
        floor, ceiling = map(self._kwargs.get, ["floor", "ceiling"], [-1., 1.])
        parameter = np.random.random([input_dim, output_dim]).astype(np.float32)
        parameter *= (ceiling - floor)
        parameter += floor
        return parameter


__all__ = ["Layer", "Loss", "Initializer"]
