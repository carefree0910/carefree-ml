import numpy as np

from abc import ABCMeta, abstractmethod
from typing import *

from .modules import *
from ..mixins import NormalizeMixin, BinaryClassifierMixin
from ...misc.optim import GradientDescentMixin


class FCNNMixin(NormalizeMixin, GradientDescentMixin, metaclass=ABCMeta):
    @property
    def lb(self) -> float:
        return getattr(self, "_lb", 0.)

    @property
    def initializer(self) -> Initializer:
        key = "_initializer"
        initializer = getattr(self, key, None)
        if initializer is None:
            initializer = Initializer()
            setattr(self, key, initializer)
        return initializer

    @property
    def hidden_layers(self) -> List[Layer]:
        key = "_hidden_layers"
        hidden_layers = getattr(self, key, None)
        if hidden_layers is None:
            hidden_layers = [Layer(100, "relu")]
            setattr(self, key, hidden_layers)
        return hidden_layers

    @property
    def all_layers(self) -> List[Layer]:
        return self.hidden_layers + [self._final_affine]

    @property
    def loss(self) -> Loss:
        loss = getattr(self, "_loss", None)
        if loss is None:
            raise ValueError("loss is not defined")
        return loss

    @staticmethod
    def w_key(i):
        return f"_w{i}"

    @staticmethod
    def b_key(i):
        return f"_b{i}"

    def w(self, i):
        return getattr(self, self.w_key(i))

    def b(self, i):
        return getattr(self, self.b_key(i))

    def wb(self, i):
        return self.w(i), self.b(i)

    def parameter_names(self) -> List[str]:
        return sum([[self.w_key(i), self.b_key(i)] for i in range(len(self.hidden_layers) + 1)], [])

    def loss_function(self,
                      x_batch: np.ndarray,
                      y_batch: np.ndarray,
                      batch_indices: np.ndarray) -> Dict[str, Any]:
        net = x_batch
        for i, layer in enumerate(self.hidden_layers):
            net = layer(net, *self.wb(i))
        final_affine = self._final_affine(net, *self.wb(len(self.hidden_layers)))
        loss = self.loss(final_affine, y_batch)
        return {"loss": loss}

    def gradient_function(self,
                          x_batch: np.ndarray,
                          y_batch: np.ndarray,
                          batch_indices: np.ndarray,
                          loss_dict: Dict[str, Any]) -> Dict[str, np.ndarray]:
        num_layers = len(self.hidden_layers)
        # local gradients
        deltas = [self.loss.backward()]
        for i in range(-1, -num_layers-1, -1):
            w = self.w(num_layers + i + 1)
            deltas.append(self.hidden_layers[i].backward(w, deltas[-1]))
        # parameter gradients
        gradient_dict = {}
        all_layers = self.all_layers
        for i in range(num_layers, 0, -1):
            delta = deltas[num_layers - i]
            w_key, b_key = self.w_key(i), self.b_key(i)
            gw = all_layers[i - 1].forward.T.dot(delta)
            gb = np.sum(delta, axis=0, keepdims=True)
            if self.lb > 0.:
                gw += self.lb * self.w(i)
            gradient_dict[w_key], gradient_dict[b_key] = gw, gb

        return gradient_dict

    def _initialize_parameters(self,
                               input_dim: int,
                               output_dim: int):
        in_dim = input_dim
        for i, layer in enumerate(self.hidden_layers):
            out_dim = layer.num_units
            w_key, b_key = self.w_key(i), self.b_key(i)
            setattr(self, w_key, self.initializer.initialize(in_dim, out_dim))
            setattr(self, b_key, self.initializer.initialize(1, out_dim))
            in_dim = out_dim
        num_layers = len(self.hidden_layers)
        w_key, b_key = self.w_key(num_layers), self.b_key(num_layers)
        setattr(self, w_key, self.initializer.initialize(in_dim, output_dim))
        setattr(self, b_key, self.initializer.initialize(1, output_dim))
        self._final_affine = Layer(output_dim, None)

    def _fit_fcnn(self,
                  x: np.ndarray,
                  y: np.ndarray):
        self._initialize_statistics(x, y)
        self._initialize_parameters(x.shape[1], y.shape[1])
        self._gradient_descent(self._x_normalized, self._y_normalized)

    def fit(self,
            x: np.ndarray,
            y: np.ndarray) -> "FCNNMixin":
        self._fit_fcnn(x, y)
        return self

    def _predict_normalized(self,
                            x_normalized: np.ndarray) -> np.ndarray:
        net = x_normalized
        for i, layer in enumerate(self.hidden_layers):
            net = layer(net, *self.wb(i))
        return self._final_affine(net, *self.wb(len(self.hidden_layers)))


class FCNNRegressorMixin(FCNNMixin, metaclass=ABCMeta):
    def predict(self,
                x: np.ndarray) -> np.ndarray:
        return self.predict_raw(x)


__all__ = ["Layer", "Loss", "FCNNRegressorMixin"]
