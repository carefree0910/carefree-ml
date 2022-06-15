import numpy as np
import matplotlib.pyplot as plt

from typing import *
from tqdm import tqdm
from functools import partial
from abc import abstractmethod
from abc import ABC
from cftool.misc import register_core
from cftool.misc import fix_float_to_length


optimizer_dict: Dict[str, Type["Optimizer"]] = {}


class Optimizer(ABC):
    def __init__(self, lr, **kwargs):
        self._lr = lr
        self._caches = {}

    @abstractmethod
    def step(
        self,
        model: "GradientDescentMixin",
        gradient_dict: Dict[str, np.ndarray],
    ):
        pass

    def reset(self):
        self._caches.clear()

    def apply(self, key, gradient, model):
        attr = getattr(model, key)
        attr -= self._lr * gradient
        setattr(model, key, attr)

    @classmethod
    def register(cls, name):
        global optimizer_dict
        return register_core(name, optimizer_dict)


class GradientDescentMixin(ABC):
    @property
    def lr(self):
        return getattr(self, "_lr", 0.01)

    @property
    def opt(self):
        return getattr(self, "_opt", "sgd")

    @property
    def epoch(self):
        return getattr(self, "_epoch", 20)

    @property
    def batch_size(self):
        return getattr(self, "_batch_size", 32)

    @property
    def optimizer_config(self):
        return getattr(self, "_optimizer_config", {})

    @property
    def show_tqdm(self):
        return getattr(self, "_show_tqdm", True)

    @show_tqdm.setter
    def show_tqdm(self, value: bool):
        self._show_tqdm = value

    @abstractmethod
    def parameter_names(self) -> List[str]:
        """
        This method returns all parameters' names, each name should
        correspond to a property.

        e.g. in LinearRegression, if self._w & self._b are the parameters,
        this method should return : ["_w", "_b"].

        Returns
        -------
        names : List[str]

        """

    @abstractmethod
    def loss_function(
        self,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        batch_indices: np.ndarray,
    ) -> Dict[str, Any]:
        """
        This method calculate the loss of one batch.

        Parameters
        ----------
        x_batch : np.ndarray, one batch of training set features.
        y_batch : np.ndarray, one batch of training set labels.
        batch_indices : np.ndarray, indices of current batch.

        Returns
        -------
        results : Dict[str, Any], contains loss value and intermediate results
                                  which are critical.
        * Must contains 'loss' key, whose value should be a float.

        """

    @abstractmethod
    def gradient_function(
        self,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        batch_indices: np.ndarray,
        loss_dict: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        """
        This method calculate the gradients of one batch.

        Parameters
        ----------
        x_batch : np.ndarray, one batch of training set features.
        y_batch : np.ndarray, one batch of training set labels.
        batch_indices : np.ndarray, indices of current batch.
        loss_dict : Dict[str, Any], results from self.loss_function.

        Returns
        -------
        results : Dict[str, np.ndarray], contains all gradients needed.
        * set(results.keys()) should be identical with set(self.parameter_names)

        """

    def _setup_optimizer(self, **kwargs):
        if getattr(self, "_optimizer", None) is None:
            self._optimizer = optimizer_dict[self.opt](self.lr, **kwargs)

    def setup_optimizer(
        self,
        optimizer: str,
        lr: float,
        *,
        epoch: int = 20,
        batch_size: int = 32,
        **kwargs
    ) -> "GradientDescentMixin":
        self._lr = lr
        self._opt = optimizer
        self._epoch = epoch
        self._batch_size = batch_size
        self._setup_optimizer(**kwargs)
        return self

    def gradient_descent(self, x: np.ndarray, y: np.ndarray) -> "GradientDescentMixin":
        self._setup_optimizer(**self.optimizer_config)
        self._optimizer.reset()
        n_sample = len(x)
        b_size = min(n_sample, self.batch_size)
        n_step = n_sample // b_size
        n_step += int(n_step * b_size < n_sample)
        iterator = range(self.epoch)
        if self.show_tqdm:
            iterator = tqdm(iterator, total=self.epoch)
        self._losses = []
        for _ in iterator:
            local_losses = []
            indices = np.random.permutation(n_sample)
            for j in range(n_step):
                batch_indices = indices[j * b_size : (j + 1) * b_size]
                x_batch, y_batch = x[batch_indices], y[batch_indices]
                loss_dict = self.loss_function(x_batch, y_batch, batch_indices)
                gradient_dict = self.gradient_function(
                    x_batch, y_batch, batch_indices, loss_dict
                )
                self._optimizer.step(self, gradient_dict)
                local_losses.append(loss_dict["loss"])
            if self.show_tqdm:
                iterator.set_postfix({"loss": fix_float_to_length(local_losses[0], 6)})
            self._losses.append(local_losses)
        return self

    def plot_loss_curve(self) -> "GradientDescentMixin":
        base = np.arange(len(self._losses))
        to_array = partial(np.array, dtype=np.float32)
        losses = list(map(to_array, self._losses))
        mean, std = map(
            to_array,
            map(list, [map(np.mean, losses), map(np.std, losses)]),
        )
        ax = plt.gca()
        ax.plot(base, mean)
        plt.title("loss curve")
        ax.fill_between(base, mean + std, mean - std, color="#b9cfe7")
        plt.show()
        return self


__all__ = ["Optimizer", "GradientDescentMixin", "optimizer_dict"]
