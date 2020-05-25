import numpy as np

from tqdm import tqdm
from typing import *
from abc import ABC, abstractmethod

from ..toolkit import register_core, fix_float_to_length

optimizer_dict: Dict[str, Type["Optimizer"]] = {}


class Optimizer(ABC):
    def __init__(self, lr, **kwargs):
        self._lr = lr

    @abstractmethod
    def step(self,
             model: "GradientDescentMixin",
             gradient_dict: Dict[str, np.ndarray]):
        pass

    @classmethod
    def register(cls, name):
        global optimizer_dict
        return register_core(name, optimizer_dict)


class GradientDescentMixin(ABC):
    @abstractmethod
    def parameter_names(self) -> List[str]:
        """ this method returns all parameters' names, each name should correspond to a property

        e.g. in LinearRegression, if self._w & self._b are the parameters,
        this method should return : ["_w", "_b"]

        Returns
        -------
        names : List[str]

        """

    @abstractmethod
    def loss_function(self,
                      x_batch: np.ndarray,
                      y_batch: np.ndarray) -> Dict[str, Any]:
        """ this method calculate the loss of one batch

        Parameters
        ----------
        x_batch : np.ndarray, one batch of training set features
        y_batch : np.ndarray, one batch of training set labels

        Returns
        -------
        results : Dict[str, Any], contains loss value and intermediate results which are critical
        * must contains 'loss' key, whose value should be a float

        """

    @abstractmethod
    def gradient_function(self,
                          x_batch: np.ndarray,
                          y_batch: np.ndarray,
                          loss_dict: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """ this method calculate the gradients of one batch

        Parameters
        ----------
        x_batch : np.ndarray, one batch of training set features
        y_batch : np.ndarray, one batch of training set labels
        loss_dict : Dict[str, Any], results from self.loss_function

        Returns
        -------
        results : Dict[str, np.ndarray], contains all gradients needed
        * set(results.keys()) should be identical with set(self.parameter_names)

        """

    def _setup_optimizer(self):
        optimizer = getattr(self, "_optimizer", None)
        if optimizer is None:
            self.setup_optimizer("sgd", 0.01)
        elif not isinstance(optimizer, Optimizer):
            raise ValueError(f"optimizer should be subclass of Optimizer, '{type(optimizer)}' found")

    def setup_optimizer(self,
                        optimizer: str,
                        lr: float,
                        *,
                        epoch: int = 20,
                        batch_size: int = 32,
                        **kwargs) -> "GradientDescentMixin":
        self._losses = []
        self._epoch = epoch
        self._batch_size = batch_size
        self._optimizer = optimizer_dict[optimizer](lr, **kwargs)
        return self

    def _gradient_descent(self,
                          x: np.ndarray,
                          y: np.ndarray):
        self._setup_optimizer()
        n_sample = len(x)
        b_size = min(n_sample, self._batch_size)
        n_step = n_sample // b_size
        n_step += int(n_step * b_size < n_sample)
        iterator = tqdm(range(self._epoch), total=self._epoch)
        for _ in iterator:
            local_losses = []
            indices = np.random.permutation(n_sample)
            for j in range(n_step):
                batch_indices = indices[j*b_size:(j+1)*b_size]
                x_batch, y_batch = x[batch_indices], y[batch_indices]
                loss_dict = self.loss_function(x_batch, y_batch)
                gradient_dict = self.gradient_function(x_batch, y_batch, loss_dict)
                self._optimizer.step(self, gradient_dict)
                local_losses.append(loss_dict["loss"])
            iterator.set_postfix({"loss": fix_float_to_length(local_losses[0], 6)})
            self._losses.append(local_losses)


__all__ = ["Optimizer", "GradientDescentMixin", "optimizer_dict"]
