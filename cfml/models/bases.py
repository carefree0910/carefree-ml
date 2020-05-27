import numpy as np
import matplotlib.pyplot as plt

from typing import *
from abc import ABC, ABCMeta, abstractmethod

from ..misc.toolkit import register_core, Visualizer

model_dict: Dict[str, Type["Base"]] = {}


class Base(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def fit(self,
            x: np.ndarray,
            y: np.ndarray) -> "Base":
        """ fit the model

        Parameters
        ----------
        x : np.ndarray, training set features
        y : np.ndarray, training set labels
        * should be a column vector (i.e. y.shape == [n, 1])

        Returns
        -------
        self : Base

        """

    @abstractmethod
    def predict(self,
                x: np.ndarray) -> np.ndarray:
        """ make predictions with the model

        Parameters
        ----------
        x : np.ndarray, test set features

        Returns
        -------
        y_hat : np.ndarray, the predictions
        * should be a column vector (i.e. y_hat.shape == [n, 1])

        """

    @staticmethod
    def make(model: str, *args, **kwargs) -> "Base":
        return model_dict[model](*args, **kwargs)

    @staticmethod
    def raise_not_fit(model):
        raise ValueError(f"{type(model).__name__} need to be fit before predict")

    @classmethod
    def register(cls, name):
        global model_dict
        return register_core(name, model_dict)


class ClassifierBase(Base, metaclass=ABCMeta):
    def predict(self,
                x: np.ndarray) -> np.ndarray:
        """ make label predictions with the model

        Parameters
        ----------
        x : np.ndarray, test set features

        Returns
        -------
        y_hat : np.ndarray, the label predictions
        * dtype should be np.int
        * should be a column vector (i.e. y_hat.shape == [n, 1])

        """
        return np.argmax(self.predict_prob(x), axis=1).reshape([-1, 1])

    @abstractmethod
    def predict_prob(self,
                     x: np.ndarray) -> np.ndarray:
        """ make probabilistic predictions with the model

        Parameters
        ----------
        x : np.ndarray, test set features

        Returns
        -------
        y_hat : np.ndarray, the probabilistic predictions
        * dtype should be np.float32
        * should be a matrix (i.e. y_hat.shape == [n, k])

        """


class RegressorBase(Base, metaclass=ABCMeta):
    @abstractmethod
    def predict(self,
                x: np.ndarray) -> np.ndarray:
        """ make value predictions with the model

        Parameters
        ----------
        x : np.ndarray, test set features

        Returns
        -------
        y_hat : np.ndarray, the value predictions
        * dtype should be np.float32
        * should be a column vector (i.e. y_hat.shape == [n, 1])

        """

    def visualize1d(self,
                    x: np.ndarray,
                    y: np.ndarray = None,
                    **kwargs) -> "RegressorBase":
        Visualizer.visualize1d(self.predict, x, y, **kwargs)
        return self


__all__ = ["Base", "ClassifierBase", "RegressorBase", "model_dict"]
