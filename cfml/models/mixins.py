import numpy as np

from typing import *
from abc import ABC, abstractmethod

from ..misc.toolkit import Metrics


class StrMixin:
    @property
    def name(self):
        return getattr(self, "_name", "")

    @property
    def kwargs(self):
        return getattr(self, "_kwargs", {})

    def __str__(self):
        if not self.kwargs:
            kwarg_str = ""
        else:
            kwarg_str = "\n".join([" " * 2 + f"{k} : {v}" for k, v in self.kwargs.items()])
            kwarg_str = f"\n{{\n{kwarg_str}\n}}"
        return f"{type(self).__name__}({self.name}){kwarg_str}"

    __repr__ = __str__


class NormalizeMixin(ABC):
    @property
    def eps(self):
        return 1e-8

    @property
    def x_mean(self):
        return getattr(self, "_x_mean", None)

    @property
    def x_std(self):
        return getattr(self, "_x_std", None)

    @property
    def y_mean(self):
        return getattr(self, "_y_mean", None)

    @property
    def y_std(self):
        return getattr(self, "_y_std", None)

    @property
    def x_normalized(self):
        return getattr(self, "_x_normalized", None)

    @property
    def y_normalized(self):
        return getattr(self, "_y_normalized", None)

    @property
    def normalize_labels(self):
        return getattr(self, "_normalize_labels", False)

    def _initialize_statistics(self,
                               x: np.ndarray,
                               y: np.ndarray):
        self._x_mean, self._x_std = x.mean(0), x.std(0)
        self._x_normalized = self.normalize_x(x)
        if not self.normalize_labels:
            self._y_normalized = y
        else:
            self._y_mean, self._y_std = y.mean(0), y.std(0)
            self._y_normalized = self.normalize_y(y)

    def normalize_x(self,
                    x: np.ndarray) -> np.ndarray:
        return (x - self._x_mean) / (self._x_std + self.eps)

    def normalize_y(self,
                    y: np.ndarray) -> np.ndarray:
        return (y - self._y_mean) / (self._y_std + self.eps)

    def recover_y(self,
                  y_normalized: np.ndarray,
                  *,
                  in_place: bool = True):
        if not self.normalize_labels:
            return y_normalized
        if not in_place:
            y_normalized = y_normalized.copy()
        y_normalized *= self._y_std
        y_normalized += self._y_mean
        return y_normalized

    @abstractmethod
    def _predict_normalized(self,
                            x: np.ndarray) -> np.ndarray:
        pass

    def predict_raw(self,
                    x: np.ndarray) -> np.ndarray:
        predictions = self._predict_normalized(self.normalize_x(x))
        self.recover_y(predictions)
        return predictions


class BinaryClassifierMixin(ABC):
    @property
    def threshold(self):
        return getattr(self, "_binary_threshold", 0.5)

    @property
    def binary_metric(self):
        return getattr(self, "_binary_metric", "acc")

    @property
    def allow_multiclass(self):
        return getattr(self, "_allow_multiclass", False)

    def check_binary_classification(self, y: np.ndarray):
        num_classes = y.max() + 1
        if num_classes > 2:
            raise ValueError(
                f"{type(self).__name__} only supports num_classes=2.\n"
                "* For multi-class problems, please use NeuralNetwork instead"
            )

    @staticmethod
    def _preprocess_data(x: np.ndarray,
                         y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return x, y

    def _generate_binary_threshold(self,
                                   x: np.ndarray,
                                   y: np.ndarray):
        probabilities = self.predict_prob(x)
        self._binary_threshold = Metrics.get_binary_threshold(y, probabilities, self.binary_metric)

    @abstractmethod
    def predict_prob(self,
                     x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def _fit_core(self,
                  x_processed: np.ndarray,
                  y_processed: np.ndarray):
        pass

    def fit(self,
            x: np.ndarray,
            y: np.ndarray) -> "BinaryClassifierMixin":
        if not self.allow_multiclass:
            self.check_binary_classification(y)
        x_processed, y_processed = self._preprocess_data(x, y)
        self._fit_core(x_processed, y_processed)
        self._is_binary = False
        if not self.allow_multiclass or y_processed.shape[1] == 2:
            self._generate_binary_threshold(x, y)
            self._is_binary = True
        return self

    def predict(self,
                x: np.ndarray) -> np.ndarray:
        probabilities = self.predict_prob(x)
        if not self._is_binary:
            return probabilities.argmax(1).reshape([-1, 1])
        return (probabilities[..., 1] >= self.threshold).astype(np.int).reshape([-1, 1])


__all__ = ["StrMixin", "NormalizeMixin", "BinaryClassifierMixin"]
