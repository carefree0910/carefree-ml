import numpy as np

from abc import ABC, abstractmethod


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

    def _initialize(self,
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
        return (x - self._x_mean) / self._x_std

    def normalize_y(self,
                    y: np.ndarray) -> np.ndarray:
        return (y - self._y_mean) / self._y_std

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

    def predict(self,
                x: np.ndarray) -> np.ndarray:
        predictions = self._predict_normalized(self.normalize_x(x))
        self.recover_y(predictions)
        return predictions


__all__ = ["StrMixin", "NormalizeMixin"]
