import numpy as np

from abc import ABCMeta
from typing import *
from cftool.optim import GradientDescentMixin

from ..bases import Base
from ..mixins import NormalizeMixin, BinaryClassifierMixin


class LinearMixin(NormalizeMixin, GradientDescentMixin):
    @property
    def lb(self):
        return getattr(self, "_lb", 0.0)

    @property
    def loss(self):
        return getattr(self, "_loss", "mse")

    @property
    def fit_intersect(self):
        return getattr(self, "_fit_intersect", True)

    def parameter_names(self) -> List[str]:
        parameters = ["_w"]
        if self.fit_intersect:
            parameters.append("_b")
        return parameters

    def loss_function(
        self,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        batch_indices: np.ndarray,
    ) -> Dict[str, Any]:
        predictions = self._predict_normalized(x_batch)
        diff = predictions - y_batch
        loss = np.abs(diff).mean() if self.loss == "l1" else np.linalg.norm(diff)
        return {"diff": diff, "loss": loss.item()}

    def gradient_function(
        self,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        batch_indices: np.ndarray,
        loss_dict: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        diff = loss_dict["diff"]
        coeff = np.sign(diff) if self.loss == "l1" else diff
        gradient_dict = {"_w": (coeff * x_batch).mean(0).reshape([-1, 1])}
        if self.lb > 0.0:
            gradient_dict["_w"] += self.lb * self._w
        if self.fit_intersect:
            gradient_dict["_b"] = diff.mean(0).reshape([1, 1])
        return gradient_dict

    def _fit_linear(self, x: np.ndarray, y: np.ndarray):
        self._initialize_statistics(x, y)
        self._w = np.random.random([x.shape[1], 1])
        if self.fit_intersect:
            self._b = np.random.random([1, 1])
        self.gradient_descent(self._x_normalized, self._y_normalized)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LinearMixin":
        self._fit_linear(x, y)
        return self

    def _predict_normalized(self, x_normalized: np.ndarray) -> np.ndarray:
        if self._w is None:
            Base.raise_not_fit(self)
        affine = x_normalized.dot(self._w)
        if self.fit_intersect:
            affine += self._b
        return affine


class LinearRegressorMixin(LinearMixin):
    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.predict_raw(x)


class LinearBinaryClassifierMixin(
    BinaryClassifierMixin,
    LinearMixin,
    metaclass=ABCMeta,
):
    def _fit_core(self, x_processed: np.ndarray, y_processed: np.ndarray):
        self._fit_linear(x_processed, y_processed)


__all__ = ["LinearMixin", "LinearRegressorMixin", "LinearBinaryClassifierMixin"]
