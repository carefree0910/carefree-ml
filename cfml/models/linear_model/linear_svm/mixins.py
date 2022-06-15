import numpy as np

from abc import ABCMeta, abstractmethod
from typing import Any, Dict

from ..mixins import *
from ....misc.toolkit import Activations


class LinearSVMMixin(LinearMixin, metaclass=ABCMeta):
    @property
    def w(self):
        return getattr(self, "_w")

    @property
    def lb(self):
        return getattr(self, "_lb")

    @abstractmethod
    def get_diffs(
        self,
        y_batch: np.ndarray,
        predictions: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        pass

    def loss_function(
        self,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        batch_indices: np.ndarray,
    ) -> Dict[str, Any]:
        predictions = self._predict_normalized(x_batch)
        diffs = self.get_diffs(y_batch, predictions)
        diff = diffs["diff"]
        critical_mask = (diff > 0).ravel()
        loss = 0.5 * np.linalg.norm(self.w)
        has_critical = np.any(critical_mask)
        if has_critical:
            loss += self.lb * diff[critical_mask].mean()
        diffs.update(
            {
                "loss": loss,
                "has_critical": has_critical,
                "critical_mask": critical_mask,
            }
        )
        return diffs

    def gradient_function(
        self,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        batch_indices: np.ndarray,
        loss_dict: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        delta_coeff = loss_dict["delta_coeff"]
        has_critical = loss_dict["has_critical"]
        critical_mask = loss_dict["critical_mask"]
        if not has_critical:
            delta = None
            gradient_dict = {"_w": self._w}
        else:
            x_critical = x_batch[critical_mask]
            coeff_critical = delta_coeff[critical_mask]
            delta = self.lb * coeff_critical
            gradient_dict = {
                "_w": self._w + (x_critical * delta).sum(0).reshape([-1, 1])
            }
        if self.fit_intersect:
            gb = (
                np.zeros([1, 1], np.float32)
                if delta is None
                else delta.sum(0).reshape([1, 1])
            )
            gradient_dict["_b"] = gb
        return gradient_dict


class LinearSVCMixin(LinearBinaryClassifierMixin, LinearSVMMixin, metaclass=ABCMeta):
    def predict_prob(self, x: np.ndarray) -> np.ndarray:
        affine = self.predict_raw(x)
        sigmoid = Activations.sigmoid(affine * 5.0)
        return np.hstack([1.0 - sigmoid, sigmoid])


class LinearSVRMixin(LinearRegressorMixin, LinearSVMMixin, metaclass=ABCMeta):
    pass


__all__ = ["LinearSVCMixin", "LinearSVRMixin"]
