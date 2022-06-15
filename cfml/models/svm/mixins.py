import numpy as np

from typing import *
from abc import ABCMeta, abstractmethod

from .kernel import Kernel
from ..bases import Base
from ..mixins import NormalizeMixin
from ..mixins import BinaryClassifierMixin
from ...misc.optim import GradientDescentMixin
from ...misc.toolkit import Activations


class SVMMixin(NormalizeMixin, GradientDescentMixin, metaclass=ABCMeta):
    @property
    def lb(self):
        return getattr(self, "_lb", 1.0)

    @property
    def kernel(self):
        key = "_kernel"
        kernel = getattr(self, key, None)
        if kernel is None:
            kernel = Kernel()
            setattr(self, key, kernel)
        return kernel

    def parameter_names(self) -> List[str]:
        return ["_alpha", "_b"]

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
        ak = self._alpha.dot(self._k_mat)
        predictions = ak.T[batch_indices] + self._b
        diffs = self.get_diffs(y_batch, predictions)
        diff = diffs["diff"]
        critical_mask = (diff > 0).ravel()
        loss = 0.5 * ak.dot(self._alpha.T).item()
        has_critical = np.any(critical_mask)
        if has_critical:
            loss += self.lb * diff[critical_mask].mean().item()
        diffs.update(
            {
                "loss": loss,
                "ak": ak,
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
        ak, has_critical, critical_mask = map(
            loss_dict.get, ["ak", "has_critical", "critical_mask"]
        )
        if not has_critical:
            delta = None
            gradient_dict = {"_alpha": ak}
        else:
            k_critical = self._k_mat[batch_indices][critical_mask]
            coeff_critical = delta_coeff[critical_mask]
            delta = self.lb * coeff_critical
            gradient_dict = {
                "_alpha": ak + (k_critical * delta).mean(0).reshape([1, -1])
            }
        gb = (
            np.zeros([1, 1], np.float32)
            if delta is None
            else delta.mean(0).reshape([1, 1])
        )
        gradient_dict["_b"] = gb
        return gradient_dict

    def _fit_svm(self, x: np.ndarray, y: np.ndarray):
        self._initialize_statistics(x, y)
        self._k_mat = self.kernel.project(self._x_normalized, self._x_normalized)
        self._alpha = np.zeros([1, x.shape[0]], np.float32)
        self._b = np.zeros([1, 1], np.float32)
        self.gradient_descent(self._x_normalized, self._y_normalized)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "SVMMixin":
        self._fit_svm(x, y)
        return self

    def _predict_normalized(self, x_normalized: np.ndarray) -> np.ndarray:
        if self._alpha is None:
            Base.raise_not_fit(self)
        projection = self.kernel.project(self._x_normalized, x_normalized)
        affine = self._alpha.dot(projection) + self._b
        return affine.T


class SVCMixin(BinaryClassifierMixin, SVMMixin, metaclass=ABCMeta):
    def _fit_core(self, x_processed: np.ndarray, y_processed: np.ndarray):
        self._fit_svm(x_processed, y_processed)

    def predict_prob(self, x: np.ndarray) -> np.ndarray:
        affine = self.predict_raw(x)
        sigmoid = Activations.sigmoid(np.clip(affine, -2.0, 2.0) * 5.0)
        return np.hstack([1.0 - sigmoid, sigmoid])


class SVRMixin(SVMMixin, metaclass=ABCMeta):
    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.predict_raw(x)


class CoreSVCMixin:
    @staticmethod
    def _preprocess_data(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        y_svm = y.copy()
        y_svm[y_svm == 0] = -1
        return x, y_svm

    @staticmethod
    def get_diffs(
        y_batch: np.ndarray,
        predictions: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        return {"diff": 1.0 - y_batch * predictions, "delta_coeff": -y_batch}


class CoreSVRMixin:
    @property
    def eps(self):
        return getattr(self, "_eps", 0.0)

    @property
    def raw_lb(self):
        return getattr(self, "_raw_lb", None)

    def get_diffs(
        self,
        y_batch: np.ndarray,
        predictions: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        if self.raw_lb == "auto":
            self._lb = float(len(y_batch))
        raw_diff = predictions - y_batch
        l1_diff = np.abs(raw_diff)
        if self.eps <= 0.0:
            tube_diff = l1_diff
        else:
            tube_diff = l1_diff - self.eps
        return {"diff": tube_diff, "delta_coeff": np.sign(raw_diff)}


__all__ = ["SVCMixin", "SVRMixin", "CoreSVCMixin", "CoreSVRMixin"]
