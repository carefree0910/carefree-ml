import numpy as np

from functools import partial

from ..bases import ClassifierBase
from ...misc.toolkit import Activations


@ClassifierBase.register("gaussian_nb")
class GaussianNB(ClassifierBase):
    def __init__(self, *,
                 alpha: float = 1.,
                 var_smoothing: float = 1e-9):
        self._alpha = alpha
        self._var_smoothing = var_smoothing
        self._mu = self._sigma2 = self._class_prior = None
        self._pi_sigma = self._2sigma2 = None
        self._softmax = Activations("softmax")

    @property
    def mean(self):
        return self._mu

    @property
    def variance(self):
        return self._sigma2

    @property
    def class_prior(self):
        return self._class_prior

    def fit(self,
            x: np.ndarray,
            y: np.ndarray) -> "GaussianNB":
        y = y.ravel()
        y_bincount = np.bincount(y).reshape([1, -1])
        num_samples, num_classes = len(x), y_bincount.shape[1]
        self._class_prior = y_bincount / num_samples
        masked_xs = [x[y == i] for i in range(num_classes)]
        self._mu = np.array(list(map(partial(np.mean, axis=0), masked_xs)))     # [num_classes, num_features]
        self._sigma2 = np.array(list(map(partial(np.var, axis=0), masked_xs)))  # [num_classes, num_features]
        self._sigma2 += self._var_smoothing * np.var(x, axis=0).max()
        self._pi_sigma = ((2 * np.pi * self._sigma2) ** 0.5)[None, ...]
        self._2sigma2 = (2 * self._sigma2)[None, ...]
        return self

    def predict_prob(self,
                     x: np.ndarray) -> np.ndarray:
        subtract_mean = (x[..., None, :] - self._mu) ** 2
        gaussian = np.exp(-subtract_mean / self._2sigma2) / self._pi_sigma
        gaussian = np.prod(gaussian, axis=-1)
        posterior = gaussian * self._class_prior
        return posterior / posterior.sum(1, keepdims=True)


__all__ = ["GaussianNB"]
