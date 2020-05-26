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
        self._log_class_prior = np.log(self._class_prior)
        masked_xs = [x[y == i] for i in range(num_classes)]
        self._mu = np.array(list(map(partial(np.mean, axis=0), masked_xs)))     # [num_classes, num_features]
        self._sigma2 = np.array(list(map(partial(np.var, axis=0), masked_xs)))  # [num_classes, num_features]
        self._sigma2 += self._var_smoothing * np.var(x, axis=0).max()
        self._log_pi_sigma = np.log(np.sqrt(2 * np.pi * self._sigma2))[None, ...]
        self._2sigma2 = (2 * self._sigma2)[None, ...]
        return self

    def predict_prob(self,
                     x: np.ndarray) -> np.ndarray:
        subtract_mean = (x[..., None, :] - self._mu) ** 2
        log_gaussian = -subtract_mean / self._2sigma2 - self._log_pi_sigma
        log_gaussian = log_gaussian.sum(-1)
        log_posterior = log_gaussian + self._log_class_prior
        return self._softmax(log_posterior)


__all__ = ["GaussianNB"]
