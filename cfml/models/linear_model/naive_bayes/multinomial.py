import numpy as np

from functools import partial

from ...bases import ClassifierBase
from ....misc.toolkit import Activations


@ClassifierBase.register("multinomial_nb")
class MultinomialNB(ClassifierBase):
    def __init__(self, *,
                 alpha: float = 1.):
        self._alpha = alpha
        self._w = self._b = None
        self._softmax = Activations("softmax")

    @property
    def class_log_prior(self):
        return self._b

    @property
    def feature_log_prob(self):
        return self._w.T

    def fit(self,
            x: np.ndarray,
            y: np.ndarray) -> "MultinomialNB":
        y = y.ravel()
        y_bincount = np.bincount(y).reshape([1, -1])
        num_samples, num_classes = len(x), y_bincount.shape[1]
        self._b = np.log(y_bincount / num_samples)
        masked_xs = [x[y == i] for i in range(num_classes)]
        feature_counts = np.array(list(map(partial(np.sum, axis=0), masked_xs)))
        smoothed_fc = feature_counts + self._alpha
        self._w = np.log(smoothed_fc / smoothed_fc.sum(1, keepdims=True)).T
        return self

    def predict_prob(self,
                     x: np.ndarray) -> np.ndarray:
        affine = x.dot(self._w) + self._b
        return self._softmax(affine)


__all__ = ["MultinomialNB"]
