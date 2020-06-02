import numpy as np
import matplotlib.pyplot as plt

from cftool import *
from typing import *
from functools import partial


class Comparer:
    def __init__(self,
                 cfml_models: Dict[str, Any],
                 sklearn_models: Dict[str, Any],
                 *,
                 dtype: str = "clf"):
        self._dtype = dtype
        sklearn_predict = lambda arr, sklearn_model: sklearn_model.predict(arr).reshape([-1, 1])
        predict_methods = {k: v.predict for k, v in cfml_models.items()}
        predict_methods.update({
            k: partial(sklearn_predict, sklearn_model=v)
            for k, v in sklearn_models.items()
        })
        if dtype == "reg":
            self._l1_estimator = Estimator("mae")
            self._mse_estimator = Estimator("mse")
            self._methods = predict_methods
        else:
            self._auc_estimator = Estimator("auc")
            self._acc_estimator = Estimator("acc")
            self._acc_methods = predict_methods
            self._auc_methods = {k: v.predict_prob for k, v in cfml_models.items()}
            self._auc_methods.update({
                k: getattr(v, "predict_proba", None)
                for k, v in sklearn_models.items()
            })

    def compare(self,
                x: np.ndarray,
                y: np.ndarray):
        if self._dtype == "reg":
            self._l1_estimator.estimate(x, y, self._methods)
            self._mse_estimator.estimate(x, y, self._methods)
        else:
            self._auc_estimator.estimate(x, y, self._auc_methods)
            self._acc_estimator.estimate(x, y, self._acc_methods)


class Experiment:
    def __init__(self,
                 cfml_models: Dict[str, Any],
                 sklearn_models: Dict[str, Any],
                 *,
                 dtype: str = "clf",
                 show_images: bool = False):
        self._show_images = show_images
        self._cfml_models = cfml_models
        self._sklearn_models = sklearn_models
        self._comparer = Comparer(cfml_models, sklearn_models, dtype=dtype)

    @staticmethod
    def suppress_warnings():
        def warn(*args, **kwargs):
            pass
        import warnings
        warnings.warn = warn

    def run(self, tr_set, te_set=None):
        if te_set is None:
            te_set = tr_set
        v2d_condition = tr_set.dtype == "clf" and tr_set.num_features == 2
        v1d_condition = tr_set.dtype == "reg" and tr_set.num_features == 1
        for model_name, model in self._cfml_models.items():
            model.show_tqdm = False
            with timeit(model_name, precision=8):
                model.fit(*tr_set.xy)
            if self._show_images:
                if v2d_condition:
                    model.visualize2d(*te_set.xy)
                elif v1d_condition:
                    model.visualize1d(*te_set.xy)
                else:
                    plot = getattr(model, "plot_loss_curve", None)
                    if plot is not None:
                        plot()
        for model_name, model in self._sklearn_models.items():
            with timeit(model_name, precision=8):
                model.fit(tr_set.x, tr_set.y.ravel())
            if self._show_images:
                if v2d_condition:
                    Visualizer.visualize2d(model.predict, *te_set.xy)
                elif v1d_condition:
                    Visualizer.visualize1d(model.predict, *te_set.xy)
        self._comparer.compare(*te_set.xy)


class Activations:
    def __init__(self, activation: str):
        self._activation = activation

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return getattr(Activations, self._activation)(x)

    def grad(self, forward: np.ndarray) -> np.ndarray:
        return getattr(Activations, f"{self._activation}_grad")(forward)

    def visualize(self, x_min: float = -5., x_max: float = 5.):
        plt.figure()
        x0 = np.linspace(x_min, x_max)
        plt.plot(x0, self(x0))
        plt.show()

    @staticmethod
    def relu(x):
        return np.maximum(0., x)

    @staticmethod
    def relu_grad(forward):
        return (forward != 0.).astype(np.float32)

    @staticmethod
    def sigmoid(x):
        return 1. / (1. + np.exp(-x))

    @staticmethod
    def sigmoid_grad(forward):
        return forward * (1. - forward)

    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    @staticmethod
    def softmax_grad(forward):
        return forward * (1. - forward)


__all__ = ["Comparer", "Activations", "Experiment"]
