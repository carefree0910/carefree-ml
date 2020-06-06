import numpy as np
import matplotlib.pyplot as plt

from typing import *
from cftool.ml import *
from cftool.misc import timeit
from cfdata.tabular import TaskTypes


def make_cfml_pattern(cfml_model, requires_prob) -> ModelPattern:
    predict_method = "predict_prob" if requires_prob else "predict"
    return ModelPattern(init_method=lambda: cfml_model, predict_method=predict_method)


def make_sklearn_pattern(sk_model, requires_prob) -> Union[ModelPattern, None]:
    if requires_prob:
        predict_method = getattr(sk_model, "predict_proba", None)
        if predict_method is None:
            return
    else:
        predict_method = lambda x: sk_model.predict(x).reshape([-1, 1])
    return ModelPattern(predict_method=predict_method)


class SklearnComparer:
    def __init__(self,
                 cfml_models: Dict[str, Any],
                 sklearn_models: Dict[str, Any],
                 *,
                 task_type: TaskTypes = TaskTypes.CLASSIFICATION):
        patterns = {}
        models_bundle = [cfml_models, sklearn_models]
        make_functions = [make_cfml_pattern, make_sklearn_pattern]
        if task_type is TaskTypes.REGRESSION:
            metrics = ["mae", "mse"]
            for models, make_function in zip(models_bundle, make_functions):
                new_patterns = {k: make_function(v, False) for k, v in models.items()}
                patterns.update({k: v for k, v in new_patterns.items() if v is not None})
        else:
            metrics = ["auc", "acc"]
            for models, make_function in zip(models_bundle, make_functions):
                for metric in metrics:
                    for model_name, model in models.items():
                        local_patterns = patterns.setdefault(model_name, {})
                        new_pattern = make_function(model, metric in Metrics.requires_prob_metrics)
                        if new_pattern is not None:
                            local_patterns[metric] = new_pattern
        self._core = Comparer(patterns, list(map(Estimator, metrics)))

    def compare(self,
                x: np.ndarray,
                y: np.ndarray):
        self._core.compare(x, y)


class Experiment:
    def __init__(self,
                 cfml_models: Dict[str, Any],
                 sklearn_models: Dict[str, Any],
                 *,
                 task_type: TaskTypes = TaskTypes.CLASSIFICATION,
                 show_images: bool = False):
        self._show_images = show_images
        self._cfml_models = cfml_models
        self._sklearn_models = sklearn_models
        self._comparer = SklearnComparer(cfml_models, sklearn_models, task_type=task_type)

    @staticmethod
    def suppress_warnings():
        def warn(*args, **kwargs):
            pass
        import warnings
        warnings.warn = warn

    def run(self, tr_set, te_set=None):
        if te_set is None:
            te_set = tr_set
        v2d_condition = tr_set.is_clf and tr_set.num_features == 2
        v1d_condition = tr_set.is_reg and tr_set.num_features == 1
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


__all__ = ["SklearnComparer", "Experiment", "Activations"]
