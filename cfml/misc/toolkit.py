import os
import math
import shutil
import logging

import numpy as np
import matplotlib.pyplot as plt

from typing import *
from cftool.misc import *
from abc import abstractmethod
from abc import ABC
from sklearn import metrics
from functools import partial
from cfdata.tabular import TaskTypes
from scipy import stats as ss


def show_or_save(
    export_path: str,
    fig: Optional[plt.figure] = None,
    **kwargs: Any,
) -> None:
    """
    Utility function to deal with figure.

    Parameters
    ----------
    export_path : {None, str}
    * If None, the figure will be shown.
    * If str, it represents the path where the figure should be saved to.
    fig : {None, plt.Figure}
    * If None, default figure contained in plt will be executed.
    * If plt.figure, it will be executed

    """

    if export_path is None:
        fig.show(**kwargs) if fig is not None else plt.show(**kwargs)
    else:
        if fig is not None:
            fig.savefig(export_path)
        else:
            plt.savefig(export_path, **kwargs)
    plt.close()


def show_or_return(return_canvas: bool) -> Union[None, np.ndarray]:
    """
    Utility function to deal with current plt.

    Parameters
    ----------
    return_canvas : bool, whether return canvas or not.

    """

    if not return_canvas:
        plt.show()
        return

    buffer_ = io.BytesIO()
    plt.savefig(buffer_, format="png")
    plt.close()
    buffer_.seek(0)
    image = Image.open(buffer_)
    canvas = np.asarray(image)[..., :3]
    buffer_.close()
    return canvas


class Anneal:
    """
    Util class which can provide annealed numbers with given `method`.
    * Formulas could be found in `_initialize` method.

    Parameters
    ----------
    method : str, indicates which anneal method to be used.
    n_iter : int, indicates how much 'steps' will be taken to reach
                  `ceiling` from `floor`.
    floor : float, indicates the start point of the annealed number.
    ceiling : float, indicates the end point of the annealed number.

    Examples
    --------
    >>> from cfml.misc.toolkit import Anneal
    >>>
    >>> anneal = Anneal("linear", 50, 0.01, 0.99)
    >>> for i in range(100):
    >>>     # for i == 0, 1, ..., 48, 49, it will pop 0.01, 0.03, ..., 0.97, 0.99
    >>>     # for i == 50, 51, ..., 98, 99, it will pop 0.99, 0.99, ..., 0.99, 0.99
    >>>     print(anneal.pop())

    """

    def __init__(self, method, n_iter, floor, ceiling):
        self._n_iter = max(1, n_iter)
        self._method, self._max_iter = method, n_iter
        self._floor, self._ceiling = floor, ceiling
        self._cache = self._rs = self._cursor = 0
        self._initialize()

    def __str__(self):
        return f"Anneal({self._method})"

    __repr__ = __str__

    def _initialize(self):
        n_anneal = max(1, self._n_iter - 1)
        if self._method == "linear":
            self._cache = (self._ceiling - self._floor) / n_anneal
        elif self._method == "log":
            self._cache = (self._ceiling - self._floor) / math.log(n_anneal)
        elif self._method == "quad":
            self._cache = (self._ceiling - self._floor) / (n_anneal**2)
        elif self._method == "sigmoid":
            self._cache = 8 / n_anneal
        self._rs = self._floor - self._cache

    def _update_linear(self):
        self._rs += self._cache

    def _update_log(self):
        self._rs = math.log(self._cursor) * self._cache

    def _update_quad(self):
        self._rs = self._cursor**2 * self._cache

    def _update_sigmoid(self):
        self._rs = self._ceiling / (1 + math.exp(4 - self._cursor * self._cache))

    def pop(self):
        self._cursor += 1
        if self._cursor >= self._n_iter:
            return self._ceiling
        getattr(self, f"_update_{self._method}")()
        return self._rs

    def visualize(self):
        rs = [self.pop() for _ in range(self._max_iter)]
        plt.figure()
        plt.plot(range(len(rs)), rs)
        plt.show()
        self._initialize()
        return self


class Metrics(LoggingMixin):
    """
    Util class to calculate a whole variety of metrics.

    Warnings
    ----------
    * Notice that 2-dimensional arrays are desired, not flattened arrays.
    * Notice that first two args of each metric method must be `y` & `pred`.

    Parameters
    ----------
    metric_type : str, indicates which kind of metric is to be calculated.
    config : dict, configuration for the specific metric.
    * e.g. For quantile metric, you need to specify which quantile is to be evaluated.
    verbose_level : int, verbose level of `Metrics`.

    Examples
    --------
    >>> import numpy as np
    >>>
    >>> from cfml.misc.toolkit import Metrics
    >>>
    >>> predictions, y_true = map(np.atleast_2d, [[1., 2., 3.], [0., 2., 1.]])
    >>> print(Metrics("mae", {}).metric(y_true.T, predictions.T))  # will be 1.

    """

    sign_dict = {
        "f1_score": 1,
        "r2_score": 1,
        "auc": 1,
        "acc": 1,
        "mae": -1,
        "mse": -1,
        "ber": -1,
        "quantile": -1,
        "cdf_loss": -1,
        "correlation": 1,
    }
    requires_prob_metrics = {"auc"}
    optimized_binary_metrics = {"acc", "ber"}
    custom_metrics = {}

    def __init__(self, metric_type=None, config=None, verbose_level=None):
        if config is None:
            config = {}
        self.type = metric_type
        self.config = config
        self._verbose_level = verbose_level

    def __str__(self):
        return f"Metrics({self.type})"

    __repr__ = __str__

    @property
    def sign(self):
        return Metrics.sign_dict[self.type]

    @property
    def use_loss(self):
        return self.type == "loss"

    @property
    def requires_prob(self):
        return self.type in self.requires_prob_metrics

    def _handle_nan(self, y, pred):
        pred_valid_mask = np.all(~np.isnan(pred), axis=1)
        valid_ratio = pred_valid_mask.mean()
        if valid_ratio == 0:
            self.log_msg("all pred are nan", self.error_prefix, 2, logging.ERROR)
            return None, None
        if valid_ratio != 1:
            self.log_msg(
                f"pred contains nan (ratio={valid_ratio:6.4f})",
                self.error_prefix,
                2,
                logging.ERROR,
            )
            y, pred = y[pred_valid_mask], pred[pred_valid_mask]
        return y, pred

    @classmethod
    def add_metric(cls, f, name, sign, requires_prob):
        if name in cls.sign_dict:
            print(
                f"{LoggingMixin.warning_prefix}'{name}' "
                "is already registered in Metrics"
            )
        cls.sign_dict[name] = sign
        cls.custom_metrics[name] = {
            "f": f,
            "sign": sign,
            "requires_prob": requires_prob,
        }
        if requires_prob:
            cls.requires_prob_metrics.add(name)

    def metric(self, y, pred, **kwargs: Any):
        if self.type is None:
            msg = "`score` method was called but type is not specified in Metrics"
            raise ValueError(msg)
        y, pred = self._handle_nan(y, pred)
        if y is None or pred is None:
            return float("nan")
        custom_metric_info = self.custom_metrics.get(self.type)
        if custom_metric_info is not None:
            return custom_metric_info["f"](self, y, pred, **kwargs)
        return getattr(self, self.type)(y, pred, **kwargs)

    # config-dependent metrics

    def quantile(self, y, pred):
        q, error = self.config["q"], y - pred
        if not isinstance(q, float):
            q = np.asarray(q, np.float32).reshape([1, -1])
        return np.maximum(q * error, (q - 1) * error).mean(0).sum()

    def cdf_loss(self, y, pred, yq=None):
        if yq is None:
            eps = self.config.setdefault("eps", 1e-6)
            mask = y <= self.config["anchor"]
            pred = np.clip(pred, eps, 1 - eps)
            cdf_raw = pred / (1 - pred)
            return -np.mean(mask * cdf_raw - np.log(1 - pred))
        q, self.config["q"] = self.config.get("q"), pred
        loss = self.quantile(y, yq)
        if q is None:
            self.config.pop("q")
        else:
            self.config["q"] = q
        return loss

    # static metrics

    @staticmethod
    def f1_score(y, pred):
        return metrics.f1_score(y.ravel(), pred.ravel())

    @staticmethod
    def r2_score(y, pred):
        return metrics.r2_score(y.ravel(), pred.ravel())

    @staticmethod
    def auc(y, pred):
        n_classes = pred.shape[1]
        if n_classes == 2:
            return metrics.roc_auc_score(y.ravel(), pred[..., 1])
        return metrics.roc_auc_score(y.ravel(), pred, multi_class="ovr")

    @staticmethod
    def acc(y, pred):
        return np.mean(y == pred)

    @staticmethod
    def mae(y, pred):
        return np.mean(np.abs(y - pred))

    @staticmethod
    def mse(y, pred):
        return np.mean(np.square(y - pred))

    @staticmethod
    def ber(y, pred):
        mat = metrics.confusion_matrix(y.ravel(), pred.ravel())
        tp = np.diag(mat)
        fp = mat.sum(axis=0) - tp
        fn = mat.sum(axis=1) - tp
        tn = mat.sum() - (tp + fp + fn)
        return 0.5 * np.mean((fn / (tp + fn) + fp / (tn + fp)))

    @staticmethod
    def correlation(y, pred):
        return float(ss.pearsonr(y.ravel(), pred.ravel())[0])

    # auxiliaries

    @staticmethod
    def get_binary_threshold(y, probabilities, metric_type):
        pos_probabilities = probabilities[..., 1]
        fpr, tpr, thresholds = metrics.roc_curve(y, pos_probabilities)
        _, counts = np.unique(y, return_counts=True)
        pos = counts[1] / len(y)
        if metric_type == "ber":
            metric = 0.5 * (1 - tpr + fpr)
        elif metric_type == "acc":
            metric = tpr * pos + (1 - fpr) * (1 - pos)
        else:
            msg = f"transformation from fpr, tpr -> '{metric_type}' is not implemented"
            raise NotImplementedError(msg)
        metric *= Metrics.sign_dict[metric_type]
        return thresholds[np.argmax(metric)]


def register_metric(name, sign, requires_prob):
    def _register(f):
        Metrics.add_metric(f, name, sign, requires_prob)
        return f

    return _register


generic_data_type = Union[np.ndarray, Any]
estimate_fn_type = Callable[[generic_data_type], np.ndarray]
scoring_fn_type = Callable[[List[float], float, float], float]
collate_fn_type = Callable[[List[np.ndarray], bool], np.ndarray]
predict_method_type = Optional[estimate_fn_type]


class Statistics(NamedTuple):
    msg: str
    data: Dict[str, Dict[str, float]]


class Estimator(LoggingMixin):
    """
    Util class to estimate the performances of a group of methods,
    on specific dataset & metric.

    Parameters
    ----------
    metric_type : str, indicates which kind of metric is to be calculated.
    verbose_level : int, verbose level used in `LoggingMixin`.
    metric_config : used to initialize `Metrics` instance.

    Examples
    --------
    >>> import numpy as np
    >>>
    >>> from cfml.misc.toolkit import Estimator
    >>>
    >>> x, y = map(np.atleast_2d, [[1., 2., 3.], [0., 2., 1.]])
    >>> identical = lambda x_: x_
    >>> minus_one = lambda x_: x_ - 1
    >>> # >  [ info ] Results
    >>> # ==========================================================
    >>> # |             identical  |    mae     |  1.000000  |
    >>> # |             minus_one  |    mae     |  0.666667  |  <-
    >>> # ----------------------------------------------------------
    >>> estimator = Estimator("mae")
    >>> estimator.estimate(x, y, {"identical": identical, "minus_one": minus_one})

    """

    def __init__(
        self,
        metric_type: str,
        *,
        verbose_level: int = 2,
        metric_config: Optional[Dict[str, Any]] = None,
    ):
        self._reset()
        self._verbose_level = verbose_level
        self._metric = Metrics(metric_type, metric_config)

    def __str__(self):
        return f"Estimator({self.type})"

    __repr__ = __str__

    @property
    def type(self) -> str:
        return self._metric.type

    @property
    def sign(self) -> int:
        return self._metric.sign

    @property
    def requires_prob(self) -> bool:
        return self._metric.requires_prob

    # Core

    def _reset(self):
        self.raw_metrics = {}
        self.final_scores = {}
        self.best_method = None

    def _default_scoring(self, raw_metrics, mean, std) -> float:
        return mean - self.sign * std

    def _mean_scoring(self, raw_metrics, mean, std) -> float:
        return mean

    def _std_scoring(self, raw_metrics, mean, std) -> float:
        return mean + self.sign * std

    # API

    def scoring_fn(self, scoring_function: str) -> Callable[[np.ndarray], float]:
        score_fn = getattr(self, f"_{scoring_function}_scoring")

        def _inner(raw_metrics: np.ndarray) -> float:
            mean = raw_metrics.mean().item()
            std = raw_metrics.std().item()
            return score_fn(raw_metrics, mean, std) * self.sign

        return _inner

    def get_statistics(
        self,
        scoring_function: Union[str, scoring_fn_type] = "default",
    ) -> Statistics:
        msg_list = []
        statistics = {}
        best_idx, best_score = -1, -math.inf
        sorted_method_names = sorted(self.raw_metrics)
        if not isinstance(scoring_function, str):
            scoring_fn = scoring_function
        else:
            scoring_fn = self.scoring_fn(scoring_function)
        for i, name in enumerate(sorted_method_names):
            raw_metrics = self.raw_metrics[name]
            mean, std = raw_metrics.mean().item(), raw_metrics.std().item()
            msg = f"|  {name:>20s}  |  {self.type:^8s}  |  {mean:8.6f} ± {std:8.6f}  |"
            msg_list.append(msg)
            new_score = scoring_fn(raw_metrics)
            self.final_scores[name] = new_score
            if new_score > best_score:
                best_idx, best_score = i, new_score
            method_statistics = statistics.setdefault(name, {})
            method_statistics["mean"], method_statistics["std"] = mean, std
            method_statistics["score"] = new_score
        self.best_method = sorted_method_names[best_idx]
        msg_list[best_idx] += "  <-  "
        width = max(map(len, msg_list))
        msg_list.insert(0, "=" * width)
        msg_list.append("-" * width)
        return Statistics("\n".join(msg_list), statistics)

    def estimate(
        self,
        x: generic_data_type,
        y: generic_data_type,
        methods: Dict[str, Union[estimate_fn_type, List[estimate_fn_type]]],
        *,
        scoring_function: Union[str, scoring_fn_type] = "default",
        verbose_level: int = 1,
    ) -> Dict[str, Dict[str, float]]:
        self._reset()
        for k, v in methods.items():
            if not isinstance(v, list):
                methods[k] = [v]
        self.raw_metrics = {
            name: np.array(
                [self._metric.metric(y, method(x)) for method in sub_methods],
                np.float32,
            )
            for name, sub_methods in methods.items()
        }
        statistics = self.get_statistics(scoring_function)
        self.log_block_msg(statistics.msg, self.info_prefix, "Results", verbose_level)
        return statistics.data

    def select(self, method_names: List[str]) -> "Estimator":
        new_estimator = Estimator(
            self.type,
            verbose_level=self._verbose_level,
            metric_config=self._metric.config,
        )
        new_estimator.raw_metrics = {k: self.raw_metrics[k] for k in method_names}
        new_estimator.final_scores = {k: self.final_scores[k] for k in method_names}
        return new_estimator

    @classmethod
    def merge(cls, estimators: List["Estimator"]) -> "Estimator":
        new_raw_metrics, new_final_scores = {}, {}
        for estimator in estimators:
            for key, value in estimator.raw_metrics.items():
                final_score = estimator.final_scores[key]
                new_raw_metrics.setdefault(key, []).append(value)
                new_final_scores.setdefault(key, []).append(final_score)
        new_raw_metrics = {k: np.concatenate(v) for k, v in new_raw_metrics.items()}
        new_final_scores = {k: sum(v) / len(v) for k, v in new_final_scores.items()}
        first_estimator = estimators[0]
        first_metric_ins = first_estimator._metric
        new_estimator = cls(
            first_metric_ins.type,
            verbose_level=first_estimator._verbose_level,
            metric_config=first_metric_ins.config,
        )
        new_estimator.raw_metrics = new_raw_metrics
        new_estimator.final_scores = new_final_scores
        return new_estimator


class PatternBase(ABC):
    @abstractmethod
    def predict_method(self, requires_prob: bool) -> predict_method_type:
        pass

    def predict(
        self,
        x: Union[np.ndarray, Any],
        *,
        requires_prob: bool = False,
    ) -> np.ndarray:
        predict_method = self.predict_method(requires_prob)
        if predict_method is None:
            msg = f"predicting with requires_prob={requires_prob} is not defined"
            raise ValueError(msg)
        return predict_method(x)


class ModelPattern(PatternBase, LoggingMixin):
    """
    Util class to create an interface for users to leverage `Comparer` & `HPO`
    (and more in the future).

    Parameters
    ----------
    init_method : Callable[[], object]
    * If None, then `ModelPattern` will not perform model creation.
    * If Callable, then `ModelPattern` will initialize a model with it.
    train_method : Callable[[object], None]
    * If None, then `ModelPattern` will not perform model training.
    * If Callable, then `ModelPattern` will train the created model
      (from `init_method`) with it.
    predict_method : Union[str, Callable[[np.ndarray], np.ndarray]]
    * If str, then `ModelPattern` will use `getattr` to get the label
      predict method of the model obtained from above. In this case, `init_method`
      must be provided (`train_method` is still optional, because you can create
      a trained model in `init_method`).
    * If Callable, then `ModelPattern` will use it for label prediction.
    * Notice that predict_method should return a column vector
      (e.g. out.shape = [n, 1])
    predict_prob_method : Union[str, Callable[[np.ndarray], np.ndarray]]
    * If str, then `ModelPattern` will use `getattr` to get the probability prediction
      method of the model obtained from above. In this case, `init_method` must be
      provided (`train_method` is still optional, because you can create a trained
      model in `init_method`).
    * If Callable, then `ModelPattern` will use it for probability prediction.

    Examples
    --------
    >>> import numpy as np
    >>>
    >>> from cfml.misc.toolkit import ModelPattern
    >>>
    >>> x, y = map(np.atleast_2d, [[1., 2., 3.], [0., 2., 1.]])
    >>> predict_method = lambda x_: x_ - 1
    >>> init_method = lambda: type(
    >>>     "Test",
    >>>     (),
    >>>     {"predict": lambda self, x_: predict_method(x_)},
    >>> )()
    >>> # Will both be [[0., 1., 2.]]
    >>> ModelPattern(init_method=init_method).predict(x)
    >>> ModelPattern(predict_method=predict_method).predict(x)

    """

    def __init__(
        self,
        *,
        init_method: Optional[Callable[[], object]] = None,
        train_method: Optional[Callable[[object], None]] = None,
        predict_method: Union[str, predict_method_type] = "predict",
        predict_prob_method: Union[str, predict_method_type] = "predict_prob",
        verbose_level: int = 2,
    ):
        if init_method is None:
            self.model = None
        else:
            self.model = init_method()
        if train_method is not None:
            train_method(self.model)
        self._predict_method = predict_method
        self._predict_prob_method = predict_prob_method
        self._verbose_level = verbose_level

    def predict_method(self, requires_prob: bool) -> predict_method_type:
        predict_method = (
            self._predict_prob_method if requires_prob else self._predict_method
        )
        if isinstance(predict_method, str):
            if self.model is None:
                raise ValueError(
                    "Either init_method or Callable predict_method is required "
                    f"in ModelPattern (requires_prob={requires_prob})"
                )
            predict_method = getattr(self.model, predict_method, None)
        return predict_method

    @classmethod
    def repeat(cls, n: int, **kwargs) -> List["ModelPattern"]:
        return [cls(**kwargs) for _ in range(n)]


class EnsemblePattern(PatternBase):
    """
    Util class to create an interface for users to leverage `Comparer` & `HPO`
    in an ensembled way.

    Parameters
    ----------
    model_patterns : List[ModelPattern]
      list of `ModelPattern` we want to ensemble from.
    ensemble_method : Union[str, collate_fn_type]
      ensemble method we use to collate the results.
    * If str, then `EnsemblePattern` will use `getattr` to get the collate function.
        * Currently only 'default' is supported, which implements voting for
          classification and averaging for regression.
    * If collate_fn_type, then `EnsemblePattern` will use it to collate the
      results directly.

    Examples
    --------
    >>> import numpy as np
    >>>
    >>> from cfml.misc.toolkit import ModelPattern, EnsemblePattern
    >>>
    >>> x, y = map(np.atleast_2d, [[1., 2., 3.], [0., 2., 1.]])
    >>> identical = lambda x_: x_
    >>> minus_one = lambda x_: x_ - 1
    >>> identical_pattern = ModelPattern(predict_method=identical)
    >>> minus_one_pattern = ModelPattern(predict_method=minus_one)
    >>> # Averaging 'identical' & 'minus_one' -> 'minus_0.5'
    >>> ensemble = EnsemblePattern([identical_pattern, minus_one_pattern])
    >>> # [[0.5 1.5 2.5]]
    >>> print(ensemble.predict(x))

    """

    def __init__(
        self,
        model_patterns: List[ModelPattern],
        ensemble_method: Union[str, collate_fn_type] = "default",
    ):
        self._patterns = model_patterns
        self._ensemble_method = ensemble_method

    def __len__(self):
        return len(self._patterns)

    @property
    def collate_fn(self) -> collate_fn_type:
        if callable(self._ensemble_method):
            return self._ensemble_method
        return getattr(self, f"_{self._ensemble_method}_collate")

    # Core

    @staticmethod
    def vote(arr: np.ndarray, num_classes: int) -> np.ndarray:
        counts = np.apply_along_axis(
            partial(np.bincount, minlength=num_classes), 1, arr
        )
        return counts.argmax(1).reshape([-1, 1])

    @staticmethod
    def _default_collate(arrays: List[np.ndarray], requires_prob: bool) -> np.ndarray:
        predictions = np.array(arrays)
        if not requires_prob and np.issubdtype(predictions.dtype, np.integer):
            num_classes = predictions.max() + 1
            if len(predictions.shape) == 3:
                predictions = predictions.squeeze(2)
            predictions = predictions.T
            return EnsemblePattern.vote(predictions, num_classes)
        return predictions.mean(0)

    # API

    def predict_method(self, requires_prob: bool) -> predict_method_type:
        predict_methods = list(
            map(
                ModelPattern.predict_method,
                self._patterns,
                len(self) * [requires_prob],
            )
        )
        predict_methods = [method for method in predict_methods if method is not None]
        if not predict_methods:
            return

        def _predict(x: np.ndarray):
            predictions = [method(x) for method in predict_methods]
            return self.collate_fn(predictions, requires_prob)

        return _predict

    @classmethod
    def from_same_methods(
        cls,
        n: int,
        ensemble_method: Union[str, collate_fn_type] = "default",
        **kwargs,
    ):
        return cls([ModelPattern(**kwargs) for _ in range(n)], ensemble_method)


pattern_type = Union[ModelPattern, EnsemblePattern]
patterns_type = Union[pattern_type, List[pattern_type]]
choices_type = Optional[List[Optional[Union[int, Set[int]]]]]


class Comparer(LoggingMixin):
    """
    Util class to compare a group of `patterns_type`s on a group of `Estimator`s.

    Parameters
    ----------
    patterns : Dict[str, Union[patterns_type, Dict[str, patterns_type]]]
    * If values are `patterns_type`, then all estimators will use this only
      `patterns_type` make predictions.
    * If values are Dict[str, patterns_type], then each estimator will use
      values.get(estimator.type) to make predictions. If corresponding `patterns`
      does not exist (values.get(estimator.type) is None), then corresponding
      estimation will be skipped.
    estimators : List[Estimator], list of estimators which we are interested in.
    verbose_level : int, verbose level used in `LoggingMixin`.

    Examples
    --------
    >>> import numpy as np
    >>>
    >>> from cfml.misc.toolkit import ModelPattern, Estimator, Comparer
    >>>
    >>> x, y = map(np.atleast_2d, [[1., 2., 3.], [0., 2., 1.]])
    >>> identical = lambda x_: x_
    >>> minus_one = lambda x_: x_ - 1
    >>> patterns = {
    >>>     "identical": ModelPattern(predict_method=identical),
    >>>     "minus_one": ModelPattern(predict_method=minus_one)
    >>> }
    >>> estimators = [Estimator("mse"), Estimator("mae")]
    >>> # >  [ info ] Results
    >>> # ==========================================================
    >>> # |             identical  |    mse     |  1.666667  |
    >>> # |             minus_one  |    mse     |  0.666667  |  <-
    >>> # ----------------------------------------------------------
    >>> # >  [ info ] Results
    >>> # ==========================================================
    >>> # |             identical  |    mae     |  1.000000  |
    >>> # |             minus_one  |    mae     |  0.666667  |  <-
    >>> # ----------------------------------------------------------
    >>> comparer = Comparer(patterns, estimators).compare(x, y)
    >>> # {'mse': {'identical': 1.666667, 'minus_one': 0.666666},
    >>> # 'mae': {'identical': 1.0, 'minus_one': 0.666666}}
    >>> print(comparer.final_scores)
    >>> # {'mse': 'minus_one', 'mae': 'minus_one'}
    >>> print(comparer.best_methods)

    """

    def __init__(
        self,
        patterns: Dict[str, Union[patterns_type, Dict[str, patterns_type]]],
        estimators: List[Estimator],
        *,
        verbose_level: int = 2,
    ):
        self.patterns = patterns
        self.estimators = dict(
            zip(
                [estimator.type for estimator in estimators],
                estimators,
            )
        )
        self._verbose_level = verbose_level

    @property
    def raw_metrics(self) -> Dict[str, Dict[str, np.ndarray]]:
        return {k: v.raw_metrics for k, v in self.estimators.items()}

    @property
    def final_scores(self) -> Dict[str, Dict[str, float]]:
        return {k: v.final_scores for k, v in self.estimators.items()}

    @property
    def best_methods(self) -> Dict[str, str]:
        return {k: v.best_method for k, v in self.estimators.items()}

    def log_statistics(self, verbose_level: Optional[int] = 1, **kwargs) -> str:
        sorted_metrics = sorted(self.estimator_statistics)
        body = {}
        same_choices: choices_type = None
        best_choices: choices_type = None
        need_display_best_choice = False
        sub_header = sorted_methods = None
        stat_types = ["mean", "std", "score"]
        for metric_idx, metric_type in enumerate(sorted_metrics):
            statistics = self.estimator_statistics[metric_type]
            if sorted_methods is None:
                sorted_methods = sorted(statistics)
                need_display_best_choice = len(sorted_methods) > 1
            if sub_header is None:
                sub_header = stat_types * len(sorted_metrics)
            if best_choices is None and need_display_best_choice:
                same_choices = [None] * len(sub_header)
                best_choices = [None] * len(sub_header)
            for method_idx, method in enumerate(sorted_methods):
                method_statistics = statistics.get(method)
                if method_statistics is None:
                    method_statistics = [math.nan for _ in stat_types]
                else:
                    method_statistics = [
                        method_statistics[stat_type] for stat_type in stat_types
                    ]
                    if best_choices is not None:
                        for stat_idx, method_statistic in enumerate(method_statistics):
                            choice_idx = metric_idx * len(stat_types) + stat_idx
                            current_idx_choice = best_choices[choice_idx]
                            if current_idx_choice is None:
                                best_choices[choice_idx] = method_idx
                            else:
                                stat_type = stat_types[stat_idx]
                                chosen_stat = statistics[
                                    sorted_methods[current_idx_choice]
                                ][stat_type]
                                if method_statistic == chosen_stat:
                                    if same_choices[choice_idx] is None:
                                        same_choices[choice_idx] = {method_idx}
                                    else:
                                        same_choices[choice_idx].add(method_idx)
                                elif stat_type == "std":
                                    if method_statistic < chosen_stat:
                                        same_choices[choice_idx] = None
                                        best_choices[choice_idx] = method_idx
                                elif stat_type == "score":
                                    if method_statistic > chosen_stat:
                                        same_choices[choice_idx] = None
                                        best_choices[choice_idx] = method_idx
                                else:
                                    assert stat_type == "mean"
                                    sign = Metrics.sign_dict[metric_type]
                                    if method_statistic * sign > chosen_stat * sign:
                                        same_choices[choice_idx] = None
                                        best_choices[choice_idx] = method_idx
                body.setdefault(method, []).extend(method_statistics)
        padding = 2 * (kwargs.get("padding", 1) + 3)
        method_length = kwargs.get("method_length", 16) + padding
        float_length = kwargs.get("float_length", 8)
        cell_length = float_length + padding
        num_statistic_types = len(stat_types)
        metric_type_length = num_statistic_types * cell_length + 2
        header_msg = (
            f"|{'metrics':^{method_length}s}|"
            + "|".join(
                [
                    f"{metric_type:^{metric_type_length}s}"
                    for metric_type in sorted_metrics
                ]
            )
            + "|"
        )
        subs = [f"{sub_header_item:^{cell_length}s}" for sub_header_item in sub_header]
        sub_header_msg = f"|{' ' * method_length}|" + "|".join(subs) + "|"
        body_msgs = []
        for method_idx, method in enumerate(sorted_methods):
            cell_msgs = []
            for cell_idx, cell_item in enumerate(body[method]):
                cell_str = fix_float_to_length(cell_item, float_length)
                if best_choices is not None and (
                    best_choices[cell_idx] == method_idx
                    or same_choices[cell_idx] is not None
                    and method_idx in same_choices[cell_idx]
                ):
                    cell_str = f" -- {cell_str} -- "
                else:
                    cell_str = f"{cell_str:^{cell_length}s}"
                cell_msgs.append(cell_str)
            body_msgs.append(
                f"|{method:^{method_length}s}|" + "|".join(cell_msgs) + "|"
            )
        msgs = [header_msg, sub_header_msg] + body_msgs
        length = len(body_msgs[0])
        single_split = "-" * length
        double_split = "=" * length
        main_msg = f"\n{single_split}\n".join(msgs)
        final_msg = f"{double_split}\n{main_msg}\n{double_split}"
        self.log_block_msg(final_msg, self.info_prefix, "Results", verbose_level)
        return final_msg

    def compare(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        scoring_function: Union[str, scoring_fn_type] = "default",
        verbose_level: int = 1,
        **kwargs,
    ) -> "Comparer":
        self.estimator_statistics: Dict[str, Dict[str, Dict[str, float]]] = {}
        for estimator in self.estimators.values():
            methods = {}
            for model_name, patterns in self.patterns.items():
                if isinstance(patterns, dict):
                    patterns = patterns.get(estimator.type)
                if patterns is None:
                    continue
                if not isinstance(patterns, list):
                    patterns = [patterns]
                invalid = False
                predict_methods = []
                requires_prob = estimator.requires_prob
                for pattern in patterns:
                    if pattern is None:
                        invalid = True
                        break
                    predict_methods.append(pattern.predict_method(requires_prob))
                    if predict_methods[-1] is None:
                        invalid = True
                        self.log_msg(
                            f"{estimator} requires probability predictions but "
                            f"{model_name} does not have probability predicting "
                            "method, skipping",
                            self.warning_prefix,
                            verbose_level,
                            logging.WARNING,
                        )
                        break
                if invalid:
                    continue
                methods[model_name] = predict_methods
            self.estimator_statistics[estimator.type] = estimator.estimate(
                x,
                y,
                methods,
                scoring_function=scoring_function,
                verbose_level=None if verbose_level is None else verbose_level + 5,
            )
        self.log_statistics(verbose_level, **kwargs)
        return self

    def select(self, method_names: List[str]) -> "Comparer":
        new_patterns = {}
        for name in method_names:
            pattern = self.patterns.get(name)
            if pattern is None:
                raise ValueError(f"'{name}' is not found in current patterns")
            new_patterns[name] = pattern
        new_estimators = [
            estimator.select(method_names) for estimator in self.estimators.values()
        ]
        return Comparer(
            new_patterns,
            new_estimators,
            verbose_level=self._verbose_level,
        )

    @classmethod
    def merge(
        cls,
        comparer_list: List["Comparer"],
        scoring_function: Union[str, scoring_fn_type] = "default",
    ) -> "Comparer":
        first_comparer = comparer_list[0]
        new_estimators = [
            Estimator.merge([comparer.estimators[k] for comparer in comparer_list])
            for k in first_comparer.estimators.keys()
        ]
        new_comparer = cls(
            first_comparer.patterns,
            new_estimators,
            verbose_level=first_comparer._verbose_level,
        )
        new_comparer.estimator_statistics = {
            estimator.type: estimator.get_statistics(scoring_function).data
            for estimator in new_estimators
        }
        return new_comparer


class ScalarEMA:
    """
    Util class to record Exponential Moving Average (EMA) for scalar value.

    Parameters
    ----------
    decay : float, decay rate for EMA.
    * new = (1 - decay) * current + decay * history; history = new

    Examples
    --------
    >>> from cfml.misc.toolkit import ScalarEMA
    >>>
    >>> ema = ScalarEMA(0.5)
    >>> for i in range(4):
    >>>     print(ema.update("score", 0.5 ** i))  # 1, 0.75, 0.5, 0.3125

    """

    def __init__(self, decay):
        self._decay = decay
        self._ema_records = {}

    def __str__(self):
        return f"ScalarEMA({self._decay})"

    __repr__ = __str__

    def get(self, name):
        return self._ema_records.get(name)

    def update(self, name, new_value):
        history = self._ema_records.get(name)
        if history is None:
            updated = new_value
        else:
            updated = (1 - self._decay) * new_value + self._decay * history
        self._ema_records[name] = updated
        return updated


class Visualizer:
    """
    Visualization class.

    Methods
    ----------
    bar(self, data, classes, categories, save_name="bar_plot", title="",
            padding=1e-3, expand_floor=5, replace=True)
        Make bar plot with given `data`.
        * data : np.ndarray, containing values for the bar plot, where data.shape =
            * (len(categories), ), if len(classes) == 1.
            * (len(classes), len(categories)), otherwise.
        * classes : list(str), list of str which indicates each class.
            * each class will has its own color.
            * len(classes) indicates how many bars are there in one category
              (side by side).
        * categories : list(str), list of str which indicates each category.
            * a category will be a tick along x-axis.
        * save_name : str, saving name of this bar plot.
        * title : str, title of this bar plot.
        * padding : float, minimum value of each bar.
        * expand_floor : int, when len(categories) > `expand_floor`, the width of
          the figure will expand.
            * for len(classes) == 1, `expand_floor` will be multiplied by 2 internally.
        * overwrite : bool
            whether overwrite the existing file with the same file name of this
            plot's saving name.

    function(self, f, x_min, x_max, classes, categories, save_names=None,
             n_sample=1000, expand_floor=5, overwrite=True):
        Make multiple (len(categories)) line plots with given function (`f`)
        * f : function
            * input should be an np.ndarray with shape == (n, n_categories).
            * output should be an np.ndarray with
              shape == (n, n_categories, n_categories).
        * x_min : np.ndarray, minimum x-values for each line plot.
            * len(x_min) should be len(categories).
        * x_max : np.ndarray, maximum x-values for each line plot.
            * len(x_max) should be len(categories).
        * classes : list(str), list of str which indicates each class.
            * each class will has its own color.
            * len(classes) indicates how many bars are there in one category
              (side by side).
        * categories : list(str), list of str which indicates each category.
            * every category will correspond to a line plot.
        * save_names : list(str), saving names of these line plots.
        * n_sample : int, sample density along x-axis.
        * expand_floor : int, the width of the figures will be expanded with
          ratios calculated by:
            expand_ratios = np.maximum(
                1.0,
                np.abs(x_min) / expand_floor, x_max / expand_floor,
            )
        * overwrite : bool
            whether overwrite the existing file with the same file name of
            this plot's saving name.

    """

    def __init__(self, export_folder):
        self.export_folder = os.path.abspath(export_folder)
        os.makedirs(self.export_folder, exist_ok=True)

    def _get_save_name(self, save_name):
        counter, tmp_save_name = 0, save_name
        while os.path.isfile(os.path.join(self.export_folder, f"{tmp_save_name}.png")):
            counter += 1
            tmp_save_name = f"{save_name}_{counter}"
        return tmp_save_name

    def bar(
        self,
        data: np.ndarray,
        classes: List[Union[str, Any]],
        categories: List[Union[str, Any]],
        *,
        title: str = "",
        save_name: str = "bar_plot",
        padding: float = 1e-3,
        expand_floor: int = 5,
        overwrite: bool = True,
    ):
        num_classes, num_categories = map(len, [classes, categories])
        data = (
            [data / data.sum() + padding]
            if num_classes == 1
            else data / data.sum(0) + padding
        )
        expand_floor = expand_floor * 2 if num_classes == 1 else expand_floor
        colors = plt.cm.Paired([i / num_classes for i in range(num_classes)])
        x_base = np.arange(1, num_categories + 1)
        expand_ratio = max(1.0, num_categories / expand_floor)
        fig = plt.figure(figsize=(6.4 * expand_ratio, 4.8))
        plt.title(title)
        n_divide = num_classes - 1
        width = 0.35 / max(1, n_divide)
        cls_ratio = 0.5 if num_classes == 1 else 1
        for cls in range(num_classes):
            plt.bar(
                x_base - width * (0.5 * n_divide - cls_ratio * cls),
                data[cls],
                width=width,
                facecolor=colors[cls],
                edgecolor="white",
                label=classes[cls],
            )
        plt.xticks([i for i in range(len(categories) + 2)], [""] + categories + [""])
        plt.legend()
        plt.setp(plt.xticks()[1], rotation=30, horizontalalignment="right")
        plt.ylim(0, 1.2 + padding)
        fig.tight_layout()
        if not overwrite:
            save_name = self._get_save_name(save_name)
        plt.savefig(os.path.join(self.export_folder, f"{save_name}.png"))
        plt.close()

    def function(
        self,
        f: Callable[[np.ndarray], np.ndarray],
        x_min: np.ndarray,
        x_max: np.ndarray,
        classes: List[Union[str, Any]],
        categories: List[Union[str, Any]],
        *,
        save_names: Optional[List[str]] = None,
        num_sample: int = 1000,
        expand_floor: int = 5,
        overwrite: bool = True,
    ):
        num_classes, num_categories = map(len, [classes, categories])
        gaps = x_max - x_min
        x_base = np.linspace(x_min - 0.1 * gaps, x_max + 0.1 * gaps, num_sample)
        f_values = np.split(f(x_base), num_classes, axis=1)
        if save_names is None:
            save_names = ["function_plot"] * num_categories
        colors = plt.cm.Paired([i / num_classes for i in range(num_classes)])
        expand_ratios = np.maximum(
            1.0, np.abs(x_min) / expand_floor, x_max / expand_floor
        )
        for i, (category, save_name, ratio, local_min, local_max, gap) in enumerate(
            zip(categories, save_names, expand_ratios, x_min, x_max, gaps)
        ):
            plt.figure(figsize=(6.4 * ratio, 4.8))
            plt.title(f"pdf for {category}")
            local_base = x_base[..., i]
            for c in range(num_classes):
                f_value = f_values[c][..., i].ravel()
                plt.plot(
                    local_base,
                    f_value,
                    c=colors[c],
                    label=f"class: {classes[c]}",
                )
            plt.xlim(local_min - 0.2 * gap, local_max + 0.2 * gap)
            plt.legend()
            if not overwrite:
                save_name = self._get_save_name(save_name)
            plt.savefig(os.path.join(self.export_folder, f"{save_name}.png"))

    @staticmethod
    def visualize1d(
        method: Callable,
        x: np.ndarray,
        y: Optional[np.ndarray] = None,
        *,
        title: Optional[str] = None,
        num_samples: int = 100,
        expand_ratio: float = 0.25,
        return_canvas: bool = False,
    ) -> Optional[np.ndarray]:
        if x.shape[1] != 1:
            raise ValueError("visualize1d only supports 1-dimensional features")
        plt.figure()
        plt.title(title)
        if y is not None:
            plt.scatter(x, y, c="g", s=20)
        x_min, x_max = x.min(), x.max()
        expand = expand_ratio * (x_max - x_min)
        x0 = np.linspace(x_min - expand, x_max + expand, num_samples).reshape([-1, 1])
        plt.plot(x0, method(x0).ravel())
        return show_or_return(return_canvas)

    @staticmethod
    def visualize2d(
        method,
        x: np.ndarray,
        y: Optional[np.ndarray] = None,
        *,
        title: Optional[str] = None,
        dense: int = 200,
        padding: float = 0.1,
        return_canvas: bool = False,
        draw_background: bool = True,
        extra_scatters: Optional[np.ndarray] = None,
        emphasize_indices: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        axis = x.T
        if axis.shape[0] != 2:
            raise ValueError("visualize2d only supports 2-dimensional features")
        nx, ny, padding = dense, dense, padding
        x_min, x_max = np.min(axis[0]), np.max(axis[0])
        y_min, y_max = np.min(axis[1]), np.max(axis[1])
        x_padding = max(abs(x_min), abs(x_max)) * padding
        y_padding = max(abs(y_min), abs(y_max)) * padding
        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding

        def get_base(_nx, _ny):
            _xf = np.linspace(x_min, x_max, _nx)
            _yf = np.linspace(y_min, y_max, _ny)
            n_xf, n_yf = np.meshgrid(_xf, _yf)
            return _xf, _yf, np.c_[n_xf.ravel(), n_yf.ravel()]

        xf, yf, base_matrix = get_base(nx, ny)
        z = method(base_matrix).reshape((nx, ny))

        labels = y.ravel()
        num_labels = y.max().item() + 1
        colors = plt.cm.rainbow([i / num_labels for i in range(num_labels)])[labels]

        plt.figure()
        plt.title(title)
        if draw_background:
            xy_xf, xy_yf = np.meshgrid(xf, yf, sparse=True)
            plt.pcolormesh(xy_xf, xy_yf, z, cmap=plt.cm.Pastel1)
        else:
            plt.contour(xf, yf, z, c="k-", levels=[0])
        plt.scatter(axis[0], axis[1], c=colors)

        if emphasize_indices is not None:
            indices = np.array([False] * len(axis[0]))
            indices[np.asarray(emphasize_indices)] = True
            plt.scatter(
                axis[0][indices],
                axis[1][indices],
                s=80,
                facecolors="None",
                zorder=10,
            )
        if extra_scatters is not None:
            plt.scatter(
                *np.asarray(extra_scatters).T,
                s=80,
                zorder=25,
                facecolors="red",
            )

        return show_or_return(return_canvas)


class Tracker:
    """
    Util class to track records in an experiment.
    * Currently only scalars are supported.

    Parameters
    ----------
    project_name : {str, None}, the project name of the experiment.
    * If None, then `Tracker.default_project_name()` will be used.
    task_name : {str, None}, the task name of the experiment.
    * If None, then `timestamp(ensure_different=True)` will be used.
    base_folder : {str, None}, where the records will be stored.
    * If None, then `Tracker.default_base_folder()` will be used.
    overwrite : bool, whether overwrite the existing records.
    * If False (which is by default), `Tracker` will load the existing records.

    Examples
    --------
    >>> from cfml.misc.toolkit import Tracker
    >>>
    >>> tracker = Tracker()
    >>> tracker.track_scalar("acc", 0.34)
    >>> tracker.track_scalar("acc", 0.67)
    >>> tracker.visualize_scalars()

    """

    def __init__(
        self,
        project_name: Optional[str] = None,
        task_name: Optional[str] = None,
        *,
        base_folder: Optional[str] = None,
        overwrite: bool = False,
    ):
        if base_folder is None:
            base_folder = self.default_base_folder()
        if project_name is None:
            project_name = self.default_project_name()
        if task_name is None:
            task_name = timestamp(ensure_different=True)
        self.base_folder = base_folder
        self.project_name = project_name
        self.task_name = task_name
        self.project_folder = os.path.join(base_folder, project_name)
        self.log_folder = os.path.join(self.project_folder, task_name)
        exists = os.path.isdir(self.log_folder)
        if exists:
            if not overwrite:
                print(
                    f"{LoggingMixin.info_prefix}loading tracker "
                    f"from '{self.log_folder}'"
                )
                self._load()
            else:
                print(
                    f"{LoggingMixin.warning_prefix}'{self.log_folder}' already exists,"
                    " it will be overwritten"
                )
                self.clear(confirm=False)
        if not exists or overwrite:
            os.makedirs(self.log_folder)
            self.reset()

    def __str__(self):
        return f"Tracker(project={self.project_name}, task={self.task_name})"

    __repr__ = __str__

    # core

    @property
    def scalars_folder(self) -> str:
        folder = os.path.join(self.log_folder, "scalars")
        os.makedirs(folder, exist_ok=True)
        return folder

    @property
    def messages_folder(self) -> str:
        folder = os.path.join(self.log_folder, "messages")
        os.makedirs(folder, exist_ok=True)
        return folder

    @staticmethod
    def default_base_folder() -> str:
        home = os.path.expanduser("~")
        return os.path.join(home, ".carefree-toolkit", ".tracker")

    @staticmethod
    def default_project_name() -> str:
        cwd = os.getcwd()
        return f"{hash_code(cwd)}_{os.path.split(cwd)[-1]}"

    def _load(self) -> None:
        self._load_scalars()
        self._load_messages()

    def _load_scalars(self) -> None:
        self.scalars = {}
        for file in os.listdir(self.scalars_folder):
            data = []
            name = os.path.splitext(file)[0]
            with open(os.path.join(self.scalars_folder, file), "r") as f:
                for line in f:
                    iteration, value = line.strip().split()
                    data.append((int(iteration), float(value)))
            self.scalars[name] = data

    def _load_messages(self) -> None:
        self.messages = {}
        for file in os.listdir(self.messages_folder):
            name = os.path.splitext(file)[0]
            with open(os.path.join(self.messages_folder, file), "r") as f:
                self.messages[name] = f.read()

    # api

    def reset(self) -> None:
        self.messages: Dict[str, str] = {}
        self.scalars: Dict[str, List[Tuple[int, float]]] = {}

    def track_scalar(self, name: str, value: float, *, iteration: int = None) -> None:
        file = os.path.join(self.scalars_folder, f"{name}.txt")
        data = self.scalars.setdefault(name, [])
        if iteration is None:
            iteration = 0 if not data else data[-1][0] + 1
        data.append((iteration, value))
        if not os.path.isfile(file):
            with open(file, "w") as f:
                f.write("\n".join(map(lambda line: " ".join(map(str, line)), data)))
        else:
            with open(file, "a") as f:
                f.write(f"\n{iteration} {value}")

    def track_message(self, name: str, message: str, *, append: bool = True) -> None:
        file = os.path.join(self.messages_folder, f"{name}.txt")
        existing_message = self.messages.get(name, "")
        if not existing_message or not append:
            self.messages[name] = message
            with open(file, "w") as f:
                f.write(message)
            return
        self.messages[name] = f"{existing_message}\n{message}"
        with open(file, "a") as f:
            f.write(f"\n{message}")

    def visualize_scalars(
        self,
        types: Optional[List[str]] = None,
        *,
        export_folder: Optional[str] = None,
        merge: bool = True,
    ) -> None:
        if export_folder is not None:
            os.makedirs(export_folder, exist_ok=True)
        if types is None:
            types = list(self.scalars.keys())
        plt.figure()
        for i, name in enumerate(sorted(types)):
            data = self.scalars[name]
            iterations, values = map(np.array, map(list, zip(*data)))
            mask = iterations >= 0
            plt.plot(iterations[mask], values[mask], label=name)
            if not merge:
                plt.legend()
                export_path = (
                    None
                    if export_folder is None
                    else os.path.join(export_folder, f"{name}.png")
                )
                show_or_save(export_path)
                if i != len(types) - 1:
                    plt.figure()
        if merge:
            plt.legend()
            export_path = (
                None
                if export_folder is None
                else os.path.join(export_folder, "merged.png")
            )
            show_or_save(export_path)

    # clear

    @staticmethod
    def _confirm(confirm: bool) -> bool:
        if not confirm:
            return True
        if input("[Y/n] (default : n)").lower() != "y":
            print("canceled")
            return False
        return True

    def clear(self, *, confirm: bool = True) -> None:
        print(f"{LoggingMixin.info_prefix}clearing '{self.log_folder}'")
        if self._confirm(confirm):
            shutil.rmtree(self.log_folder)

    def clear_project(self, *, confirm: bool = True) -> None:
        print(f"{LoggingMixin.info_prefix}clearing '{self.project_folder}'")
        if self._confirm(confirm):
            shutil.rmtree(self.project_folder)

    def clear_all(self, *, confirm: bool = True) -> None:
        print(f"{LoggingMixin.info_prefix}clearing '{self.base_folder}'")
        if self._confirm(confirm):
            shutil.rmtree(self.base_folder)

    # class methods

    @classmethod
    def compare(
        cls,
        project_name: Optional[str] = None,
        task_names: Optional[List[str]] = None,
        *,
        base_folder: Optional[str] = None,
        visualize: bool = True,
        types: Optional[List[str]] = None,
        export_folder: Optional[str] = None,
        merge: bool = False,
    ) -> List["Tracker"]:
        if base_folder is None:
            base_folder = Tracker.default_base_folder()
        if project_name is None:
            project_name = Tracker.default_project_name()
        project_folder = os.path.join(base_folder, project_name)
        if task_names is None:
            task_names = os.listdir(project_folder)
        else:
            for task_name in task_names:
                task_folder = os.path.join(project_folder, task_name)
                if not os.path.isdir(task_folder):
                    raise ValueError(f"'{task_folder}' does not exist")
        trackers = [
            Tracker(project_name, task_name, base_folder=base_folder)
            for task_name in task_names
        ]
        if visualize:
            plt.figure()
            if types is None:
                types = set()
                for tracker in trackers:
                    types |= set(tracker.scalars.keys())
                types = list(types)
            for i, name in enumerate(sorted(types)):
                for task_name, tracker in zip(task_names, trackers):
                    data = tracker.scalars.get(name)
                    if data is not None:
                        iterations, values = map(np.array, map(list, zip(*data)))
                        mask = iterations >= 0
                        plt.plot(
                            iterations[mask],
                            values[mask],
                            label=f"{name} - {task_name}",
                        )
                if not merge:
                    plt.legend()
                    export_path = (
                        None
                        if export_folder is None
                        else os.path.join(export_folder, f"{name}.png")
                    )
                    show_or_save(export_path)
                    if i != len(types) - 1:
                        plt.figure()
            if merge:
                plt.legend()
                export_path = (
                    None
                    if export_folder is None
                    else os.path.join(export_folder, "merged.png")
                )
                show_or_save(export_path)
        return trackers


def make_cfml_pattern(cfml_model) -> ModelPattern:
    return ModelPattern(
        init_method=lambda: cfml_model,
        predict_method="predict",
        predict_prob_method="predict_prob",
    )


def make_sklearn_pattern(sk_model) -> Union[ModelPattern, None]:
    predict_prob_method = getattr(sk_model, "predict_proba", None)
    predict_method = lambda x: sk_model.predict(x).reshape([-1, 1])
    return ModelPattern(
        predict_method=predict_method,
        predict_prob_method=predict_prob_method,
    )


class SklearnComparer:
    def __init__(
        self,
        cfml_models: Dict[str, Any],
        sklearn_models: Dict[str, Any],
        *,
        task_type: TaskTypes = TaskTypes.CLASSIFICATION,
    ):
        patterns = {}
        models_bundle = [cfml_models, sklearn_models]
        make_functions = [make_cfml_pattern, make_sklearn_pattern]
        if task_type is TaskTypes.REGRESSION:
            metrics = ["mae", "mse"]
            for models, make_function in zip(models_bundle, make_functions):
                new_patterns = {k: make_function(v) for k, v in models.items()}
                new_patterns = {k: v for k, v in new_patterns.items() if v is not None}
                patterns.update(new_patterns)
        else:
            metrics = ["auc", "acc"]
            for models, make_function in zip(models_bundle, make_functions):
                for metric in metrics:
                    for model_name, model in models.items():
                        local_patterns = patterns.setdefault(model_name, {})
                        new_pattern = make_function(model)
                        if new_pattern is not None:
                            local_patterns[metric] = new_pattern
        self._core = Comparer(patterns, list(map(Estimator, metrics)))

    def compare(self, x: np.ndarray, y: np.ndarray):
        self._core.compare(x, y)


class Experiment:
    def __init__(
        self,
        cfml_models: Dict[str, Any],
        sklearn_models: Dict[str, Any],
        *,
        task_type: TaskTypes = TaskTypes.CLASSIFICATION,
        show_images: bool = False,
    ):
        self._show_images = show_images
        self._cfml_models = cfml_models
        self._sklearn_models = sklearn_models
        self._comparer = SklearnComparer(
            cfml_models,
            sklearn_models,
            task_type=task_type,
        )

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

    def visualize(self, x_min: float = -5.0, x_max: float = 5.0):
        plt.figure()
        x0 = np.linspace(x_min, x_max)
        plt.plot(x0, self(x0))
        plt.show()

    @staticmethod
    def relu(x):
        return np.maximum(0.0, x)

    @staticmethod
    def relu_grad(forward):
        return (forward != 0.0).astype(np.float32)

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def sigmoid_grad(forward):
        return forward * (1.0 - forward)

    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    @staticmethod
    def softmax_grad(forward):
        return forward * (1.0 - forward)


class DataInspector:
    def __init__(self, data: Union[np.ndarray, Any]):
        self._data = np.asarray(data, np.float32)
        self._sorted_data = np.sort(self._data)
        self._num_samples = len(self._data)
        self._mean = self._variance = self._std = None
        self._moments = []
        self._q1 = self._q3 = self._median = None

    def get_moment(self, k: int) -> float:
        if len(self._moments) < k:
            self._moments += [None] * (k - len(self._moments))
        if self._moments[k - 1] is None:
            self._moments[k - 1] = (
                np.sum((self._data - self.mean) ** k) / self._num_samples
            )
        return self._moments[k - 1]

    def get_quantile(self, q: float) -> float:
        if not 0.0 <= q <= 1.0:
            raise ValueError("`q` should be in [0, 1]")
        anchor = self._num_samples * q
        int_anchor = int(anchor)
        if not int(anchor % 1):
            return self._sorted_data[int_anchor]
        dq = self._sorted_data[int_anchor - 1] + self._sorted_data[int_anchor]
        return 0.5 * dq

    @property
    def min(self) -> float:
        return self._sorted_data[0]

    @property
    def max(self) -> float:
        return self._sorted_data[-1]

    @property
    def mean(self) -> float:
        if self._mean is None:
            self._mean = self._data.mean()
        return self._mean

    @property
    def variance(self) -> float:
        if self._variance is None:
            square_sum = np.sum((self._data - self.mean) ** 2)
            self._variance = square_sum / (self._num_samples - 1)
        return self._variance

    @property
    def std(self) -> float:
        if self._std is None:
            self._std = self.variance**0.5
        return self._std

    @property
    def skewness(self) -> float:
        n, moment3 = self._num_samples, self.get_moment(3)
        return n**2 * moment3 / ((n - 1) * (n - 2) * self.std**3)

    @property
    def kurtosis(self) -> float:
        n, moment4 = self._num_samples, self.get_moment(4)
        return n**2 * (n + 1) * moment4 / (
            (n - 1) * (n - 2) * (n - 3) * self.std**4
        ) - 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))

    @property
    def median(self) -> float:
        if self._median is None:
            self._median = self.get_quantile(0.5)
        return self._median

    @property
    def q1(self) -> float:
        if self._q1 is None:
            self._q1 = self.get_quantile(0.25)
        return self._q1

    @property
    def q3(self) -> float:
        if self._q3 is None:
            self._q3 = self.get_quantile(0.75)
        return self._q3

    @property
    def range(self) -> float:
        return self._sorted_data[-1] - self._sorted_data[0]

    @property
    def iqr(self) -> float:
        return self.q3 - self.q1

    @property
    def trimean(self) -> float:
        return 0.25 * (self.q1 + self.q3) + 0.5 * self.median

    @property
    def lower_cutoff(self) -> float:
        return self.q1 - 1.5 * self.iqr

    @property
    def upper_cutoff(self) -> float:
        return self.q3 + 1.5 * self.iqr

    def draw_histogram(
        self,
        bin_size: int = 10,
        export_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        bins = np.arange(
            self._sorted_data[0] - self.iqr,
            self._sorted_data[-1] + self.iqr,
            bin_size,
        )
        plt.hist(self._data, bins=bins, alpha=0.5)
        plt.title(f"Histogram (bin_size: {bin_size})")
        show_or_save(export_path, **kwargs)

    def qq_plot(self, export_path: Optional[str] = None, **kwargs) -> None:
        ss.probplot(self._data, dist="norm", plot=plt)
        show_or_save(export_path, **kwargs)

    def box_plot(self, export_path: Optional[str] = None, **kwargs) -> None:
        plt.figure()
        plt.boxplot(self._data, vert=False, showmeans=True)
        show_or_save(export_path, **kwargs)
