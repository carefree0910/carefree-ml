import math
import pprint
import random
import logging

import numpy as np

from typing import *
from cftool.misc import *
from abc import abstractmethod
from abc import ABCMeta
from tqdm import tqdm
from functools import partial
from collections import defaultdict
from cftool.dist import Parallel

from ..toolkit import *
from ..param_utils import *
from ...models.bases import Base


hpo_dict: Dict[str, Type["HPOBase"]] = {}
created_type = Union[pattern_type, Any]
creator_type = Callable[[np.ndarray, np.ndarray, Dict[str, Any]], created_type]


class HPOBase(LoggingMixin, metaclass=ABCMeta):
    def __init__(
        self,
        creator: creator_type,
        params: Dict[str, DataType],
        *,
        converter: Optional[Callable[[List[created_type]], patterns_type]] = None,
        verbose_level: int = 2,
        **kwargs,
    ):
        self._caches = {}
        self._init_config(**kwargs)
        self._creator = creator
        self._converter = converter
        self.param_generator = ParamsGenerator(params)
        self._verbose_level = verbose_level

    @property
    @abstractmethod
    def is_sequential(self) -> bool:
        pass

    @property
    def last_param(self) -> nested_type:
        return self.param_mapping[self.last_code]

    @property
    def last_patterns(self) -> List[pattern_type]:
        return self.patterns[self.last_code]

    def _init_config(self, **kwargs):
        pass

    def _sample_param(self) -> Union[None, nested_type]:
        if self.is_sequential:
            raise NotImplementedError
        return

    def _get_scores(self, patterns: List[pattern_type]) -> Dict[str, float]:
        key = "core"
        comparer = Comparer({key: patterns}, self.estimators)
        final_scores = comparer.compare(
            self.x_validation,
            self.y_validation,
            scoring_function=self._estimator_scoring_function,
            verbose_level=6,
        ).final_scores
        return {metric: scores[key] for metric, scores in final_scores.items()}

    def _core(
        self,
        param: nested_type,
        *,
        convert: bool = True,
        parallel_run: bool = False,
    ) -> List[created_type]:
        range_list = list(range(self._num_retry))
        _task = lambda _=0: self._creator(self.x_train, self.y_train, param)
        tqdm_config = {"position": 1, "leave": False}
        if not parallel_run:
            if self._use_tqdm and len(range_list) > 1:
                range_list = tqdm(range_list, **tqdm_config)
            created = [_task() for _ in range_list]
        else:
            parallel = Parallel(
                self._num_jobs,
                use_tqdm=self._use_tqdm,
                tqdm_config=tqdm_config,
            )
            created = parallel(_task, range_list).ordered_results
        if not convert:
            return created
        patterns = created if self._converter is None else self._converter(created)
        return patterns

    def search(
        self,
        x: generic_data_type,
        y: generic_data_type,
        estimators: List[Estimator],
        x_validation: Optional[generic_data_type] = None,
        y_validation: Optional[generic_data_type] = None,
        *,
        num_jobs: int = 4,
        num_retry: int = 4,
        num_search: Union[str, int, float] = 10,
        score_weights: Optional[Dict[str, float]] = None,
        estimator_scoring_function: Union[str, scoring_fn_type] = "default",
        use_tqdm: bool = True,
        verbose_level: int = 2,
        **kwargs,
    ) -> "HPOBase":

        if x_validation is None and y_validation is None:
            x_validation, y_validation = x, y

        self.estimators = estimators
        self.x_train, self.y_train = x, y
        self.x_validation, self.y_validation = x_validation, y_validation

        num_params = self.param_generator.num_params
        if isinstance(num_search, str):
            if num_search != "all":
                raise ValueError(
                    "num_search can only be 'all' when it is a string, "
                    f"'{num_search}' found"
                )
            if num_params == math.inf:
                raise ValueError(
                    "num_search is 'all' but we have infinite params to search"
                )
            num_search = num_params
        if num_search > num_params:
            self.log_msg(
                f"`n` is larger than total choices we've got ({num_params}), "
                f"therefore only {num_params} searches will be run",
                self.warning_prefix,
                msg_level=logging.WARNING,
            )
            num_search = num_params
        num_jobs = min(num_search, num_jobs)

        self._use_tqdm = use_tqdm
        if score_weights is None:
            score_weights = {estimator.type: 1.0 for estimator in estimators}
        self._score_weights = score_weights
        self._estimator_scoring_function = estimator_scoring_function
        self._num_retry, self._num_jobs = num_retry, num_jobs

        with timeit("Generating Patterns"):
            if self.is_sequential:
                self.patterns, self.param_mapping = {}, {}
                iterator = list(range(num_search))
                if use_tqdm:
                    iterator = tqdm(iterator, position=0)
                for _ in iterator:
                    param = self._sample_param()
                    self.last_code = hash_code(str(param))
                    self.param_mapping[self.last_code] = param
                    self.patterns[self.last_code] = self._core(
                        param, convert=True, parallel_run=num_jobs > 1
                    )
            else:
                if num_params == math.inf:
                    all_params = [self.param_generator.pop() for _ in range(num_search)]
                else:
                    all_params = []
                    sampled = random.sample(list(range(num_search)), k=num_search)
                    all_indices = set(sampled)
                    for i, param in enumerate(self.param_generator.all()):
                        if i in all_indices:
                            all_params.append(param)
                        if len(all_params) == num_search:
                            break

                codes = list(map(hash_code, map(str, all_params)))
                self.param_mapping = dict(zip(codes, all_params))
                if num_jobs <= 1:
                    if self._use_tqdm:
                        all_params = tqdm(all_params)
                    patterns = list(map(self._core, all_params))
                else:
                    logging_folder = kwargs.get("parallel_logging_folder")
                    parallel = Parallel(num_jobs, logging_folder=logging_folder)
                    core = partial(self._core, convert=False)
                    created_list = parallel(core, all_params).ordered_results
                    patterns = list(map(self._converter, created_list))
                self.patterns = dict(zip(codes, patterns))
                self.last_code = codes[-1]

        self.comparer = Comparer(self.patterns, estimators)
        self.comparer.compare(
            x_validation,
            y_validation,
            scoring_function=estimator_scoring_function,
            verbose_level=verbose_level,
        )

        weighted_scores = defaultdict(float)
        for metric, scores in self.comparer.final_scores.items():
            for method, score in scores.items():
                weighted_scores[method] += self._score_weights[metric] * score
        sorted_methods = sorted(weighted_scores)
        sorted_methods_scores = [weighted_scores[key] for key in sorted_methods]
        best_method = sorted_methods[np.argmax(sorted_methods_scores).item()]
        self.best_param = self.param_mapping[best_method]

        best_methods = self.comparer.best_methods
        self.best_params = {k: self.param_mapping[v] for k, v in best_methods.items()}
        param_msgs = {k: pprint.pformat(v) for k, v in self.best_params.items()}
        estimator_statistics = self.comparer.estimator_statistics
        sorted_metrics = sorted(param_msgs)
        target_statistics = []
        for metric in sorted_metrics:
            metric_method = best_methods[metric]
            metric_statistics = estimator_statistics[metric][metric_method]
            target_statistics.append(
                {
                    "method": metric_method,
                    "mean": fix_float_to_length(metric_statistics["mean"], 8),
                    "std": fix_float_to_length(metric_statistics["std"], 8),
                }
            )
        msg = "\n".join(
            sum(
                [
                    [
                        "-" * 100,
                        f"{metric}  ({stat['method']}) "
                        f"({stat['mean']} ± {stat['std']})",
                        "-" * 100,
                        param_msgs[metric],
                    ]
                    for metric, stat in zip(sorted_metrics, target_statistics)
                ],
                [],
            )
            + [
                "-" * 100,
                f"best ({best_method})",
                "-" * 100,
                pprint.pformat(self.best_param),
            ]
            + ["-" * 100]
        )
        self.log_block_msg(msg, self.info_prefix, "Best Parameters", verbose_level)

        return self

    @staticmethod
    def make(method: str, *args, **kwargs) -> "HPOBase":
        return hpo_dict[method](*args, **kwargs)

    @classmethod
    def register(cls, name):
        global hpo_dict
        return register_core(name, hpo_dict)


def HPO(
    model: str,
    params: Dict[str, DataType],
    *,
    hpo_method: str = "naive",
    verbose_level: int = 2,
):
    def _creator(x, y, params_):
        m = Base.make(model, **params_)
        m.show_tqdm = False
        return ModelPattern(init_method=lambda: m.fit(x, y))

    return HPOBase.make(hpo_method, _creator, params, verbose_level=verbose_level)


__all__ = [
    "HPOBase",
    "hpo_dict",
    "HPO",
]
