from typing import *

from ..base import HPOBase
from ...optim import BayesianOptimization
from ...toolkit import Metrics


@HPOBase.register("bo")
class BayesianHPO(HPOBase):
    @property
    def is_sequential(self) -> bool:
        return True

    def _init_config(self, **kwargs):
        self._bo_config = kwargs.get("bo_config", {})
        bo_normalization = self._bo_config.setdefault("normalization", "cube")
        if bo_normalization == "cube":
            bo_norm_cfg = self._bo_config.setdefault("normalization_config", {})
            bo_norm_cfg.setdefault("convert_only", False)
        self._num_iter = kwargs.get("num_iter", 10)
        self._num_warmup = kwargs.get("num_warmup", 10000)
        self._init_points = kwargs.get("init_points", 5)
        if self._init_points <= 1:
            msg = f"init_points should larger than 1, {self._init_points} found"
            raise ValueError(msg)
        self._bo_core, self._iteration = None, 0

    def _score(self, final_scores: Dict[str, float]) -> float:
        return sum(
            [
                self._score_weights.setdefault(k, 1.0) * v * Metrics.sign_dict[k]
                for k, v in final_scores.items()
            ]
        )

    def _sample_param(self) -> Union[None, Dict[str, Any]]:
        self._iteration += 1
        if self._bo_core is None:
            params = self.param_generator.params
            self._bo_core = BayesianOptimization(None, params, **self._bo_config)
        if self._iteration <= self._init_points:
            return self.param_generator.pop()
        if not self._bo_core.space.is_empty:
            nested = self.last_param
            flattened = self.param_generator.flatten_nested(nested)
            self._bo_core.register(
                flattened,
                self._score(self._get_scores(self.last_patterns)),
            )
        else:
            for code, params in self.param_mapping.items():
                patterns = self.patterns[code]
                flattened = self.param_generator.flatten_nested(params)
                self._bo_core.register(
                    flattened,
                    self._score(self._get_scores(patterns)),
                )
        flattened = self._bo_core.suggest(self._num_warmup, self._num_iter)
        return self.param_generator.nest_flattened(flattened)


__all__ = ["BayesianHPO"]
