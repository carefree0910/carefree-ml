import numpy as np

from typing import *
from cftool.misc import *

from ...param_utils import *


fn_type = Optional[Callable[[nested_type], float]]


class Result(NamedTuple):
    params: nested_type
    score: float


class TargetSpace:
    def __init__(
        self,
        fn: fn_type,
        params: params_type,
        *,
        normalization: Optional[str],
        normalization_config: Optional[Dict[str, Any]],
    ):
        self.fn = fn
        self.params_gen = ParamsGenerator(
            params,
            normalize_method=normalization,
            normalize_config=normalization_config,
        )
        self._codes2scores = {}
        self._sorted_keys = self.params_gen.sorted_flattened_keys
        self._tried_flattened_arrays = np.empty([0, self.dim], np.float32)
        self._tried_scores = np.empty([0], np.float32)

    @property
    def dim(self) -> int:
        return self.params_gen.array_dim

    @property
    def num_try(self) -> int:
        return len(self._codes2scores)

    @property
    def tried_flattened_params(self) -> np.ndarray:
        return self._tried_flattened_arrays

    @property
    def tried_scores(self) -> np.ndarray:
        return self._tried_scores

    @property
    def is_empty(self) -> bool:
        return not self._codes2scores

    @property
    def best_result(self) -> Result:
        best_idx = np.argmax(self._tried_scores).item()
        return self._make_result(best_idx)

    @property
    def all_results(self) -> List[Result]:
        return list(map(self._make_result, range(self.num_try)))

    def _make_result(self, idx: int) -> Result:
        array = self._tried_flattened_arrays[idx]
        flattened = self.array2param(array)
        nested = self.params_gen.nest_flattened(flattened)
        return Result(nested, self._tried_scores[idx])

    def param2array(self, param: flattened_type) -> np.ndarray:
        return self.params_gen.flattened2array(param)

    def array2param(self, array: np.ndarray) -> flattened_type:
        return self.params_gen.array2flattened(array)

    def register(self, param: flattened_type, score: float) -> "TargetSpace":
        self._codes2scores[hash_code(str(param))] = score
        param_array = self.param2array(param).reshape(1, -1)
        self._tried_flattened_arrays = np.concatenate(
            [self._tried_flattened_arrays, param_array]
        )
        self._tried_scores = np.concatenate([self._tried_scores, [score]])
        return self

    def probe(self, param: flattened_type) -> "TargetSpace":
        if self.fn is None:
            msg = "fn is not provided, so `probe` method should not be called"
            raise ValueError(msg)
        code = hash_code(str(param))
        score = self._codes2scores.get(code)
        if score is not None:
            return self
        nested = self.params_gen.nest_flattened(param)
        self.register(param, self.fn(nested))
        return self

    def sample(self) -> flattened_type:
        return self.params_gen.flatten_nested(self.params_gen.pop())


__all__ = ["Result", "TargetSpace", "fn_type"]
