import math
import numpy as np

from typing import Dict
from typing import List
from typing import Union
from typing import Iterator
from typing import Optional
from cftool.misc import *

from .types import *
from .data_types import *
from .normalizers import *
from .distributions import *


params_type = Dict[str, Union[DataType, Iterable, "params_type"]]


class ParamsGenerator:
    """
    Parameter generator for param searching, see cfml.misc.hpo.base.HPOBase for usage.

    Parameters
    ----------
    params : params_type, parameter settings.

    Examples
    ----------
    >>> grid = ParamsGenerator({
    >>>     "a": Any(Choice(values=[1, 2, 3])),
    >>>     "c": {
    >>>         "d": Int(Choice(values=[1, 2, 3])),
    >>>         "e": Float(Choice(values=[1, 2])),
    >>>     }
    >>> })
    >>> for param in grid.all():
    >>>     print(param)
    >>> # output : {'a': 1, 'c': {'d': 1, 'e': 1, 'f': 3}}, {'a': 1, 'c': {'d': 1, 'e': 1, 'f': 4}}
    >>> #          {'a': 1, 'c': {'d': 1, 'e': 2, 'f': 3}}, {'a': 1, 'c': {'d': 1, 'e': 2, 'f': 4}}
    >>> #          {'a': 1, 'c': {'d': 2, 'e': 1, 'f': 3}}, {'a': 1, 'c': {'d': 2, 'e': 1, 'f': 4}}
    >>> #          {'a': 1, 'c': {'d': 2, 'e': 2, 'f': 3}}, {'a': 1, 'c': {'d': 2, 'e': 2, 'f': 4}}
    >>> #          ......
    >>> #          {'a': 3, 'c': {'d': 3, 'e': 2, 'f': 3}}, {'a': 3, 'c': {'d': 3, 'e': 2, 'f': 4}}

    """

    def __init__(
        self,
        params: params_type,
        *,
        normalize_method: Optional[str] = None,
        normalize_config: Optional[Dict[str, Any]] = None,
    ):
        self._data_types = params

        def _data_type_offset(value: DataType) -> int:
            if not isinstance(value, Iterable):
                return 1
            return len(value.values)

        self._data_types_nested = Nested(params, offset_fn=_data_type_offset)

        if normalize_method is None:
            self._normalizers_flattened = None
        else:
            if normalize_config is None:
                normalize_config = {}

            def _data_type_normalizer(value: DataType) -> Normalizer:
                return Normalizer(normalize_method, value, **normalize_config)

            normalizers_nested = self._data_types_nested.apply(_data_type_normalizer)
            self._normalizers_flattened = normalizers_nested.flattened

        self._all_params_nested = self._all_flattened_data_types = None
        self._array_dim = self._all_bounds = None

    @property
    def params(self) -> params_type:
        return self._data_types

    @property
    def num_params(self) -> number_type:
        def _num_params(params):
            if isinstance(params, (DataType, Iterable)):
                return params.num_params
            assert isinstance(params, dict)
            num_params = prod(_num_params(v) for v in params.values())
            if math.isinf(num_params):
                return num_params
            return int(num_params)

        return _num_params(self._data_types)

    @property
    def array_dim(self) -> int:
        if self._array_dim is None:
            self._array_dim = self.flattened2array(
                self.flatten_nested(self.pop())
            ).shape[0]
        return self._array_dim

    @property
    def all_bounds(self) -> np.ndarray:
        if self._all_bounds is None:
            bounds_list = []
            for key in self.sorted_flattened_keys:
                if self._normalizers_flattened is None:
                    normalizer = None
                else:
                    normalizer = self._normalizers_flattened[key]
                if normalizer is None:
                    data_type = self._data_types_nested.get_value_from(key)
                    if not isinstance(data_type, Iterable):
                        bounds_list.append(list(data_type.bounds))
                    else:
                        bounds_list.extend(list(map(list, data_type.bounds)))
                else:
                    if normalizer.is_iterable:
                        bounds_list.extend(list(map(list, normalizer.bounds)))
                    else:
                        bounds_list.append(list(normalizer.bounds))
            self._all_bounds = np.array(bounds_list, np.float32)
        return self._all_bounds

    @property
    def all_flattened_params(self) -> all_flattened_type:
        if self._all_params_nested is None:
            apply = lambda data_type: data_type.all()
            self._all_params_nested = self._data_types_nested.apply(apply)
        return self._all_params_nested.flattened

    @property
    def sorted_flattened_keys(self) -> List[str]:
        return self._data_types_nested.sorted_flattened_keys

    def pop(self) -> nested_type:
        def _pop(src: dict, tgt: dict):
            for k, v in src.items():
                if isinstance(v, dict):
                    next_tgt = tgt.setdefault(k, {})
                    _pop(v, next_tgt)
                else:
                    tgt[k] = v.pop()
            return tgt

        return _pop(self._data_types, {})

    def all(self) -> Iterator[nested_type]:
        for flattened_params in Grid(self.all_flattened_params):
            yield self._data_types_nested.nest_flattened(flattened_params)

    def flatten_nested(self, nested: nested_type) -> nested_type:
        return self._data_types_nested.flatten_nested(nested)

    def nest_flattened(self, flattened: flattened_type) -> nested_type:
        return self._data_types_nested.nest_flattened(flattened)

    def flattened2array(self, flattened: flattened_type) -> np.ndarray:
        if self._normalizers_flattened is None:
            normalized = flattened
        else:
            normalized = {
                k: self._normalizers_flattened[k].normalize(v)
                for k, v in flattened.items()
            }
        return self._data_types_nested.flattened2array(normalized)

    def array2flattened(self, array: np.ndarray) -> flattened_type:
        normalized = self._data_types_nested.array2flattened(array)
        if self._normalizers_flattened is None:
            flattened = normalized
        else:
            flattened = {
                k: self._normalizers_flattened[k].recover(v)
                for k, v in normalized.items()
            }
        for key, value in flattened.items():
            data_type = self._data_types_nested.get_value_from(key)
            flattened[key] = data_type.transform(value)
        return flattened


__all__ = ["ParamsGenerator", "params_type"]
