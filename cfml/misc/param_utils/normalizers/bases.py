from abc import ABC, abstractmethod
from typing import Dict, List, Type, Union
from functools import partial
from cftool.misc import register_core

from ..types import *
from ..data_types import *
from ..distributions import *


union_data_type = Union[DataType, Iterable]
union_value_type = Union[generic_number_type, iterable_generic_number_type]
union_float_type = Union[float, List[float]]
union_bounds_type = Union[bounds_type, List[bounds_type]]
single_normalizer_dict: Dict[str, Type["SingleNormalizer"]] = {}


class SingleNormalizer(ABC):
    def __init__(self, data_type: DataType, **kwargs):
        self.core, self.dist = data_type, data_type.dist
        self._is_string = isinstance(data_type, String)
        self._is_exponential = isinstance(self.dist, Exponential)
        if not self.core.distribution_is_inf:
            self._choices = self.core.all()
        else:
            if not self.core.is_inf:
                self._choices = self.core.all()
            else:
                self._choices = None
                if not self._is_string:
                    self._lower, self._upper = self.core.lower, self.core.upper
                else:
                    self._lower, self._upper = self.dist.lower, self.dist.upper
                self._diff = self._upper - self._lower
        self._init_config(**kwargs)

    @property
    @abstractmethod
    def bounds(self) -> bounds_type:
        pass

    @abstractmethod
    def normalize(self, value: generic_number_type) -> float:
        pass

    @abstractmethod
    def recover(self, value: float) -> generic_number_type:
        pass

    def _init_config(self, **kwargs):
        pass

    @classmethod
    def register(cls, name: str):
        global single_normalizer_dict
        return register_core(name, single_normalizer_dict)


class IterableNormalizer:
    def __init__(
        self,
        data_type: Iterable,
        single_normalizer_base: Type[SingleNormalizer],
        **kwargs
    ):
        self._data_type = data_type
        self._normalizers = list(
            map(partial(single_normalizer_base, **kwargs), data_type.values)
        )

    @property
    def bounds(self) -> List[bounds_type]:
        return [normalizer.bounds for normalizer in self._normalizers]

    def normalize(self, value: iterable_generic_number_type) -> List[float]:
        return [
            normalizer.normalize(v) for normalizer, v in zip(self._normalizers, value)
        ]

    def recover(self, value: List[float]) -> iterable_generic_number_type:
        results = [
            normalizer.recover(v) for normalizer, v in zip(self._normalizers, value)
        ]
        if not self._data_type.is_list:
            results = tuple(results)
        return results


class Normalizer:
    def __init__(self, method: str, data_type: union_data_type, **kwargs):
        self._data_type = data_type
        self.is_iterable = isinstance(data_type, Iterable)
        single_normalizer_base = single_normalizer_dict[method]
        if not self.is_iterable:
            self._single_normalizer = single_normalizer_base(data_type, **kwargs)
        else:
            self._iterable_normalizer = IterableNormalizer(
                data_type,
                single_normalizer_base,
                **kwargs,
            )

    @property
    def bounds(self) -> union_bounds_type:
        if self.is_iterable:
            return self._iterable_normalizer.bounds
        return self._single_normalizer.bounds

    def normalize(self, value: union_value_type) -> union_float_type:
        if self.is_iterable:
            return self._iterable_normalizer.normalize(value)
        return self._single_normalizer.normalize(value)

    def recover(self, value: union_float_type) -> union_value_type:
        if self.is_iterable:
            return self._iterable_normalizer.recover(value)
        return self._single_normalizer.recover(value)


__all__ = ["Normalizer"]
