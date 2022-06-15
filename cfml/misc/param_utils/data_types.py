import math
import random

from abc import *
from typing import *
from cftool.misc import prod
from cftool.misc import Grid

from .types import *
from .distributions import DistributionBase


class DataType(metaclass=ABCMeta):
    def __init__(self, distribution: Optional[DistributionBase] = None, **kwargs: Any):
        self.dist, self.config = distribution, kwargs

    @property
    @abstractmethod
    def num_params(self) -> number_type:
        raise NotImplementedError

    @abstractmethod
    def _transform(self, value) -> generic_number_type:
        raise NotImplementedError

    def __str__(self):
        return f"{type(self).__name__}({self.dist})"

    __repr__ = __str__

    @property
    def lower(self) -> nullable_number_type:
        dist_lower = self.dist.lower
        if dist_lower is None:
            return
        return self._transform(dist_lower)

    @property
    def upper(self) -> nullable_number_type:
        dist_upper = self.dist.upper
        if dist_upper is None:
            return
        return self._transform(dist_upper)

    @property
    def values(self) -> Optional[List[generic_number_type]]:
        dist_values = self.dist.values
        if dist_values is None:
            return None
        return list(map(self._transform, dist_values))

    @property
    def bounds(self) -> bounds_type:
        lower, upper = map(self._transform, self.dist.bounds)
        return lower, upper

    @property
    def is_inf(self) -> bool:
        return math.isinf(self.num_params)

    @property
    def distribution_is_inf(self) -> bool:
        return math.isinf(self.dist.num_params)

    def _all(self) -> List[generic_number_type]:
        return self.values

    def pop(self) -> generic_number_type:
        return self._transform(self.dist.pop())

    def all(self) -> List[generic_number_type]:
        if self.is_inf:
            raise ValueError("'all' method could be called iff n_params is finite")
        return self._all()

    def transform(self, value: generic_number_type) -> generic_number_type:
        return self._transform(self.dist.clip(value))


class Any(DataType):
    @property
    def num_params(self) -> number_type:
        return self.dist.num_params

    def _transform(self, value) -> generic_number_type:
        return value


class Int(DataType):
    @property
    def lower(self) -> nullable_number_type:
        dist_lower = self.dist.lower
        if dist_lower is None:
            return
        return int(math.ceil(self.dist.lower))

    @property
    def upper(self) -> nullable_number_type:
        dist_lower = self.dist.upper
        if dist_lower is None:
            return
        return int(math.floor(self.dist.upper))

    @property
    def num_params(self) -> int:
        if self.distribution_is_inf:
            return int(self.upper - self.lower) + 1
        return self.dist.num_params

    def _all(self) -> List[int]:
        if self.distribution_is_inf:
            return list(range(self.lower, self.upper + 1))
        return super()._all()

    def _transform(self, value) -> int:
        return int(round(value + random.random() * 2e-4 - 1e-4))


class Float(DataType):
    @property
    def num_params(self) -> number_type:
        return self.dist.num_params

    def _transform(self, value) -> float:
        return float(value)


class Bool(DataType):
    @property
    def num_params(self) -> int:
        if self.distribution_is_inf:
            return 2
        return len(self._all())

    def _all(self) -> List[bool]:
        return sorted(super()._all())

    def _transform(self, value) -> bool:
        return bool(value)


class String(DataType):
    @property
    def num_params(self) -> number_type:
        return self.dist.num_params

    def _transform(self, value) -> str:
        return str(value)


iterable_data_type = Union[List[DataType], Tuple[DataType, ...]]
iterable_generic_number_type = Union[
    List[generic_number_type], Tuple[generic_number_type, ...]
]


class Iterable:
    def __init__(self, values: iterable_data_type):
        self._values = values
        self.is_list = isinstance(values, list)
        self._constructor = list if self.is_list else tuple

    def __str__(self):
        braces = "[]" if self._constructor is list else "()"
        return f"{braces[0]}{', '.join(map(str, self._values))}{braces[1]}"

    __repr__ = __str__

    def pop(self) -> iterable_generic_number_type:
        return self._constructor(v.pop() for v in self._values)

    def all(self) -> Iterator[generic_number_type]:
        grid = Grid([v.all() for v in self._values])
        for v in grid:
            yield self._constructor(v)

    def transform(
        self,
        value: iterable_generic_number_type,
    ) -> iterable_generic_number_type:
        return self._constructor(v.transform(vv) for v, vv in zip(self._values, value))

    @property
    def values(self) -> iterable_data_type:
        return self._values

    @property
    def bounds(self) -> List[bounds_type]:
        return [v.bounds for v in self._values]

    @property
    def num_params(self) -> number_type:
        num_params = prod(v.num_params for v in self._values)
        if math.isinf(num_params):
            return num_params
        return int(num_params)


__all__ = [
    "DataType",
    "Any",
    "Int",
    "Float",
    "Bool",
    "String",
    "Iterable",
    "iterable_data_type",
    "iterable_generic_number_type",
    "bounds_type",
]
