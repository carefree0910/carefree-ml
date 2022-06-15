import math
import random

import numpy as np
import matplotlib.pyplot as plt

from abc import *
from typing import *

from .types import *


class DistributionBase(metaclass=ABCMeta):
    def __init__(
        self,
        lower: Optional[number_type] = None,
        upper: Optional[number_type] = None,
        *,
        values: Optional[List[Any]] = None,
        **kwargs,
    ):
        number_types = (int, float)
        if lower is not None and not isinstance(lower, number_types):
            raise ValueError(f"lower should be a number, {type(lower)} found")
        if upper is not None and not isinstance(upper, number_types):
            raise ValueError(f"upper should be a number, {type(upper)} found")
        self.lower, self.upper, self.values, self.config = lower, upper, values, kwargs

    @property
    @abstractmethod
    def num_params(self) -> number_type:
        raise NotImplementedError

    @abstractmethod
    def pop(self) -> generic_number_type:
        raise NotImplementedError

    @abstractmethod
    def clip(self, value: generic_number_type) -> generic_number_type:
        pass

    @property
    def bounds(self) -> bounds_type:
        if self.values is None:
            return self.lower, self.upper
        return min(self.values), max(self.values)

    def __str__(self):
        if self.values is not None:
            return f"{type(self).__name__}[{', '.join(map(str, self.values))}]"
        if not self.config:
            cfg_str = ""
        else:
            sorted_key = sorted(self.config)
            cfg_str = f", {', '.join([f'{k}={self.config[k]}' for k in sorted_key])}"
        return f"{type(self).__name__}[{self.lower:.2f}, {self.upper:.2f}{cfg_str}]"

    __repr__ = __str__

    def _assert_lower_and_upper(self):
        assert self.lower is not None, "lower should be provided"
        assert self.upper is not None, "upper should be provided"

    def _assert_values(self):
        assert isinstance(self.values, list), "values should be a list"

    def visualize(self, n: int = 100) -> "DistributionBase":
        plt.figure()
        plt.scatter(list(range(n)), sorted(self.pop() for _ in range(n)))
        plt.show()
        return self


class Uniform(DistributionBase):
    @property
    def num_params(self) -> number_type:
        return math.inf

    def pop(self) -> number_type:
        self._assert_lower_and_upper()
        return random.random() * (self.upper - self.lower) + self.lower

    def clip(self, value: number_type) -> number_type:
        lower, upper = self.bounds
        return max(lower, min(upper, value))


class Exponential(Uniform):
    def __init__(
        self,
        lower: Optional[number_type] = None,
        upper: Optional[number_type] = None,
        *,
        values: Optional[List[Any]] = None,
        **kwargs,
    ):
        super().__init__(lower, upper, values=values, **kwargs)
        self._assert_lower_and_upper()
        assert_msg = "lower should be greater than 0 in exponential distribution"
        assert self.lower > 0, assert_msg
        self.base = self.config.setdefault("base", 2)
        assert self.base > 1, "base should be greater than 1"
        self.lower, self.upper = map(
            math.log,
            [self.lower, self.upper],
            2 * [self.base],
        )

    @property
    def bounds(self) -> bounds_type:
        lower, upper = map(math.pow, 2 * [self.base], [self.lower, self.upper])
        return lower, upper

    def pop(self) -> number_type:
        return math.pow(self.base, super().pop())


class Choice(DistributionBase):
    @property
    def num_params(self) -> int:
        return len(self.values)

    def pop(self) -> Any:
        self._assert_values()
        return random.choice(self.values)

    def clip(self, value: generic_number_type) -> generic_number_type:
        if value is None or isinstance(value, str):
            return value
        diff = np.array([v - value for v in self.values], np.float32)
        best_idx = np.argmin(np.abs(diff)).item()
        return self.values[best_idx]


__all__ = ["DistributionBase", "Uniform", "Exponential", "Choice"]
