from typing import *

number_type = Union[int, float]
generic_number_type = Union[number_type, Any]
nullable_number_type = Optional[number_type]
bounds_type = Tuple[generic_number_type, generic_number_type]


__all__ = ["number_type", "generic_number_type", "nullable_number_type", "bounds_type"]
