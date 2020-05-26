import numpy as np

from ..bases import RegressorBase


@RegressorBase.register("poly")
class Polygon(RegressorBase):
    def __init__(self,
                 deg: int = 3):
        self._deg = deg
        self._param = None

    def fit(self,
            x: np.ndarray,
            y: np.ndarray) -> "Polygon":
        if x.shape[1] != 1:
            raise ValueError("Polygon only supports 1-dimensional features")
        self._param = np.polyfit(x.ravel(), y.ravel(), self._deg)
        return self

    def predict(self,
                x: np.ndarray) -> np.ndarray:
        if self._param is None:
            self.raise_not_fit(self)
        return np.polyval(self._param, x.ravel()).reshape([-1, 1])


__all__ = ["Polygon"]
