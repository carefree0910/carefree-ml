from .mixins import *
from ..bases import RegressorBase


@RegressorBase.register("fcnn_reg")
class FCNNRegressor(FCNNInitializerMixin, FCNNRegressorMixin, RegressorBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._normalize_labels = True


__all__ = ["FCNNRegressor"]
