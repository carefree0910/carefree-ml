from typing import *

from .mixins import *
from .modules import *
from ..bases import RegressorBase


@RegressorBase.register("fcnn_reg")
class FCNNRegressor(FCNNRegressorMixin, RegressorBase):
    def __init__(self, *,
                 lb: float = 0.,
                 lr: float = 3e-4,
                 loss: str = "l1",
                 epoch: int = 200,
                 optimizer: str = "sgd",
                 initializer: str = "uniform",
                 initializer_config: Dict[str, Any] = None,
                 hidden_layers: List[Tuple[int, Union[str, None]]] = None):
        self._lb = lb
        self._lr = lr
        self._epoch = epoch
        self._opt = optimizer
        # loss
        self._loss = Loss(loss)
        # initializer
        if initializer_config is None:
            initializer_config = {}
        self._initializer = Initializer(initializer, **initializer_config)
        # hidden layers
        self._hidden_layers = None
        if hidden_layers is not None:
            self._hidden_layers = [Layer(*layer) for layer in hidden_layers]
        # normalize_labels
        self._normalize_labels = True


__all__ = ["FCNNRegressor"]
