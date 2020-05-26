from .mixins import *
from ..bases import ClassifierBase


@ClassifierBase.register("fcnn_clf")
class FCNNClassifier(FCNNInitializerMixin, FCNNClassifierMixin, ClassifierBase):
    def __init__(self, **kwargs):
        self._allow_multiclass = True
        kwargs.setdefault("loss", "cross_entropy")
        kwargs.setdefault("epoch", 100)
        super().__init__(**kwargs)
        if self._opt == "sgd":
            self._optimizer_config = kwargs.get("optimizer_config", {"momentum": 0.999})


__all__ = ["FCNNClassifier"]
