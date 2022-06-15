from typing import Dict
from cftool.ml import ModelPattern
from cftool.ml.hpo import HPOBase
from cftool.ml.param_utils import DataType

from ...models.bases import Base


def HPO(
    model: str,
    params: Dict[str, DataType],
    *,
    hpo_method: str = "naive",
    verbose_level: int = 2
):
    def _creator(x, y, params_):
        m = Base.make(model, **params_)
        m.show_tqdm = False
        return ModelPattern(init_method=lambda: m.fit(x, y))

    return HPOBase.make(hpo_method, _creator, params, verbose_level=verbose_level)


__all__ = ["HPO"]
