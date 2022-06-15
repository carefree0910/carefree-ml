from cftool.ml import *
from cftool.ml.param_utils import *
from cfdata.tabular import TabularDataset
from cfdata.tabular.toolkit import DataSplitter

from cfml.models import Base
from cfml.misc.hpo import HPO


def test():
    dataset = TabularDataset.digits()
    splitter = DataSplitter().fit(dataset)
    te, cv, tr = splitter.split_multiple([0.1, 0.1], return_remained=True)
    train_set, cv_set, test_set = tr.dataset, cv.dataset, te.dataset

    estimators = list(map(Estimator, ["acc", "auc"]))

    model = "fcnn_clf"
    hpo = HPO(
        model,
        {
            "lb": Float(Uniform(0.0, 1.0)),
            "lr": Float(Exponential(1e-5, 0.1)),
            "epoch": Int(Choice(values=[2, 20, 200])),
            "optimizer": String(Choice(values=["sgd", "rmsprop", "adam"])),
        },
        hpo_method="naive",
    ).search(
        train_set.x,
        train_set.y,
        estimators,
        cv_set.x,
        cv_set.y,
        num_jobs=1,
    )

    naive_pattern = ModelPattern.repeat(
        5,
        init_method=lambda: Base.make(model),
        train_method=lambda m: m.fit(*train_set.xy),
    )
    best_pattern = ModelPattern.repeat(
        5,
        init_method=lambda: Base.make(model, **hpo.best_param),
        train_method=lambda m: m.fit(*train_set.xy),
    )
    acc_pattern = ModelPattern.repeat(
        5,
        init_method=lambda: Base.make(model, **hpo.best_params["acc"]),
        train_method=lambda m: m.fit(*train_set.xy),
    )
    auc_pattern = ModelPattern.repeat(
        5,
        init_method=lambda: Base.make(model, **hpo.best_params["auc"]),
        train_method=lambda m: m.fit(*train_set.xy),
    )
    comparer = Comparer(
        {
            "naive": naive_pattern,
            "best": best_pattern,
            "acc_model": acc_pattern,
            "auc_model": auc_pattern,
        },
        estimators,
    )
    comparer.compare(*test_set.xy, verbose_level=2)


if __name__ == "__main__":
    test()
