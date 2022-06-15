from cfdata.tabular import TabularDataset
from cfdata.tabular.toolkit import DataSplitter

from cfml.misc.toolkit import *
from cfml.misc.param_utils import *
from cfml.models import Base
from cfml.misc.hpo import HPO
from cfml.misc.hpo import HPOBase
from cfml.misc.optim import GradientDescentMixin


def test1():
    class LinearRegression(GradientDescentMixin):
        def __init__(self, dim, lr, epoch):
            self.w = np.random.random([dim, 1])
            self.b = np.random.random([1])
            self._lr, self._epoch = lr, epoch

        @property
        def parameter_names(self) -> List[str]:
            return ["w", "b"]

        def loss_function(
            self,
            x_batch: np.ndarray,
            y_batch: np.ndarray,
            batch_indices: np.ndarray,
        ) -> Dict[str, Any]:
            predictions = self.predict(x_batch)
            diff = predictions - y_batch
            return {"diff": diff, "loss": np.abs(diff).mean().item()}

        def gradient_function(
            self,
            x_batch: np.ndarray,
            y_batch: np.ndarray,
            batch_indices: np.ndarray,
            loss_dict: Dict[str, Any],
        ) -> Dict[str, np.ndarray]:
            diff = loss_dict["diff"]
            sign = np.sign(diff)
            return {"w": (sign * x_batch).mean(0, keepdims=True).T, "b": sign.mean(0)}

        def fit(self, x, y):
            self.setup_optimizer("adam", self._lr, epoch=self._epoch)
            self.gradient_descent(x, y)
            return self

        def predict(self, x):
            return x.dot(self.w) + self.b

    dim = 10
    w_true = np.random.random([dim, 1])
    b_true = np.random.random([1])
    x = np.random.random([1000, dim])
    y = x.dot(w_true) + b_true

    def pattern_creator(features, labels, param):
        model = LinearRegression(dim, **param)
        model.show_tqdm = False
        model.fit(features, labels)
        return ModelPattern(init_method=lambda: model)

    params = {
        "lr": Float(Exponential(1e-5, 0.1)),
        "epoch": Int(Choice(values=[2, 20, 200])),
    }

    estimators = list(map(Estimator, ["mae", "mse"]))
    for hpo_method in ["naive", "bo"]:
        hpo = HPOBase.make(hpo_method, pattern_creator, params)
        hpo.search(x, y, estimators, num_jobs=1, use_tqdm=True, verbose_level=1)


def test2():
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
    test1()
    test2()
