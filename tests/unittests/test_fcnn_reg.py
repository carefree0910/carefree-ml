from cfml import *

Experiment.suppress_warnings()

from cfdata.tabular import *
from sklearn.neural_network import MLPRegressor


def test():
    data = TabularDataset.california()
    fcnn = Base.make("fcnn_reg")
    sk_reg = MLPRegressor()
    Experiment(
        {"cfml_fcnn": fcnn},
        {"sklearn_fcnn": sk_reg},
        task_type=TaskTypes.REGRESSION,
    ).run(data)


test()
