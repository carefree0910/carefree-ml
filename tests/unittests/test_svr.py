from cfml import *

Experiment.suppress_warnings()

from cfdata.tabular import *
from sklearn.svm import SVR, LinearSVR


def test():
    boston = TabularDataset.boston()
    svr = Base.make("svr")
    l_svr = Base.make("linear_svr")
    sk_svr = SVR()
    sk_l_svr = LinearSVR(max_iter=10000)

    cfml_models = {"cfml_svr": svr, "cfml_l_svr": l_svr}
    sklearn_models = {"sklearn_svr": sk_svr, "sklearn_l_svr": sk_l_svr}
    Experiment(cfml_models, sklearn_models, task_type=TaskTypes.REGRESSION).run(boston)


test()
