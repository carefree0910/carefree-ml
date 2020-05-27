from cfml import *
Experiment.suppress_warnings()

from sklearn.svm import SVR, LinearSVR


# basic usage

boston = Data.boston()
Base.make("svr").fit(boston.x, boston.y).plot_loss_curve()
Base.make("linear_svr").fit(boston.x, boston.y).plot_loss_curve()


# comparison

svr = Base.make("svr")
l_svr = Base.make("linear_svr")
sk_svr = SVR()
sk_l_svr = LinearSVR(max_iter=10000)

cfml_models = {"cfml_svr": svr, "cfml_l_svr": l_svr}
sklearn_models = {"sklearn_svr": sk_svr, "sklearn_l_svr": sk_l_svr}
Experiment(cfml_models, sklearn_models, dtype="reg").run(boston)
