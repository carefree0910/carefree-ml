from sklearn.svm import SVR, LinearSVR

from cfml import *


# basic usage

boston = Data.boston()
Base.make("svr").fit(boston.x, boston.y).plot_loss_curve()
Base.make("linear_svr").fit(boston.x, boston.y).plot_loss_curve()


# comparison

svr = Base.make("svr")
l_svr = Base.make("linear_svr")
with timeit("cfml_svr", precision=8):
    svr.fit(boston.x, boston.y)
    print()
with timeit("cfml_l_svr", precision=8):
    l_svr.fit(boston.x, boston.y)

sk_svr = SVR()
with timeit("sklearn_svr", precision=8):
    sk_svr.fit(boston.x, boston.y.ravel())
sk_l_svr = LinearSVR(max_iter=10000)
with timeit("sklearn_l_svr", precision=8):
    sk_l_svr.fit(boston.x, boston.y.ravel())

cfml_models = {"cfml_svr": svr, "cfml_l_svr": l_svr}
sklearn_models = {"sklearn_svr": sk_svr, "sklearn_l_svr": sk_l_svr}
Comparer(cfml_models, sklearn_models, dtype="reg").compare(*boston.xy)
