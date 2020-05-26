from sklearn.svm import SVR, LinearSVR

from cfml.models import Base
from cfml.misc.data import Data
from cfml.misc.toolkit import timeit, Metrics


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

print(f"cfml_svr       mse : {Metrics.mse(boston.y, svr.predict(boston.x)):6.4f}")
print(f"cfml_l_svr     mse : {Metrics.mse(boston.y, l_svr.predict(boston.x)):6.4f}")
print(f"sklearn_svr    mse : {Metrics.mse(boston.y, sk_svr.predict(boston.x)):6.4f}")
print(f"sklearn_l_svr  mse : {Metrics.mse(boston.y, sk_l_svr.predict(boston.x)):6.4f}")
