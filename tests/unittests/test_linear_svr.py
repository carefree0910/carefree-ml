from sklearn.svm import LinearSVR
from sklearn.linear_model import LinearRegression

from cfml.models import Base
from cfml.misc.data import Data
from cfml.misc.toolkit import timeit, Metrics


# basic usage

boston = Data.boston()
(
    Base
        .make("linear_svr")
        .fit(boston.x, boston.y)
        .plot_loss_curve()
)


# comparison

lr = Base.make("linear_regression")
l_svr = Base.make("linear_svr")
with timeit("cfml_lr", precision=8):
    lr.fit(boston.x, boston.y)
    print()
with timeit("cfml_l_svr", precision=8):
    l_svr.fit(boston.x, boston.y)

sk_lr = LinearRegression()
with timeit("sklearn_lr", precision=8):
    sk_lr.fit(boston.x, boston.y.ravel())
sk_l_svr = LinearSVR(max_iter=10000)
with timeit("sklearn_l_sr", precision=8):
    sk_l_svr.fit(boston.x, boston.y.ravel())

print(f"cfml_lr        mse : {Metrics.mse(boston.y, lr.predict(boston.x)):6.4f}")
print(f"cfml_l_svr     mse : {Metrics.mse(boston.y, l_svr.predict(boston.x)):6.4f}")
print(f"sklearn_lr     mse : {Metrics.mse(boston.y, sk_lr.predict(boston.x)):6.4f}")
print(f"sklearn_l_svr  mse : {Metrics.mse(boston.y, sk_l_svr.predict(boston.x)):6.4f}")
