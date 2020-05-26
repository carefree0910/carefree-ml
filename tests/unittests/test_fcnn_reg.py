from sklearn.neural_network import MLPRegressor

from cfml.models import Base
from cfml.misc.data import Data
from cfml.misc.toolkit import timeit, Metrics


# basic usage

boston = Data.boston()
Base.make("fcnn_reg").fit(boston.x, boston.y).plot_loss_curve()


# comparison

fcnn = Base.make("fcnn_reg")
with timeit("cfml", precision=8):
    fcnn.fit(boston.x, boston.y)
sk_reg = MLPRegressor()
with timeit("sklearn", precision=8):
    sk_reg.fit(boston.x, boston.y.ravel())

print(f"cfml     mse : {Metrics.mse(boston.y, fcnn.predict(boston.x)):6.4f}")
print(f"sklearn  mse : {Metrics.mse(boston.y, sk_reg.predict(boston.x)):6.4f}")
