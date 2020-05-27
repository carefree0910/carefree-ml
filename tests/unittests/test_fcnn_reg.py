from sklearn.neural_network import MLPRegressor

from cfml import *


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

Comparer({"cfml": fcnn}, {"sklearn": sk_reg}, dtype="reg").compare(*boston.xy)
