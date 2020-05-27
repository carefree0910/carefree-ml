from cfml import *
Experiment.suppress_warnings()

from sklearn.neural_network import MLPRegressor


# basic usage

boston = Data.boston()
Base.make("fcnn_reg").fit(boston.x, boston.y).plot_loss_curve()


# comparison

fcnn = Base.make("fcnn_reg")
sk_reg = MLPRegressor()
Experiment({"cfml": fcnn}, {"sklearn": sk_reg}, dtype="reg").run(boston)
