from cfml import *
Experiment.suppress_warnings()

from sklearn.neural_network import MLPRegressor


boston = Data.boston()
fcnn = Base.make("fcnn_reg")
sk_reg = MLPRegressor()
Experiment({"cfml_fcnn": fcnn}, {"sklearn_fcnn": sk_reg}, dtype="reg").run(boston)
