from cfml import *
Experiment.suppress_warnings()

from sklearn.neural_network import MLPClassifier


# basic usage

breast_cancer = Data.breast_cancer()
(
    Base
        .make("fcnn_clf")
        .fit(breast_cancer.x, breast_cancer.y)
        .plot_loss_curve()
)


# comparison

fcnn = Base.make("fcnn_clf")
sk_clf = MLPClassifier()
Experiment({"cfml": fcnn}, {"sklearn": sk_clf}).run(breast_cancer)
