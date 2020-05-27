from cfml import *
Experiment.suppress_warnings()

from sklearn.linear_model import LogisticRegression


# basic usage

breast_cancer = Data.breast_cancer()
(
    Base
        .make("logistic_regression")
        .fit(breast_cancer.x, breast_cancer.y)
        .plot_loss_curve()
)


# comparison

lr = Base.make("logistic_regression")
sk_clf = LogisticRegression(max_iter=10000)
Experiment({"cfml": lr}, {"sklearn": sk_clf}).run(breast_cancer)
