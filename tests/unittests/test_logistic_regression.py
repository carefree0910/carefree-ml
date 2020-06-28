from cfml import *
Experiment.suppress_warnings()

from cfdata.tabular import TabularDataset
from sklearn.linear_model import LogisticRegression


def test():
    breast_cancer = TabularDataset.breast_cancer()
    lr = Base.make("logistic_regression")
    sk_clf = LogisticRegression(max_iter=10000)
    Experiment({"cfml_lr": lr}, {"sklearn_lr": sk_clf}).run(breast_cancer)


test()
