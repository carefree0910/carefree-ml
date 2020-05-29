from cfml import *
Experiment.suppress_warnings()

from sklearn.linear_model import LogisticRegression


breast_cancer = Data.breast_cancer()
lr = Base.make("logistic_regression")
sk_clf = LogisticRegression(max_iter=10000)
Experiment({"cfml_lr": lr}, {"sklearn_lr": sk_clf}).run(breast_cancer)
