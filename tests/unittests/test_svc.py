from cfml import *
Experiment.suppress_warnings()

from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression


# basic usage

breast_cancer = Data.breast_cancer()
Base.make("svc").fit(breast_cancer.x, breast_cancer.y).plot_loss_curve()
Base.make("linear_svc").fit(breast_cancer.x, breast_cancer.y).plot_loss_curve()


# comparison

svc = Base.make("svc")
l_svc = Base.make("linear_svc")
sk_lr = LogisticRegression(max_iter=10000)
sk_l_svc = LinearSVC(max_iter=10000)
sk_svc = SVC()

cfml_models = {"cfml_svc": svc, "cfml_l_svc": l_svc}
sklearn_models = {"sklearn_lr": sk_lr, "sklearn_svc": sk_svc, "sklearn_l_svc": sk_l_svc}
Experiment(cfml_models, sklearn_models).run(breast_cancer)
