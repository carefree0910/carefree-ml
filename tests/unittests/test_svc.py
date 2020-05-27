from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression

from cfml import *


# basic usage

breast_cancer = Data.breast_cancer()
Base.make("svc").fit(breast_cancer.x, breast_cancer.y).plot_loss_curve()
Base.make("linear_svc").fit(breast_cancer.x, breast_cancer.y).plot_loss_curve()


# comparison

svc = Base.make("svc")
l_svc = Base.make("linear_svc")
with timeit("cfml_svc", precision=8):
    svc.fit(breast_cancer.x, breast_cancer.y)
    print()
with timeit("cfml_l_svc", precision=8):
    l_svc.fit(breast_cancer.x, breast_cancer.y)
sk_lr = LogisticRegression(max_iter=10000)
with timeit("sklearn_lr", precision=8):
    sk_lr.fit(breast_cancer.x, breast_cancer.y.ravel())

sk_l_svc = LinearSVC(max_iter=10000)
with timeit("sklearn_l_svc", precision=8):
    sk_l_svc.fit(breast_cancer.x, breast_cancer.y.ravel())
sk_svc = SVC()
with timeit("sklearn_svc", precision=8):
    sk_svc.fit(breast_cancer.x, breast_cancer.y.ravel())

cfml_models = {"cfml_svc": svc, "cfml_l_svc": l_svc}
sklearn_models = {"sklearn_lr": sk_lr, "sklearn_svc": sk_svc, "sklearn_l_svc": sk_l_svc}
Comparer(cfml_models, sklearn_models).compare(*breast_cancer.xy)
