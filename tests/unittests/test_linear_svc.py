from sklearn.linear_model import LogisticRegression

from cfml.models import Base
from cfml.misc.data import Data
from cfml.misc.toolkit import timeit, Metrics


# basic usage

breast_cancer = Data.breast_cancer()
Base.make("linear_svc").fit(breast_cancer.x, breast_cancer.y).plot_loss_curve()


# comparison

l_svc = Base.make("linear_svc")
with timeit("cfml", precision=8):
    l_svc.fit(breast_cancer.x, breast_cancer.y)
sk_clf = LogisticRegression(max_iter=10000)
with timeit("sklearn", precision=8):
    sk_clf.fit(breast_cancer.x, breast_cancer.y.ravel())

print(f"cfml     auc : {Metrics.auc(breast_cancer.y, l_svc.predict_prob(breast_cancer.x)):6.4f}")
print(f"sklearn  auc : {Metrics.auc(breast_cancer.y, sk_clf.predict_proba(breast_cancer.x)):6.4f}")
print(f"cfml     acc : {Metrics.acc(breast_cancer.y, l_svc.predict(breast_cancer.x)):6.4f}")
print(f"sklearn  acc : {Metrics.acc(breast_cancer.y, sk_clf.predict(breast_cancer.x)):6.4f}")
