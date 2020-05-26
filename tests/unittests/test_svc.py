from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression

from cfml.models import Base
from cfml.misc.data import Data
from cfml.misc.toolkit import timeit, Metrics


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

print(f"cfml_svc       auc : {Metrics.auc(breast_cancer.y, svc.predict_prob(breast_cancer.x)):6.4f}")
print(f"cfml_l_svc     auc : {Metrics.auc(breast_cancer.y, l_svc.predict_prob(breast_cancer.x)):6.4f}")
print(f"sklearn_lr     auc : {Metrics.auc(breast_cancer.y, sk_lr.predict_proba(breast_cancer.x)):6.4f}")
print(f"cfml_svc       acc : {Metrics.acc(breast_cancer.y, svc.predict(breast_cancer.x)):6.4f}")
print(f"cfml_l_svc     acc : {Metrics.acc(breast_cancer.y, l_svc.predict(breast_cancer.x)):6.4f}")
print(f"sklearn_lr     acc : {Metrics.acc(breast_cancer.y, sk_lr.predict(breast_cancer.x)):6.4f}")
print(f"sklearn_l_svc  acc : {Metrics.acc(breast_cancer.y, sk_l_svc.predict(breast_cancer.x)):6.4f}")
print(f"sklearn_svc    acc : {Metrics.acc(breast_cancer.y, sk_svc.predict(breast_cancer.x)):6.4f}")
