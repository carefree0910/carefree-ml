from sklearn.neural_network import MLPClassifier

from cfml import *


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
with timeit("cfml", precision=8):
    fcnn.fit(breast_cancer.x, breast_cancer.y)
sk_clf = MLPClassifier(max_iter=1000)
with timeit("sklearn", precision=8):
    sk_clf.fit(breast_cancer.x, breast_cancer.y.ravel())

print(f"cfml     auc : {Metrics.auc(breast_cancer.y, fcnn.predict_prob(breast_cancer.x)):6.4f}")
print(f"sklearn  auc : {Metrics.auc(breast_cancer.y, sk_clf.predict_proba(breast_cancer.x)):6.4f}")
print(f"cfml     acc : {Metrics.acc(breast_cancer.y, fcnn.predict(breast_cancer.x)):6.4f}")
print(f"sklearn  acc : {Metrics.acc(breast_cancer.y, sk_clf.predict(breast_cancer.x)):6.4f}")
