import numpy as np

from sklearn.naive_bayes import GaussianNB

from cfml import *


# basic usage

breast_cancer = Data.breast_cancer()
(
    Base
        .make("gaussian_nb")
        .fit(breast_cancer.x, breast_cancer.y)
)


# comparison

nb = Base.make("gaussian_nb")
with timeit("cfml", precision=8):
    nb.fit(breast_cancer.x, breast_cancer.y)
sk_clf = GaussianNB()
with timeit("sklearn", precision=8):
    sk_clf.fit(breast_cancer.x, breast_cancer.y.ravel())

cfml_prob = nb.predict_prob(breast_cancer.x)
sklearn_prob = sk_clf.predict_proba(breast_cancer.x)
print(f"cfml     auc : {Metrics.auc(breast_cancer.y, cfml_prob):6.4f}")
print(f"sklearn  auc : {Metrics.auc(breast_cancer.y, sklearn_prob):6.4f}")
print(f"cfml     acc : {Metrics.acc(breast_cancer.y, nb.predict(breast_cancer.x)):6.4f}")
print(f"sklearn  acc : {Metrics.acc(breast_cancer.y, sk_clf.predict(breast_cancer.x)):6.4f}")
print(f"identical class_prior      : {np.allclose(sk_clf.class_prior_, nb.class_prior)}")
print(f"identical mean             : {np.allclose(sk_clf.theta_, nb.mean)}")
print(f"identical variance         : {np.allclose(sk_clf.sigma_, nb.variance)}")
print(f"max prediction difference  : {np.abs(cfml_prob - sklearn_prob).max()}")
