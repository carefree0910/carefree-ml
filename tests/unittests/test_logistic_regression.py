from sklearn.linear_model import LogisticRegression

from cfml import *


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
with timeit("cfml", precision=8):
    lr.fit(breast_cancer.x, breast_cancer.y)
sk_clf = LogisticRegression(max_iter=10000)
with timeit("sklearn", precision=8):
    sk_clf.fit(breast_cancer.x, breast_cancer.y.ravel())

Comparer({"cfml": lr}, {"sklearn": sk_clf}).compare(*breast_cancer.xy)
