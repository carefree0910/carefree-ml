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

Comparer({"cfml": fcnn}, {"sklearn": sk_clf}).compare(*breast_cancer.xy)
