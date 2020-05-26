import numpy as np

from sklearn.naive_bayes import MultinomialNB

from cfml.models import Base
from cfml.misc.data import Data
from cfml.misc.toolkit import timeit, Metrics


# basic usage

digits = Data.digits().to_one_hot()
(
    Base
        .make("multinomial_nb")
        .fit(digits.x, digits.y)
)


# comparison

nb = Base.make("multinomial_nb")
with timeit("cfml", precision=8):
    nb.fit(digits.x, digits.y)
sk_clf = MultinomialNB()
with timeit("sklearn", precision=8):
    sk_clf.fit(digits.x, digits.y.ravel())

print(f"cfml     auc : {Metrics.auc(digits.y, nb.predict_prob(digits.x)):6.4f}")
print(f"sklearn  auc : {Metrics.auc(digits.y, sk_clf.predict_proba(digits.x)):6.4f}")
print(f"cfml     acc : {Metrics.acc(digits.y, nb.predict(digits.x)):6.4f}")
print(f"sklearn  acc : {Metrics.acc(digits.y, sk_clf.predict(digits.x)):6.4f}")
print(f"identical class_log_prior  : {np.allclose(sk_clf.class_log_prior_, nb.class_log_prior)}")
print(f"identical feature_log_prob : {np.allclose(sk_clf.feature_log_prob_, nb.feature_log_prob)}")
