import numpy as np

from sklearn.naive_bayes import MultinomialNB

from cfml import *


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

Comparer({"cfml": nb}, {"sklearn": sk_clf}).compare(*digits.xy)
print(f"identical class_log_prior  : {np.allclose(sk_clf.class_log_prior_, nb.class_log_prior)}")
print(f"identical feature_log_prob : {np.allclose(sk_clf.feature_log_prob_, nb.feature_log_prob)}")
cfml_prob = nb.predict_prob(digits.x)
sklearn_prob = sk_clf.predict_proba(digits.x)
print(f"max prediction difference  : {np.abs(cfml_prob - sklearn_prob).max()}")
