from cfml import *

import numpy as np
from sklearn.naive_bayes import MultinomialNB


digits = Data.digits().to_one_hot()
nb = Base.make("multinomial_nb")
sk_clf = MultinomialNB()

Experiment({"cfml_mnb": nb}, {"sklearn_mnb": sk_clf}).run(digits)
print(f"identical class_log_prior  : {np.allclose(sk_clf.class_log_prior_, nb.class_log_prior)}")
print(f"identical feature_log_prob : {np.allclose(sk_clf.feature_log_prob_, nb.feature_log_prob)}")
cfml_prob = nb.predict_prob(digits.x)
sklearn_prob = sk_clf.predict_proba(digits.x)
print(f"max prediction difference  : {np.abs(cfml_prob - sklearn_prob).max()}")
