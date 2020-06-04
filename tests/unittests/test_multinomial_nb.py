from cfml import *

import numpy as np

from cfdata.tabular import *
from sklearn.naive_bayes import MultinomialNB


digits = TabularDataset.digits()
column_indices = list(range(digits.num_features))
digits_onehot = TabularData.from_dataset(digits, categorical_columns=column_indices).to_dataset()
nb = Base.make("multinomial_nb")
sk_clf = MultinomialNB()

Experiment({"cfml_mnb": nb}, {"sklearn_mnb": sk_clf}).run(digits_onehot)
print(f"identical class_log_prior  : {np.allclose(sk_clf.class_log_prior_, nb.class_log_prior)}")
print(f"identical feature_log_prob : {np.allclose(sk_clf.feature_log_prob_, nb.feature_log_prob)}")
cfml_prob = nb.predict_prob(digits_onehot.x)
sklearn_prob = sk_clf.predict_proba(digits_onehot.x)
print(f"max prediction difference  : {np.abs(cfml_prob - sklearn_prob).max()}")
