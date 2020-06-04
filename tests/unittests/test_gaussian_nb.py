from cfml import *

import numpy as np

from cfdata.tabular import TabularDataset
from sklearn.naive_bayes import GaussianNB


breast_cancer = TabularDataset.breast_cancer()
nb = Base.make("gaussian_nb")
sk_clf = GaussianNB()
Experiment({"cfml_gnb": nb}, {"sklearn_gnb": sk_clf}).run(breast_cancer)

print(f"identical class_prior      : {np.allclose(sk_clf.class_prior_, nb.class_prior)}")
print(f"identical mean             : {np.allclose(sk_clf.theta_, nb.mean)}")
print(f"identical variance         : {np.allclose(sk_clf.sigma_, nb.variance)}")
cfml_prob = nb.predict_prob(breast_cancer.x)
sklearn_prob = sk_clf.predict_proba(breast_cancer.x)
print(f"max prediction difference  : {np.abs(cfml_prob - sklearn_prob).max()}")
