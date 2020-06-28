from cfml import *
Experiment.suppress_warnings()

from cfdata.tabular import TabularDataset
from sklearn.neural_network import MLPClassifier


def test():
    breast_cancer = TabularDataset.breast_cancer()
    fcnn = Base.make("fcnn_clf")
    sk_clf = MLPClassifier()
    Experiment({"cfml_fcnn": fcnn}, {"sklearn_fcnn": sk_clf}).run(breast_cancer)


test()
