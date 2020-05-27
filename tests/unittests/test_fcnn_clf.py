from cfml import *
Experiment.suppress_warnings()

from sklearn.neural_network import MLPClassifier


breast_cancer = Data.breast_cancer()
fcnn = Base.make("fcnn_clf")
sk_clf = MLPClassifier()
Experiment({"cfml": fcnn}, {"sklearn": sk_clf}).run(breast_cancer)
