from cfml import *
Experiment.suppress_warnings()

import numpy as np
from sklearn.neural_network import MLPClassifier

np.random.seed(142857)

fcnn = Base.make("fcnn_clf")
sk_clf = MLPClassifier()
experiment = Experiment({"cfml": fcnn}, {"sklearn": sk_clf}, show_images=True)

# xor
experiment.run(Data.xor())

# spiral
experiment.run(Data.spiral())

# two clusters
experiment.run(Data.two_clusters())

# simple non-linear
experiment.run(Data.simple_non_linear())

# nine grid
experiment.run(Data.nine_grid())
