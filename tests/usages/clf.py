from cfml import *

Experiment.suppress_warnings()

import numpy as np

from cfdata.tabular import TabularDataset
from sklearn.neural_network import MLPClassifier

np.random.seed(142857)

fcnn = Base.make("fcnn_clf")
sk_clf = MLPClassifier()
experiment = Experiment({"cfml": fcnn}, {"sklearn": sk_clf}, show_images=True)

# xor
experiment.run(TabularDataset.xor())

# spiral
experiment.run(TabularDataset.spiral())

# two clusters
experiment.run(TabularDataset.two_clusters())

# simple non-linear
experiment.run(TabularDataset.simple_non_linear())

# nine grid
experiment.run(TabularDataset.nine_grid())
