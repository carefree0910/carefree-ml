from cfml import *

Experiment.suppress_warnings()

import numpy as np

from cfdata.tabular import *
from sklearn.neural_network import MLPRegressor

np.random.seed(142857)

fcnn = Base.make("fcnn_reg")
sk_reg = MLPRegressor()
experiment = Experiment(
    {"cfml": fcnn},
    {"sklearn": sk_reg},
    task_type=TaskTypes.REGRESSION,
    show_images=True,
)

# noisy poly
experiment.run(*TabularDataset.noisy_poly(p=2, size=100, n_dim=1, n_valid=1))
experiment.run(*TabularDataset.noisy_poly(p=3, size=100, n_dim=1, n_valid=1))
experiment.run(*TabularDataset.noisy_poly(p=5, size=100, n_dim=1, n_valid=1))
experiment.run(*TabularDataset.noisy_poly(p=8, size=100, n_dim=1, n_valid=1))

# noisy linear
experiment.run(*TabularDataset.noisy_linear(size=100, n_dim=1, n_valid=1))
