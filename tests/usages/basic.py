import os

from cfdata.tabular import *

from cfml import *

# datasets
boston = TabularDataset.boston()
prices_file = os.path.join("datasets", "prices.txt")
prices = TabularData(task_type=TaskTypes.REGRESSION).read(prices_file).to_dataset()
breast_cancer = TabularDataset.breast_cancer()
digits = TabularDataset.digits()
column_indices = list(range(digits.num_features))
digits_onehot = TabularData.from_dataset(
    digits, categorical_columns=column_indices
).to_dataset()

# numpy poly fit
Base.make("poly").fit(*prices.xy).visualize1d(*prices.xy)

# linear regression
Base.make("linear_regression").fit(*prices.xy).visualize1d(*prices.xy).plot_loss_curve()

# logistic regression
Base.make("logistic_regression").fit(*breast_cancer.xy).plot_loss_curve()

# multinomial naive bayes
Base.make("multinomial_nb").fit(*digits_onehot.xy)

# gaussian naive bayes
Base.make("gaussian_nb").fit(*breast_cancer.xy)

# linear support vector machine (classification)
Base.make("linear_svc").fit(breast_cancer.x, breast_cancer.y).plot_loss_curve()

# linear support vector machine (regression)
Base.make("linear_svr").fit(boston.x, boston.y).plot_loss_curve()

# support vector machine (classification)
Base.make("svc").fit(breast_cancer.x, breast_cancer.y).plot_loss_curve()

# support vector machine (regression)
Base.make("svr").fit(boston.x, boston.y).plot_loss_curve()

# fully connected neural network (classification)
Base.make("fcnn_clf").fit(*breast_cancer.xy).plot_loss_curve()

# fully connected neural network (regression)
Base.make("fcnn_reg").fit(*boston.xy).plot_loss_curve()
