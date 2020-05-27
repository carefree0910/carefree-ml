from cfml import *

# datasets
boston = Data.boston()
prices = Data().read("prices.txt")
breast_cancer = Data.breast_cancer()
digits = Data.digits()
digits_onehot = digits.to_one_hot()

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
