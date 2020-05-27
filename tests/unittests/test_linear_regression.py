from cfml import *
Experiment.suppress_warnings()

from sklearn.linear_model import LinearRegression


# basic usage

prices = Data().read("prices.txt")
(
    Base
        .make("linear_regression")
        .fit(prices.x, prices.y)
        .visualize1d(prices.x, prices.y)
        .plot_loss_curve()
)


# comparison

lr = Base.make("linear_regression")
poly = Base.make("poly", deg=1)
sk_reg = LinearRegression(normalize=True)

experiment = Experiment({"cfml_lr": lr, "cfml_poly": poly}, {"sklearn": sk_reg}, dtype="reg")
experiment.run(prices)
