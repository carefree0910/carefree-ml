from sklearn.linear_model import LinearRegression

from cfml import *


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
with timeit("cfml_lr", precision=8):
    lr.fit(prices.x, prices.y)
with timeit("cfml_poly", precision=8):
    poly.fit(prices.x, prices.y)
sk_reg = LinearRegression(normalize=True)
with timeit("sklearn", precision=8):
    sk_reg.fit(prices.x, prices.y)

comparer = Comparer({"cfml_lr": lr, "cfml_poly": poly}, {"sklearn": sk_reg}, dtype="reg")
comparer.compare(*prices.xy)
