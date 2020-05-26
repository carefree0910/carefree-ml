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
poly = Base.make("poly")
with timeit("cfml_lr", precision=8):
    lr.fit(prices.x, prices.y)
with timeit("cfml_poly", precision=8):
    poly.fit(prices.x, prices.y)
sk_reg = LinearRegression(normalize=True)
with timeit("sklearn", precision=8):
    sk_reg.fit(prices.x, prices.y)

print(f"cfml_lr    mse : {Metrics.mse(prices.y, lr.predict(prices.x)):6.4f}")
print(f"cfml_poly  mse : {Metrics.mse(prices.y, poly.predict(prices.x)):6.4f}")
print(f"sklearn    mse : {Metrics.mse(prices.y, sk_reg.predict(prices.x)):6.4f}")
