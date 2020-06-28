import os

from cfml import *
Experiment.suppress_warnings()

from cfdata.tabular import *
from sklearn.linear_model import LinearRegression


def test():
    file_folder = os.path.dirname(__file__)
    prices_file = os.path.abspath(os.path.join(file_folder, os.pardir, "datasets", "prices.txt"))
    prices = TabularData().read(prices_file).to_dataset()
    lr = Base.make("linear_regression")
    poly = Base.make("poly", deg=1)
    sk_reg = LinearRegression(normalize=True)

    experiment = Experiment(
        {"cfml_lr": lr, "cfml_poly": poly}, {"sklearn": sk_reg},
        task_type=TaskTypes.REGRESSION
    )
    experiment.run(prices)


test()
