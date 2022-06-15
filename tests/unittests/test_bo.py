from cfml.misc.param_utils import *
from cfml.misc.optim.bo import BayesianOptimization


def test():
    float_uniform = Float(Uniform(-10, 10))
    string_choice = String(Choice(values=["a", "b", "c"]))
    params = {
        "x1": Iterable([float_uniform, float_uniform]),
        "x2": Iterable([string_choice, string_choice]),
    }

    def fn(p):
        x1, x2 = p["x1"], p["x2"]
        r1 = -((x1[0] + 2 * x1[1] - 7) ** 2) - (2 * x1[0] + x1[1] - 5) ** 2
        r2 = (x2[0] == "b") + (x2[1] == "c")
        return r1 + 10.0 * r2

    # Ground Truth is [ [1, 3], ["b", "c"] ]
    bo = BayesianOptimization(fn, params).maximize()
    print(bo.best_result)
    bo.maximize()
    print(bo.best_result)
    bo.maximize()
    print(bo.best_result)


test()
