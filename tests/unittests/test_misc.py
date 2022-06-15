import unittest
import numpy as np

from typing import *
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

from cfml.misc.optim import *
from cfml.misc.toolkit import *
from cfml.misc.param_utils import *


class TestMisc(unittest.TestCase):
    def test_anneal(self):
        anneal = Anneal("linear", 50, 0.01, 0.99)
        for i in range(100):
            self.assertEqual(
                f"{anneal.pop():.4f}",
                f"{i * 0.02 + 0.01 if i <= 49 else 0.99:.4f}",
            )

    def test_register_metric(self):
        @register_metric("large_is_good", 1, True)
        def large_is_good(self_, _, pred):
            threshold = self_.config.get("threshold", 0.5)
            return (pred[..., 1] >= threshold).mean()

        self.assertIs(large_is_good, Metrics.custom_metrics["large_is_good"]["f"])

        y = np.random.randint(0, 2, 5).reshape([-1, 1])
        predictions = np.array([0.1, 0.3, 0.5, 0.7, 0.9]).reshape([-1, 1])
        predictions = np.hstack([1 - predictions, predictions])

        metric_ins = Metrics("large_is_good")
        metric_ins.config["threshold"] = 0.5
        self.assertEqual(metric_ins.metric(y, predictions), 0.6)
        metric_ins.config["threshold"] = 0.7
        self.assertEqual(metric_ins.metric(y, predictions), 0.4)
        metric_ins.config["threshold"] = 0.9
        self.assertEqual(metric_ins.metric(y, predictions), 0.2)

    def test_patterns(self):
        x, y = load_iris(return_X_y=True)
        l_svc_creator = lambda: LinearSVC()
        lr_creator = lambda: LogisticRegression()
        train_method = lambda m: m.fit(x, y.ravel())
        l_svc_patterns = ModelPattern.repeat(
            3,
            init_method=l_svc_creator,
            train_method=train_method,
        )
        lr_patterns = ModelPattern.repeat(
            3,
            init_method=lr_creator,
            train_method=train_method,
        )
        l_svc_ensemble = EnsemblePattern(l_svc_patterns)
        lr_ensemble = EnsemblePattern(lr_patterns)
        l_svc_ensemble.predict(x)
        lr_ensemble.predict(x)
        estimators = [Estimator("acc")]
        comparer = Comparer({"l_svc": l_svc_ensemble, "lr": lr_ensemble}, estimators)
        comparer.compare(x, y).log_statistics()
        l_svc_estimator = estimators[0].select(["l_svc"])
        lr_estimator = estimators[0].select(["lr"])
        estimator = Estimator.merge([l_svc_estimator, lr_estimator])
        self.assertDictEqual(estimator.raw_metrics, estimators[0].raw_metrics)
        self.assertDictEqual(estimator.final_scores, estimators[0].final_scores)
        l_svc_comparer = comparer.select(["l_svc"])
        lr_comparer = comparer.select(["lr"])
        merged_comparer = Comparer.merge([l_svc_comparer, lr_comparer])
        self.assertDictEqual(
            comparer.estimator_statistics,
            merged_comparer.estimator_statistics,
        )

    def test_scalar_ema(self):
        ema = ScalarEMA(0.5)
        gt_list = [1, 0.75, 0.5, 0.3125]
        for i, gt in enumerate(gt_list):
            self.assertEqual(ema.update("score", 0.5**i), gt)

    def test_tracker(self):
        tracker = Tracker("__unittest__", "test_tracker", overwrite=True)
        for name in ["s1", "s2"]:
            for i in range(3):
                tracker.track_scalar(name, i, iteration=i + 1)
        tracker.track_message("m1", "Hello!")
        tracker = Tracker("__unittest__", "test_tracker")
        for name in ["s1", "s2"]:
            self.assertListEqual(tracker.scalars[name], [(i + 1, i) for i in range(3)])
        self.assertEqual(tracker.messages["m1"], "Hello!")

    def test_param_generator(self):
        params = {
            "a": Any(Choice(values=[[1, 2], [2, 3]])),
            "b": Iterable(
                [
                    Int(Choice(values=[5, 6])),
                    Float(Choice(values=[5.1, 5.9])),
                    String(Choice(values=["b1", "b2"])),
                ]
            ),
            "c": {
                "d": Int(Choice(values=[7, 8])),
                "e": Any(Choice(values=[10, 9])),
                "f": {
                    "g": String(Choice(values=["g1", "g2"])),
                    "h": Int(Choice(values=[13, 14])),
                    "i": {
                        "j": Int(Choice(values=[15, 16])),
                        "k": String(Choice(values=["k1", "k2"])),
                    },
                },
            },
            "d": {
                "d1": Iterable(
                    [
                        Int(Choice(values=[17, 18])),
                        Float(Choice(values=[16.9, 17.5, 18.1])),
                    ]
                ),
                "d2": Iterable(
                    (
                        Int(Choice(values=[19, 20, 21])),
                        Float(Choice(values=[18.5, 19.5, 20.5, 21.5])),
                    )
                ),
            },
        }
        pg = ParamsGenerator(params)
        self.assertEqual(pg.num_params, 2**11 * 3**2 * 4)

    def test_gradient_descent(self):
        def _get_stacked(x_):
            x2, x3 = map(np.power, 2 * [x_], [2, 3])
            return np.hstack([x_, x2, x3])

        class _test(GradientDescentMixin):
            def __init__(self):
                self.w = np.random.random([1, 3])

            @property
            def parameter_names(self) -> List[str]:
                return ["w"]

            def loss_function(
                self,
                x_batch: np.ndarray,
                y_batch: np.ndarray,
                batch_indices: np.ndarray,
            ) -> Dict[str, Any]:
                x_batch_stacked = _get_stacked(x_batch)
                predictions = self.w * x_batch_stacked
                diff = predictions - y_batch
                return {
                    "diff": diff,
                    "x_stacked": x_batch_stacked,
                    "loss": np.abs(diff).mean().item(),
                }

            def gradient_function(
                self,
                x_batch: np.ndarray,
                y_batch: np.ndarray,
                batch_indices: np.ndarray,
                loss_dict: Dict[str, Any],
            ) -> Dict[str, np.ndarray]:
                diff, x_stacked_ = map(loss_dict.get, ["diff", "x_stacked"])
                return {"w": (np.sign(diff) * x_stacked_).mean(0, keepdims=True)}

        num_try = 50
        for _ in tqdm(list(range(num_try))):
            x = np.random.random([1000, 1])
            x_stacked = _get_stacked(x)
            w_true = np.random.random([1, 3])
            y = x_stacked * w_true
            gd = _test().setup_optimizer("adam", 3e-4, epoch=100)
            gd.show_tqdm = False
            gd.gradient_descent(x, y)
            self.assertTrue(np.allclose(gd.w, w_true, 1e-3, 1e-3))


if __name__ == "__main__":
    unittest.main()
