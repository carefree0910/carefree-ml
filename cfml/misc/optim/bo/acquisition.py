import warnings
import numpy as np

from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor

from .target_space import TargetSpace


class Acquisition:
    def __init__(
        self,
        gp: GaussianProcessRegressor,
        method: str,
        xi: float,
        kappa: float,
        kappa_decay: float,
        kappa_decay_delay: int,
    ):
        self.gp = gp
        self.method = method
        self.xi, self.kappa = xi, kappa
        self.kappa_decay, self.kappa_decay_delay = kappa_decay, kappa_decay_delay
        self._iter_counter = 0

    def __str__(self):
        return f"Acquisition({self.method})"

    __repr__ = __str__

    def update(self) -> "Acquisition":
        self._iter_counter += 1
        if self.kappa_decay < 1 and self._iter_counter > self.kappa_decay_delay:
            self.kappa *= self.kappa_decay
        return self

    def search_max(
        self,
        best_score: float,
        space: TargetSpace,
        num_warmup: int,
        num_iter: int,
    ) -> np.ndarray:
        _array = lambda n: np.array(
            list(map(space.param2array, [space.sample() for _ in range(n)]))
        )
        # warm up
        x_tries = _array(num_warmup)
        scores = self.score(x_tries.astype(np.float32), best_score)
        best_idx = scores.argmax().item()
        best_x, best_score = x_tries[best_idx], scores[best_idx]
        # optimization
        for _ in range(num_iter):
            x_try = _array(1).ravel()
            res = minimize(
                lambda x: -self.score(x.reshape([1, -1]), best_score),
                x_try,
                bounds=space.params_gen.all_bounds,
                method="L-BFGS-B",
            )

            if not res.success:
                continue

            if -res.fun > best_score:
                best_x = res.x
                best_score = -res.fun
        return best_x

    def score(self, x: np.ndarray, best_score: float) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = self.gp.predict(x, return_std=True)
        return getattr(self, f"_{self.method}")(mean, std, best_score)

    def _ucb(self, mean: np.ndarray, std: np.ndarray, best_score: float) -> np.ndarray:
        return mean + self.kappa * std

    def _ei(self, mean: np.ndarray, std: np.ndarray, best_score: float) -> np.ndarray:
        a = mean - best_score - self.xi
        z = a / std
        return a * norm.cdf(z) + std * norm.pdf(z)

    def _poi(self, mean: np.ndarray, std: np.ndarray, best_score: float) -> np.ndarray:
        return norm.cdf((mean - best_score - self.xi) / std)


__all__ = ["Acquisition"]
