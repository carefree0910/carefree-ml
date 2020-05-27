import os
import math
import numpy as np

from typing import *
from functools import partial
from sklearn.utils import Bunch
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import *

current_path = os.path.abspath(os.path.split(__file__)[0])


class dataset(NamedTuple):
    x: np.ndarray
    y: np.ndarray
    dtype: str
    label_name: Union[None, str]
    label_names: Union[None, List[str]]
    feature_names: Union[None, List[str]]

    @property
    def xy(self):
        return self.x, self.y

    @property
    def is_clf(self):
        return self.dtype == "clf"

    @property
    def is_reg(self):
        return self.dtype == "reg"

    @staticmethod
    def to_dtype(dtype: str, x: np.ndarray, y: np.ndarray):
        x = x.astype(np.float32)
        y = y.reshape([-1, 1]).astype(np.int if dtype == "clf" else np.float32)
        return x, y

    @classmethod
    def from_bunch(cls, dtype: str, bunch: Bunch) -> "dataset":
        x, y = cls.to_dtype(dtype, bunch.data, bunch.target)
        label_names = bunch.get("target_names")
        feature_names = bunch.get("feature_names")
        return dataset(x, y, dtype, "label", label_names, feature_names)

    @classmethod
    def from_xy(cls, dtype: str, x: np.ndarray, y: np.ndarray) -> "dataset":
        x, y = cls.to_dtype(dtype, x, y)
        return dataset(x, y, dtype, "label", None, None)

    def to_one_hot(self, categories="auto") -> "dataset":
        one_hot = OneHotEncoder(categories=categories, sparse=False).fit_transform(self.x)
        return dataset(one_hot.astype(np.float32), *self[1:])


class Data:
    def __init__(self, dtype: str = None):
        self._dtype = dtype
        self._datasets_path = os.path.join(current_path, "datasets")

    def _read_core(self, f, delim, label_idx, names=None) -> dataset:
        xs, ys = [], []
        for line in f:
            line = line.strip().split(delim)
            while label_idx < 0:
                label_idx += len(line)
            ys.append([float(line.pop(label_idx))])
            xs.append(list(map(float, line)))
        xs, ys = map(np.stack, [xs, ys])
        if names is None:
            label_name = feature_names = None
        else:
            label_name = names.pop(label_idx)
            feature_names = names
        xs = xs.astype(np.float32)
        if self._dtype is not None:
            dtype = self._dtype
            ys = ys.astype(np.int if dtype == "clf" else np.float32)
        else:
            ys_int = ys.astype(np.int)
            if np.allclose(ys, ys_int):
                ys = ys_int
                dtype = "clf"
            else:
                dtype = "reg"
                ys = ys.astype(np.float32)
        return dataset(xs, ys, dtype, label_name, None, feature_names)

    def _read_txt(self, file: str, *, delim: str = ",", label_idx: int = -1) -> dataset:
        with open(os.path.join(self._datasets_path, file), "r") as f:
            return self._read_core(f, delim, label_idx)

    def _read_csv(self, file: str, *, delim: str = ",", label_idx: int = -1) -> dataset:
        with open(os.path.join(self._datasets_path, file), "r") as f:
            names = f.readline().strip().split(delim)
            return self._read_core(f, delim, label_idx, names)

    def read(self,
             file: str,
             *,
             delim: str = ",",
             label_idx: int = -1) -> dataset:
        if file.endswith(".txt"):
            return self._read_txt(file, delim=delim, label_idx=label_idx)
        if file.endswith(".csv"):
            return self._read_csv(file, delim=delim, label_idx=label_idx)
        raise NotImplementedError(f"'{file}' is not a valid file type for Data")

    @staticmethod
    def iris() -> dataset:
        return dataset.from_bunch("clf", load_iris())

    @staticmethod
    def boston() -> dataset:
        return dataset.from_bunch("reg", load_boston())

    @staticmethod
    def digits() -> dataset:
        return dataset.from_bunch("clf", load_digits())

    @staticmethod
    def breast_cancer() -> dataset:
        return dataset.from_bunch("clf", load_breast_cancer())

    @staticmethod
    def xor(*,
            size: int = 100,
            scale: float = 1.) -> dataset:
        x = np.random.randn(size) * scale
        y = np.random.randn(size) * scale
        z = (x * y >= 0).astype(np.int)
        return dataset.from_xy("clf", np.c_[x, y].astype(np.float32), z)

    @staticmethod
    def spiral(*,
               size: int = 50,
               scale: float = 4.,
               nun_spirals: int = 7,
               num_classes: int = 7) -> dataset:
        xs = np.zeros((size * nun_spirals, 2), dtype=np.float32)
        ys = np.zeros(size * nun_spirals, dtype=np.int)
        pi = math.pi
        for i in range(nun_spirals):
            ix = range(size * i, size * (i + 1))
            r = np.linspace(0.0, 1, size + 1)[1:]
            t_start = 2 * i * pi / nun_spirals
            t_end = 2 * (i + scale) * pi / nun_spirals
            t = np.linspace(t_start, t_end, size) + np.random.random(size=size) * 0.1
            xs[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
            ys[ix] = i % num_classes
        return dataset.from_xy("clf", xs, ys)

    @staticmethod
    def two_clusters(*,
                     size: int = 100,
                     scale: float = 1.,
                     center: float = 0.,
                     distance: float = 2.,
                     num_dimensions: int = 2) -> dataset:
        center1 = (np.random.random(num_dimensions) + center - 0.5) * scale + distance
        center2 = (np.random.random(num_dimensions) + center - 0.5) * scale - distance
        cluster1 = (np.random.randn(size, num_dimensions) + center1) * scale
        cluster2 = (np.random.randn(size, num_dimensions) + center2) * scale
        data = np.vstack((cluster1, cluster2)).astype(np.float32)
        labels = np.array([1] * size + [0] * size, np.int)
        indices = np.random.permutation(size * 2)
        data, labels = data[indices], labels[indices]
        return dataset.from_xy("clf", data, labels)

    @staticmethod
    def simple_non_linear(*,
                          size: int = 120) -> dataset:
        xs = np.random.randn(size, 2).astype(np.float32) * 1.5
        ys = np.zeros(size, dtype=np.int)
        mask = xs[..., 1] >= xs[..., 0] ** 2
        xs[..., 1][mask] += 2
        ys[mask] = 1
        return dataset.from_xy("clf", xs, ys)

    @staticmethod
    def nine_grid(*,
                  size: int = 120) -> dataset:
        x, y = np.random.randn(2, size).astype(np.float32)
        labels = np.zeros(size, np.int)
        xl, xr = x <= -1, x >= 1
        yf, yc = y <= -1, y >= 1
        x_mid_mask = ~xl & ~xr
        y_mid_mask = ~yf & ~yc
        mask2 = x_mid_mask & y_mid_mask
        labels[mask2] = 2
        labels[(x_mid_mask | y_mid_mask) & ~mask2] = 1
        xs = np.vstack([x, y]).T
        return dataset.from_xy("clf", xs, labels)

    @staticmethod
    def noisy_linear(*,
                     dtype: str = "reg",
                     size: int = 10000,
                     n_dim: int = 100,
                     n_valid: int = 5,
                     noise_scale: float = 0.5,
                     test_ratio: float = 0.15) -> Tuple[dataset, dataset]:
        x_train = np.random.randn(size, n_dim)
        x_train_noise = x_train + np.random.randn(size, n_dim) * noise_scale
        x_test = np.random.randn(int(size * test_ratio), n_dim)
        idx = np.random.permutation(n_dim)[:n_valid]
        w = np.random.randn(n_valid, 1)
        affine_train = x_train[..., idx].dot(w)
        affine_test = x_test[..., idx].dot(w)
        if dtype == "reg":
            y_train, y_test = affine_train, affine_test
        else:
            y_train = (affine_train > 0).astype(np.int)
            y_test = (affine_test > 0).astype(np.int)
        tr_set, te_set = map(dataset.from_xy, 2 *[dtype], [x_train_noise, x_test], [y_train, y_test])
        return tr_set, te_set

    @staticmethod
    def gen_noisy_poly(*,
                       dtype: str = "reg",
                       p: int = 3,
                       size: int = 10000,
                       n_dim: int = 100,
                       n_valid: int = 5,
                       noise_scale: float = 0.5,
                       test_ratio: float = 0.15) -> Tuple[dataset, dataset]:
        assert p > 1, "p should be greater than 1"
        x_train = np.random.randn(size, n_dim)
        x_train_list = [x_train] + [x_train ** i for i in range(2, p + 1)]
        x_train_noise = x_train + np.random.randn(size, n_dim) * noise_scale
        x_test = np.random.randn(int(size * test_ratio), n_dim)
        x_test_list = [x_test] + [x_test ** i for i in range(2, p + 1)]
        idx_list = [np.random.permutation(n_dim)[:n_valid] for _ in range(p)]
        w_list = [np.random.randn(n_valid, 1) for _ in range(p)]
        o_train = [x[..., idx].dot(w) for x, idx, w in zip(x_train_list, idx_list, w_list)]
        o_test = [x[..., idx].dot(w) for x, idx, w in zip(x_test_list, idx_list, w_list)]
        affine_train, affine_test = map(partial(np.sum, axis=0), o_train, o_test)
        if dtype == "reg":
            y_train, y_test = affine_train, affine_test
        else:
            y_train = (affine_train > 0).astype(np.int)
            y_test = (affine_test > 0).astype(np.int)
        tr_set, te_set = map(dataset.from_xy, 2 * [dtype], [x_train_noise, x_test], [y_train, y_test])
        return tr_set, te_set


__all__ = ["Data"]
