import os
import numpy as np

from typing import List, NamedTuple

current_path = os.path.abspath(os.path.split(__file__)[0])


class dataset(NamedTuple):
    x: np.ndarray
    y: np.ndarray
    dtype: str
    label_name: str
    feature_names: List[str]

    @property
    def is_clf(self):
        return self.dtype == "clf"

    @property
    def is_reg(self):
        return self.dtype == "reg"


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
            ys.append(float(line.pop(label_idx)))
            xs.append(list(map(float, line)))
        xs, ys = map(np.stack, [xs, ys])
        if names is None:
            label_name = feature_names = None
        else:
            label_name = names.pop(label_idx)
            feature_names = names
        if self._dtype is not None:
            dtype = self._dtype
        else:
            dtype = "clf" if np.allclose(ys, ys.astype(np.int)) else "reg"
        return dataset(xs, ys, dtype, label_name, feature_names)

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


__all__ = ["Data"]
