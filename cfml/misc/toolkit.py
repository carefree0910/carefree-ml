import dill
import math
import time
import hashlib
import datetime
import operator
import unicodedata

import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

from scipy import interp
from sklearn import metrics
from functools import reduce
from typing import Iterable, List, Dict

dill._dill._reverse_typemap["ClassType"] = type


# util functions

def timestamp(simplify=False, ensure_different=False):
    """
    Return current timestamp

    Parameters
    ----------
    simplify : bool. If True, format will be simplified to 'year-month-day'
    ensure_different : bool. If True, format will include millisecond

    Returns
    -------
    timestamp : str

    """

    now = datetime.datetime.now()
    if simplify:
        return now.strftime("%Y-%m-%d")
    if ensure_different:
        return now.strftime("%Y-%m-%d_%H-%M-%S-%f")
    return now.strftime("%Y-%m-%d_%H-%M-%S")


def prod(iterable):
    """ return cumulative production of an iterable """

    return float(reduce(operator.mul, iterable, 1))


def hash_code(code, encode=True):
    """ return hash code for a string """

    if encode:
        code = code.encode()
    return hashlib.md5(code).hexdigest()[:8]


def prefix_dict(d, prefix):
    """ prefix every key in dict `d` with `prefix` """

    return {f"{prefix}_{k}": v for k, v in d.items()}


def check_params(module):
    """
    Check out whether the param definitions in module is correct

    Parameters
    ----------
    module : torch.nn.Module
        Should be a torch module with `main_params` & `aux_params` defined

    """

    assert not (set(module.main_params) & set(module.aux_params))
    assert set(module.parameters()) == set(module.main_params) | set(module.aux_params)
    module._params_checked_ = True


def shallow_copy_dict(d: dict) -> dict:
    d = d.copy()
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = shallow_copy_dict(v)
    return d


def update_dict(src_dict: dict, tgt_dict: dict) -> dict:
    """
    Update tgt_dict with src_dict
    * Notice that changes will happen only on keys which src_dict holds

    Parameters
    ----------
    src_dict : dict
    tgt_dict : dict

    Returns
    -------
    tgt_dict : dict

    """

    for k, v in src_dict.items():
        tgt_v = tgt_dict.get(k)
        if tgt_v is None:
            tgt_dict[k] = v
        elif not isinstance(v, dict):
            tgt_dict[k] = v
        else:
            update_dict(v, tgt_v)
    return tgt_dict


def fix_float_to_length(num: float, length: int) -> str:
    """ change a float number to string format with fixed length """

    str_num = f"{num:f}"
    if str_num == "nan":
        return f"{str_num:^{length}s}"
    length = max(length, str_num.find("."))
    return str_num[:length].ljust(length, "0")


def truncate_string_to_length(string: str, length: int) -> str:
    """ truncate a string to make sure its length not exceeding a given length """

    if len(string) <= length:
        return string
    half_length = int(0.5 * length) - 1
    return string[:half_length] + "." * (length - 2 * half_length) + string[-half_length:]


def grouped(iterable: Iterable, n: int, *, keep_tail=False) -> List[tuple]:
    """ group an iterable every `n` elements """

    if not keep_tail:
        return list(zip(*[iter(iterable)] * n))
    with general_batch_manager(iterable, batch_size=n) as manager:
        return [tuple(batch) for batch in manager]


def is_numeric(s: str) -> bool:
    """ check whether `s` is a number """

    try:
        s = float(s)
        return True
    except (TypeError, ValueError):
        try:
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            return False


def get_one_hot(feature, num):
    """
    Get one-hot representation

    Parameters
    ----------
    feature : array-like, source data of one-hot representation
    num : int, dimension of the one-hot representation

    Returns
    -------
    one_hot : np.ndarray, one-hot representation of `feature`

    """

    if feature is None:
        return
    one_hot = np.zeros([len(feature), num], np.int64)
    one_hot[range(len(one_hot)), np.asarray(feature, np.int64).ravel()] = 1
    return one_hot


def show_or_save(export_path, fig=None, **kwargs):
    """
    Utility function to deal with figure

    Parameters
    ----------
    export_path : {None, str}
        * None : the figure will be shown
        * str  : it represents the path where the figure should be saved to
    fig : {None, plt.Figure}
        * None       : default figure contained in plt will be executed
        * plt.Figure : it will be executed

    """

    if export_path is None:
        fig.show(**kwargs) if fig is not None else plt.show(**kwargs)
    else:
        fig.savefig(export_path) if fig is not None else plt.savefig(export_path, **kwargs)
    plt.close()


def get_indices_from_another(base, segment):
    """
    Get `segment` elements' indices in `base`

    Warnings
    ----------
    All elements in segment should appear in base to ensure validity

    Parameters
    ----------
    base : np.ndarray, base array
    segment : np.ndarray, segment array

    Returns
    -------
    indices : np.ndarray, positions where elements in `segment` appear in `base`

    Examples
    -------
    >>> import numpy as np
    >>> base, segment = np.arange(100), np.random.permutation(100)[:10]
    >>> assert np.allclose(get_indices_from_another(base, segment), segment)

    """
    base_sorted_args = np.argsort(base)
    positions = np.searchsorted(base[base_sorted_args], segment)
    return base_sorted_args[positions]


def get_unique_indices(arr, return_raw=False):
    """
    Get indices for unique values of an array

    Parameters
    ----------
    arr : np.ndarray, target array which we wish to find indices of each unique value
    return_raw : bool, whether returning raw information

    Returns
    -------
    unique : np.ndarray, unique values of the given array (`arr`)
    unique_cnt : np.ndarray, counts of each unique value
        * If `return_raw`:
            sorting_indices : np.ndarray, indices which can (stably) sort the given array by its value
            split_arr : np.ndarray, array which can split the `sorting_indices` to make sure that each portion
            of the split indices belong & only belong to one of the unique values
        * If not `return_raw`:
            split_indices : list[np.ndarray], list of indices, each indices belong & only belong to
                one of the unique values

    Examples
    -------
    >>> import numpy as np
    >>> arr = np.array([1, 2, 3, 2, 4, 1, 0, 1], np.int64)
    >>> print(get_unique_indices(arr, return_raw=True), get_unique_indices(arr)[-1])
    >>> # [0, 1, 2, 3, 4]
    >>> # [1, 3, 2, 1, 1]
    >>> # [6, 0, 5, 7, 1, 3, 2, 4]
    >>> # [1, 4, 6, 7]
    >>> # [ [6], [0, 5, 7], [1, 3], [2], [4] ]

    """
    unique, unique_inv, unique_cnt = np.unique(arr, return_inverse=True, return_counts=True)
    sorting_indices, split_arr = np.argsort(unique_inv, kind="mergesort"), np.cumsum(unique_cnt)[:-1]
    if return_raw:
        return unique, unique_cnt, sorting_indices, split_arr
    return unique, unique_cnt, np.split(sorting_indices, split_arr)


def register_core(name: str,
                  global_dict: Dict[str, type], *,
                  before_register: callable = None,
                  after_register: callable = None):
    def _register(cls):
        if before_register is not None:
            before_register(cls)
        registered = global_dict.get(name)
        if registered is not None:
            print(f"~~~ [warning] '{name}' has already registered "
                  f"in the given global dict ({global_dict})")
            return cls
        global_dict[name] = cls
        if after_register is not None:
            after_register(cls)
        return cls
    return _register


# util classes

class Metrics:
    """
    Util class to calculate a whole variety of metrics

    Warnings
    ----------
    * Notice that 2-dimensional arrays are desired, not flattened arrays
    * Notice that first two args of each metric method must be `y` & `pred`

    Parameters
    ----------
    metric_type : str, indicates which kind of metric is to be calculated
    config : dict
        Configuration for the specific metric
        * e.g. for quantile metric, you need to specify which quantile is to be evaluated

    Examples
    --------
    >>> import numpy as np
    >>> predictions, y_true = map(np.atleast_2d, [[1., 2., 3.], [0., 2., 1.]])
    >>> print(Metrics("mae", {}).score(y_true.T, predictions.T))  # will be 1.

    """

    sign_dict = {
        "f1_score": 1, "r2_score": 1, "auc": 1, "multi_auc": 1,
        "acc": 1, "mae": -1, "mse": -1, "ber": -1,
        "correlation": 1
    }
    requires_prob_metrics = {"auc", "multi_auc"}
    requires_split_by_time_metrics = {"top_k_score"}
    optimized_binary_metrics = {"acc", "ber"}

    def __init__(self, metric_type=None, config=None):
        if config is None:
            config = {}
        self.type, self.config = metric_type, config

    @property
    def sign(self):
        return Metrics.sign_dict[self.type]

    @property
    def requires_prob(self):
        return self.type in self.requires_prob_metrics

    @staticmethod
    def _handle_nan(y, pred):
        pred_valid_mask = np.all(~np.isnan(pred), axis=1)
        valid_ratio = pred_valid_mask.mean()
        if valid_ratio == 0:
            print("all pred are nan")
            return None, None
        if valid_ratio != 1:
            print(f"pred contains nan (ratio={valid_ratio:6.4f})")
            y, pred = y[pred_valid_mask], pred[pred_valid_mask]
        return y, pred

    def score(self, y, pred):
        if self.type is None:
            raise ValueError("`score` method was called but type is not specified in Metrics")
        y, pred = self._handle_nan(y, pred)
        if y is None or pred is None:
            return float("nan")
        return getattr(self, self.type)(y, pred)

    # config-dependent metrics

    def quantile(self, y, pred):
        q, error = self.config["q"], y - pred
        if isinstance(q, list):
            q = np.array(q, np.float32).reshape([-1, 1])
        return np.maximum(q * error, (q - 1) * error).mean(0).sum()

    # static metrics

    @staticmethod
    def f1_score(y, pred):
        return metrics.f1_score(y.ravel(), pred.ravel())

    @staticmethod
    def r2_score(y, pred):
        return metrics.r2_score(y.ravel(), pred.ravel())

    @staticmethod
    def auc(y, pred):
        n_classes = pred.shape[1]
        if n_classes == 2:
            return metrics.roc_auc_score(y.ravel(), pred[..., 1])
        return Metrics.multi_auc(y, pred)

    @staticmethod
    def multi_auc(y, pred):
        n_classes = pred.shape[1]
        y = get_one_hot(y.ravel(), n_classes)
        fpr, tpr = [None] * n_classes, [None] * n_classes
        for i in range(n_classes):
            fpr[i], tpr[i], _ = metrics.roc_curve(y[..., i], pred[..., i])
        new_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        new_tpr = np.zeros_like(new_fpr)
        for i in range(n_classes):
            new_tpr += interp(new_fpr, fpr[i], tpr[i])
        new_tpr /= n_classes
        return metrics.auc(new_fpr, new_tpr)

    @staticmethod
    def acc(y, pred):
        return np.mean(y.ravel() == pred.ravel())

    @staticmethod
    def mae(y, pred):
        return np.mean(np.abs(y.ravel() - pred.ravel()))

    @staticmethod
    def mse(y, pred):
        return np.mean(np.square(y.ravel() - pred.ravel()))

    @staticmethod
    def ber(y, pred):
        mat = metrics.confusion_matrix(y.ravel(), pred.ravel())
        tp = np.diag(mat)
        fp = mat.sum(axis=0) - tp
        fn = mat.sum(axis=1) - tp
        tn = mat.sum() - (tp + fp + fn)
        return 0.5 * np.mean((fn / (tp + fn) + fp / (tn + fp)))

    @staticmethod
    def correlation(y, pred):
        return float(ss.pearsonr(y.ravel(), pred.ravel())[0])

    # auxiliaries

    @staticmethod
    def get_binary_threshold(y, probabilities, metric_type):
        pos_probabilities = probabilities[..., 1]
        fpr, tpr, thresholds = metrics.roc_curve(y, pos_probabilities)
        _, counts = np.unique(y, return_counts=True)
        pos = counts[1] / len(y)
        if metric_type == "ber":
            metric = 0.5 * (1 - tpr + fpr)
        elif metric_type == "acc":
            metric = tpr * pos + (1 - fpr) * (1 - pos)
        else:
            raise NotImplementedError(f"transformation from fpr, tpr -> '{metric_type}' is not implemented")
        metric *= Metrics.sign_dict[metric_type]
        return thresholds[np.argmax(metric)]


class Activations:
    def __init__(self, activation: str):
        self._activation = activation

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return getattr(Activations, self._activation)(x)

    def grad(self, forward: np.ndarray) -> np.ndarray:
        return getattr(Activations, f"{self._activation}_grad")(forward)

    def visualize(self, x_min: float = -5., x_max: float = 5.):
        plt.figure()
        x0 = np.linspace(x_min, x_max)
        plt.plot(x0, self(x0))
        plt.show()

    @staticmethod
    def sigmoid(x):
        return 1. / (1. + np.exp(-x))

    @staticmethod
    def sigmoid_grad(forward):
        return forward * (1. - forward)


class Incrementer:
    """
    Util class which can calculate running mean & running std efficiently

    Parameters
    ----------
    window_size : {int, None}, window size of running statistics
        * None : then all history records will be used for calculation

    Examples
    ----------
    >>> incrementer = Incrementer(window_size=5)
    >>> for i in range(10):
    >>>     incrementer.update(i)
    >>>     if i >= 4:
    >>>         print(incrementer.mean)  # will print 2.0, 3.0, ..., 6.0, 7.0

    """

    def __init__(self, window_size: int = None):
        if window_size is not None:
            if not isinstance(window_size, int):
                raise ValueError(f"window size should be integer, {type(window_size)} found")
            if window_size < 2:
                raise ValueError(f"window size should be greater than 2, {window_size} found")
        self._window_size = window_size
        self._n_record = self._running_sum = self._running_square_sum = self._previous = None

    @property
    def mean(self):
        return self._running_sum / self._n_record

    @property
    def std(self):
        return math.sqrt(max(self._running_square_sum / self._n_record - self.mean ** 2, 0.))

    @property
    def n_record(self):
        return self._n_record

    def update(self, new_value):
        if self._n_record is None:
            self._n_record, self._running_sum, self._running_square_sum = 1, new_value, new_value ** 2
        else:
            self._n_record += 1
            self._running_sum += new_value
            self._running_square_sum += new_value ** 2
        if self._window_size is not None:
            if self._previous is None:
                self._previous = [new_value]
            else:
                self._previous.append(new_value)
            if self._n_record == self._window_size + 1:
                self._n_record -= 1
                previous = self._previous.pop(0)
                self._running_sum -= previous
                self._running_square_sum -= previous ** 2


class ScalarEMA:
    """
    Util class to record Exponential Moving Average (EMA) for scalar value

    Parameters
    ----------
    decay : float, decay rate for EMA
        * Formula: new = (1 - decay) * current + decay * history; history = new

    Examples
    --------
    >>> ema = ScalarEMA(0.5)
    >>> for i in range(4):
    >>>     print(ema.update("score", 0.5 ** i))  # 1, 0.75, 0.5, 0.3125

    """

    def __init__(self, decay):
        self._decay = decay
        self._ema_records = {}

    def get(self, name):
        return self._ema_records.get(name)

    def update(self, name, new_value):
        history = self._ema_records.get(name)
        if history is None:
            updated = new_value
        else:
            updated = (1 - self._decay) * new_value + self._decay * history
        self._ema_records[name] = updated
        return updated


# constants

INFO_PREFIX = "~~~  [ info ] "


# contexts

class context_error_handler:
    """ Util class which provides exception handling when using context manager """

    @property
    def exception_suffix(self):
        return ""

    def _normal_exit(self, exc_type, exc_val, exc_tb):
        pass

    def _exception_exit(self, exc_type, exc_val, exc_tb):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not exc_type:
            self._normal_exit(exc_type, exc_val, exc_tb)
        else:
            self._exception_exit(exc_type, exc_val, exc_tb)


class timeit(context_error_handler):
    """
    Timing context manager

    Examples
    --------
    >>> with timeit("something"):
    >>>     # do something here
    >>> # will print "~~~  [ info ] timing for    something     : x.xxxx"

    """

    def __init__(self, msg, precision=6):
        self._msg = msg
        self._p = precision

    def __enter__(self):
        self._t = time.time()

    def _normal_exit(self, exc_type, exc_val, exc_tb):
        print(f"{INFO_PREFIX}timing for {self._msg:^16s} : {time.time() - self._t:{self._p}.{self._p-2}f}")


class general_batch_manager(context_error_handler):
    """
    Inference in batch, it could be any general instance

    Parameters
    ----------
    inputs : tuple(np.ndarray), auxiliary array inputs.
    n_elem : {int, float}, indicates how many elements will be processed in a batch
    batch_size : int, indicates the batch_size; if None, batch_size will be calculated by n_elem

    Examples
    --------
    >>> instance = type("test", (object,), {})()
    >>> with general_batch_manager(instance, np.arange(5), np.arange(1, 6), batch_size=2) as manager:
    >>>     for arr, tensor in manager:
    >>>         print(arr, tensor)
    >>>         # Will print:
    >>>         #   [0 1], [1 2]
    >>>         #   [2 3], [3 4]
    >>>         #   [4]  , [5]

    """

    def __init__(self, *inputs, n_elem=1e6, batch_size=None, max_batch_size=1024):
        if not inputs:
            raise ValueError("inputs should be provided in general_batch_manager")
        input_lengths = list(map(len, inputs))
        self._n, self._rs, self._inputs = input_lengths[0], [], inputs
        assert all(length == self._n for length in input_lengths), "inputs should be of same length"
        if batch_size is not None:
            self._batch_size = batch_size
        else:
            n_elem = int(n_elem)
            self._batch_size = int(n_elem / sum(map(lambda arr: prod(arr.shape[1:]), inputs)))
        self._batch_size = min(max_batch_size, min(self._n, self._batch_size))
        self._n_epoch = int(self._n / self._batch_size)
        self._n_epoch += int(self._n_epoch * self._batch_size < self._n)

    def __enter__(self):
        return self

    def __iter__(self):
        self._start, self._end = 0, self._batch_size
        return self

    def __next__(self):
        if self._start >= self._n:
            raise StopIteration
        batched_data = tuple(map(lambda arr: arr[self._start:self._end], self._inputs))
        self._start, self._end = self._end, self._end + self._batch_size
        if len(batched_data) == 1:
            return batched_data[0]
        return batched_data

    def __len__(self):
        return self._n_epoch


__all__ = [
    "get_indices_from_another", "get_unique_indices", "get_one_hot", "hash_code", "prefix_dict",
    "check_params", "timestamp", "fix_float_to_length", "truncate_string_to_length", "grouped",
    "is_numeric", "show_or_save", "update_dict", "Incrementer", "ScalarEMA",
    "context_error_handler", "timeit", "general_batch_manager",
     "prod", "shallow_copy_dict", "register_core"
]
