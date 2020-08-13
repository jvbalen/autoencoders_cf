
import os
import time
import datetime

import numpy as np
from scipy.sparse import csr_matrix, issparse, vstack, save_npz, load_npz
from tqdm import tqdm


class Logger(object):

    def __init__(self, base_dir, verbose=True):
        """Class that holds methods for logging config, results and model coefficients
        to a common directory.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = os.path.join(base_dir, timestamp)
        self.verbose = verbose
        os.makedirs(self.log_dir)

    def log_config(self, config):
        config_file = os.path.join(self.log_dir, 'config.gin')
        with open(config_file, 'w') as f:
            f.write(config)

    def log_metrics(self, metrics, config=None, test=False):
        if self.verbose and test:
            print(f'Test results: {metrics}')
        elif self.verbose:
            print(f'Validation results: {metrics}')

        results_file = 'test_results.csv' if test else 'results.csv'
        results_file = os.path.join(self.log_dir, results_file)
        with open(results_file, 'w') as f:
            for metric, value in metrics.items():
                f.write(f'{metric},{value}\n')
            if config is not None:
                for line in config.split('\n'):
                    if line.startswith('#'):
                        continue
                    f.write(','.join(line.split(' = ')) + '\n')

    def save_weights(self, weights, other=dict()):
        path = os.path.join(self.log_dir, 'weights.npz')
        if not issparse(weights):
            other.update({'weights': weights})
            weights = None
        save_weights(path, weights, other=other)


class Node(object):

    def __init__(self, id, children=None):

        self.id = id
        if children is None:
            self.children = []
        else:
            self.children = children

    def flatten(self):

        all_nodes = [self.id]
        for child in self.children:
            all_nodes.extend(child.flatten())

        return all_nodes

    @property
    def n_children(self):
        return len(self.children)

    @property
    def size(self):

        return len(self.flatten())

    def copy(self):

        return Node(self.id, self.children)

    def __repr__(self):

        return f"{self.id}:[{','.join(str(ch) for ch in self.children)}]"


class Clock(object):

    def __init__(self, verbose=True, prefix='  ', suffix='...'):
        self.t0 = time.perf_counter()
        self.verbose = verbose
        self.prefix = prefix
        self.suffix = suffix

    def tic(self, message=None):
        self.t0 = time.perf_counter()
        if self.verbose and message is not None:
            print(self.prefix + message + self.suffix)

    def toc(self):
        elapsed = time.perf_counter() - self.t0
        if self.verbose:
            print(f'{self.prefix}  elapsed: {elapsed:.3f}')

    def interval(self, message=None):
        self.toc()
        self.tic(message)


def prune_global(x, target_density=0.005, copy=True):
    """Prune a 2d np.array or sp.sparse matrix by setting elements
    with low absolute value to 0.0 and return as a sp.sparse.csr_matrix.

    Args:
        x: np.array or scipy.sparse matrix, array to be pruned
        target_density: the desired overall fraction of non-zeros
    """
    target_nnz = int(target_density * np.prod(x.shape))
    if issparse(x):
        x_sp = x.copy() if copy else x
        x_sp.eliminate_zeros()
        if x_sp.nnz <= target_nnz:
            return x_sp
        thr = get_pruning_threshold(x, target_density=target_density)
        try:
            x_sp.data[np.abs(x_sp.data) <= thr] = 0.0
        except AttributeError:
            x_sp = x_sp.tocsr()
            x_sp.data[np.abs(x_sp.data) <= thr] = 0.0
    else:
        x = x.copy()
        thr = get_pruning_threshold(x, target_density=target_density)
        x[np.abs(x) <= thr] = 0.0
        x_sp = csr_matrix(x)
    x_sp.eliminate_zeros()

    return x_sp


def prune_rows(x, target_nnz=30):
    """Prune the rows of 2d np.array or sp.sparse matrix by setting elements
    with low absolute value to 0.0 and return a sp.sparse.csr_matrix.

    Args:
        x: np.array or scipy.sparse matrix, array to be pruned
        target_nnz: the desired number of non-zeros per row
    """
    if issparse(x):
        x = x.tocsr()
    if x.shape[1] <= target_nnz:
        return x
    target_density = target_nnz / x.shape[1]
    pruned_rows = [prune_global(row, target_density=target_density) for row in x]

    return vstack(pruned_rows)


def prune(x, target_density=0.005, row_nnz=None):
    """Prune a 2d np.array or sp.sparse matrix by setting elements
    with low absolute value to 0.0 and return as a sp.sparse.csr_matrix.

    A global target density can be combined with a desired minimum number of non-zeros
    per row. If both are specified, a conservative threshold is used:
    only the elements considered small by *both* criteria will be removed.

    To only prune globally, use row_nnz = None,
    to only prune per-row, use target_density = None.

    Args:
        x: np.array or scipy.sparse matrix, array to be pruned
        target_density: the desired overall fraction of non-zeros
        row_nnz: the desired number of non-zeros per row
    """
    if row_nnz is None:
        return prune_global(x, target_density=target_density)
    if target_density is None:
        return prune_rows(x, target_nnz=row_nnz)

    x_glob = prune_global(x, target_density=target_density)
    x_rows = prune_rows(x, target_nnz=row_nnz)

    return sparse_union(x_glob, x_rows)


def get_pruning_threshold(x, target_density=0.005):
    """Get the pruning threshold for a sparse or dense array given
    a desired density.
    """
    if issparse(x):
        try:
            abs_values = np.abs(x.data)
        except AttributeError:
            abs_values = np.abs(x.tocoo().data)
        target_nnz = int(target_density * np.prod(x.shape))
        if target_nnz > len(abs_values):
            return 0.0
        thr = np.partition(abs_values, kth=-target_nnz)[-target_nnz]
    else:
        thr = np.quantile(np.abs(x), 1.0-target_density)

    return thr


def sparse_union(x, y):
    """Merge two sparse matrices. Specifically, return z such that:
        z = x where x != 0
        z = y where y != 0 and x == 0
        z = 0 elsewhere
    """
    x_nonzero = x.copy()
    x_nonzero.data = x_nonzero.data != 0

    return x + y - y.multiply(x_nonzero)


def save_weights(path, sparse_weights, other=None):
    """Save a sparse matrix and optional other (dense) arrays to a Numpy npz file.

    To save sparse_weights, `scipy.sparse.save_npz` is used.
    If other arrays are given, they must be dense,
    slighlty abuse the format used by `save_npz` and add an extra
    array "biases" to the same file. This does not interfere
    with `scipy.sparse.load_npz`.
    """
    save_npz(path, sparse_weights)
    if other:
        data = np.load(path)
        data.update(other)
        np.savez(path, **data)


def load_weights(path):
    """Load weights and biases from a Numpy npz file saved with `save_weights`"""
    try:
        sparse_weights = load_npz(path)  # look for sparse matrix
    except ValueError:
        sparse_weights = None  # support files not containing a sparse array
    other = np.load(path, allow_pickle=True)

    return sparse_weights, other


def load_weights_biases(path):

    weights, other = load_weights(path)
    if weights is None:
        try:
            weights = other['weights']
        except KeyError:
            weights = other['coefs']
    try:
        biases = other['biases']
    except KeyError:
        try:
            biases = other['intercepts']
        except KeyError:
            biases = None

    return weights, biases


def gen_batches(x, y=None, batch_size=100, shuffle=False, progress_bar=True):
    """Generate batches from data arrays x and y
    """
    n_examples = x.shape[0]
    batch_inds = gen_batch_inds(n_examples, batch_size=batch_size, shuffle=shuffle, progress_bar=progress_bar)
    for inds in batch_inds:
        if y is None:
            yield x[inds], None
        else:
            yield x[inds], y[inds]


def gen_batch_inds(n_examples, batch_size=100, shuffle=False, progress_bar=True):

    inds = np.array(range(n_examples)).astype(int)
    if shuffle:
        np.random.shuffle(inds)
    if batch_size is None:
        yield inds
        return
    n_batches = int(np.ceil(n_examples / batch_size))
    for i_batch, start in tqdm(enumerate(range(0, n_examples, batch_size)), total=n_batches, disable=not progress_bar):
        end = min(start + batch_size, n_examples)
        yield inds[start:end]


def sparse_info(m):
    """Print some information about a scipcy.sparse matrix"""
    print("{} of {}".format(type(m), m.dtype))
    print("shape = {}, nnz = {}".format(m.shape, m.nnz))
    print("density = {:.3}".format(m.nnz / np.prod(m.shape)))


def to_float32(x, to_dense=False):
    """Convert numpy/sp.sparse data to float32"""
    if to_dense and issparse(x):
        x = x.toarray()

    return x.astype('float32')
