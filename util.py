
import os
import time
import datetime

import numpy as np
from scipy.sparse import csr_matrix, issparse, vstack, save_npz, load_npz


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

    def log_metrics(self, metrics, config=None):
        if self.verbose:
            print(f'Validation results: {metrics}')

        results_file = os.path.join(self.log_dir, 'results.csv')
        with open(results_file, 'w') as f:
            for metric, value in metrics.items():
                f.write(f'{metric},{value}\n')
            if config is not None:
                for line in config.split('\n'):
                    if line.startswith('#'):
                        continue
                    f.write(','.join(line.split(' = ')) + '\n')

    def save_weights(self, weights, biases=None):
        path = os.path.join(self.log_dir, 'weights.npz')
        if weights is not None:
            save_weights(path, weights, biases)
        else:
            print('`coefs` is None: nothing to save')


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

    def tic(self):
        self.t0 = time.perf_counter()

    def toc(self):
        elapsed = time.perf_counter() - self.t0
        if self.verbose:
            print(f'    elapsed: {elapsed:.3f}')

    def print_message(self, message=None):
        if self.verbose and message is not None:
            print(self.prefix + message + self.suffix)

    def interval(self, message=None):
        self.toc()
        self.print_message(message)
        self.tic()


def prune_global(x, target_density=0.005, copy=True):
    """Prune a 2d np.array or sp.sparse matrix by setting elements
    with low absolute value to 0.0 and return as a sp.sparse.csr_matrix.

    Args:
        x: np.array or scipy.sparse matrix, array to be pruned
        target_density: the desired overall fraction of non-zeros
    """
    target_nnz = int(target_density * np.prod(x.shape))
    if issparse(x):
        if copy:
            x_sp = x.copy()
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


def save_weights(path, weights, biases=None):
    """Save weights and biases to a Numpy npz file with `np.savez`.

    If the weights are sparse, `scipy.sparse.save_npz` is used.
    If the weights are sparse and biases is not None, we
    slighlty abuse the format used by `save_npz` and add an extra
    array "biases" to the same file. This does not interfere
    with `scipy.sparse.load_npz`.
    """
    if issparse(weights):
        save_npz(path, weights)
        if biases is not None:
            data = np.load(path)
            np.savez(path, biases=biases, **data)
    else:
        if biases is None:
            np.savez(path, weights=weights)
        else:
            np.savez(path, weights=weights, biases=biases)


def load_weights(path):
    """Load weights and biases from a Numpy npz file saved with `save_weights`"""
    try:
        # look for sparse matrix
        weights = load_npz(path)
        # check if the same file also contains the intercepts (see `save_weights`)
        data = np.load(path, allow_pickle=True)
        try:
            biases = data.get('biases')
        except KeyError:
            biases = data.get('intercepts', None)
    except ValueError:
        # look for dense arrays
        data = np.load(path, allow_pickle=True)
        try:
            weights = data['weights']
            biases = data.get('biases', None)
        except KeyError:
            weights = data['coefs']
            biases = data.get('intercepts', None)

    return weights, biases


def gen_batches(x, y=None, batch_size=100, shuffle=False, print_interval=1):
    """Generate batches from data arrays x and y
    """
    n_examples = x.shape[0]
    batch_inds = gen_batch_inds(n_examples, batch_size=batch_size, shuffle=shuffle, print_interval=print_interval)
    for inds in batch_inds:
        if y is None:
            yield x[inds], None
        else:
            yield x[inds], y[inds]


def gen_batch_inds(n_examples, batch_size=100, shuffle=False, print_interval=1):

    inds = np.array(range(n_examples)).astype(int)
    if shuffle:
        np.random.shuffle(inds)
    if batch_size is None:
        yield inds
        return
    n_batches = int(np.ceil(n_examples / batch_size))
    for i_batch, start in enumerate(range(0, n_examples, batch_size)):
        end = min(start + batch_size, n_examples)
        if print_interval is not None and i_batch % print_interval == 0:
            print('  batch {}/{}...'.format(i_batch + 1, n_batches))
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
