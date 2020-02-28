import os
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


def prune_global(x, target_density=0.005):
    """Prune a 2d np.array or sp.sparse matrix by setting elements
    with low absolute value to 0.0 and return as a sp.sparse.csr_matrix.

    Args:
        x: np.array or scipy.sparse matrix, array to be pruned
        target_density: the desired overall fraction of non-zeros
    """
    target_nnz = int(target_density * np.prod(x.shape))
    if issparse(x):
        x_sp = x.copy().tocsr()
        x_sp.eliminate_zeros()
        if x_sp.nnz <= target_nnz:
            return x_sp

        thr = np.partition(np.abs(x_sp.data), kth=-target_nnz)[-target_nnz]
        x_sp.data[np.abs(x_sp.data) < thr] = 0.0
    else:
        x = x.copy()
        thr = np.quantile(np.abs(x), 1.0-target_density)
        x[np.abs(x) < thr] = 0.0
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
        np.savez(path, weights=weights, biases=biases)


def load_weights(path):
    """Load weights and biases from a Numpy npz file saved with `save_weights`"""
    try:
        # look for sparse matrix
        weights = load_npz(path)
        # check if the same file also contains the intercepts (see `save_weights`)
        try:
            biases = np.load(path).get('biases', None)
        except KeyError:
            biases = np.load(path).get('intercepts', None)
    except ValueError:
        # look for dense arrays
        data = np.load(path)
        try:
            weights = data['weights']
            biases = data.get('biases', None)
        except KeyError:
            weights = data['coefs']
            biases = data.get('intercepts', None)

    return weights, biases


def sparse_info(m):
    """Print some information about a scipcy.sparse matrix"""
    print("{} of {}".format(type(m), m.dtype))
    print("shape = {}, nnz = {}".format(m.shape, m.nnz))
    print("density = {:.3}".format(m.nnz / np.prod(m.shape)))
