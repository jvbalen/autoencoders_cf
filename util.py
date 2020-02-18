import os
import datetime

import numpy as np
from scipy.sparse import csr_matrix, issparse, vstack


class Logger(object):

    def __init__(self, base_dir):

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = os.path.join(base_dir, timestamp)
        os.makedirs(self.log_dir)

    def log_config(self, config):
        config_file = os.path.join(self.log_dir, 'config.gin')
        with open(config_file, 'w') as f:
            f.write(config)

    def log_results(self, metrics, config=None):
        results_file = os.path.join(self.log_dir, 'results.csv')
        with open(results_file, 'w') as f:
            for metric, value in metrics.items():
                f.write(f'{metric},{value}\n')
            if config is not None:
                for line in config.split('\n'):
                    if line.startswith('#'):
                        continue
                    f.write(','.join(line.split(' = ')) + '\n')

    def log_coefs(self, coefs, intercepts=None):
        coefs_file = os.path.join(self.log_dir, 'coefs.npz')
        if coefs is not None:
            np.savez(coefs_file, coefs=coefs, intercepts=intercepts)
        else:
            print('`coefs` is None: nothing to save')


def sparse_info(m):
    print("{} of {}".format(type(m), m.dtype))
    print("shape = {}, nnz = {}".format(m.shape, m.nnz))
    print("density = {:.3}".format(m.nnz / np.prod(m.shape)))


def prune_global(x, target_density=0.005):

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
    to only prune per-row, use target_density = None
    """
    if row_nnz is None:
        return prune_global(x, target_density=target_density)
    if target_density is None:
        return prune_rows(x, target_nnz=row_nnz)

    x_glob = prune_global(x, target_density=target_density)
    x_rows = prune_rows(x, target_nnz=row_nnz)

    return sparse_union(x_glob, x_rows)


def sparse_union(x, y):

    y_nonzero = y.copy()
    y_nonzero.data = y_nonzero.data != 0

    return x + y - x.multiply(y_nonzero)
