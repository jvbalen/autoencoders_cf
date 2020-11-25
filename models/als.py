"""Recommenders based on Alternating Least Squares optimization
"""
import time
from warnings import warn
from functools import partial
from collections import defaultdict

import gin
import numpy as np
from scipy.sparse import issparse, csr_matrix, hstack, vstack
from scipy.stats import norm, cauchy
from tqdm import tqdm

from models.base import BaseRecommender
from util import Clock, prune_global, prune_rows

np.seterr(invalid='raise')


@gin.configurable
class ALSRecommender(BaseRecommender):

    def __init__(self, log_dir, batch_size=70, save_weights=True,
                 latent_dim=50, n_iter=20, l2_reg=10., init_scale=0.1):
        """Alternating Least Squares matrix factorization recommender

        Approximates X as U @ V.T, finds U and V by alternating between solving for U
        and V using closed-form OLS.

        Parameters:
        - log_dir (str): logging directory (a new timestamped directory will be created inside of it)
        - batch_size (int): evaluation batch_size
        - save_weights (bool): whether to save weights
        - latent_dim (int): the dimension of the user and item embeddings
        - n_iter (int): number of alternating least squares steps. Users and items are each updated
            once on each iteration
        - l2_reg (float): l2 regularization parameter
        - init_scale (float): draw initial user and item vectors using a standard-normal distribution
            with this scale
        """
        self.save_weights = save_weights
        self.latent_dim = latent_dim
        self.n_iter = n_iter
        self.l2_reg = l2_reg
        self.init_scale = init_scale
        self.U = None
        self.V = None

        super().__init__(log_dir, batch_size=batch_size)

    def train(self, x_train, y_train, x_val, y_val):
        """Train and evaluate
        """
        if y_train is not None:
            warn("SLIM is unsupervised, y_train will be ignored")

        t1 = time.perf_counter()
        self.V = np.random.randn(x_train.shape[1], self.latent_dim) * self.init_scale
        for _ in tqdm(range(self.n_iter)):
            self.U = solve_ols(self.V, x_train.T, l2_reg=self.l2_reg, verbose=False).T
            self.V = solve_ols(self.U, x_train, l2_reg=self.l2_reg, verbose=False).T
        dt = time.perf_counter() - t1

        metrics = self.evaluate(x_val, y_val, other_metrics={'train_time': dt})
        if self.save_weights and self.logger is not None:
            self.logger.save_weights(None, other={'U': self.U, 'V': self.V})

        return metrics

    def predict(self, x, y=None):
        """Predict scores
        """
        u = solve_ols(self.V, x.T, l2_reg=self.l2_reg, verbose=False).T
        y_pred = u @ self.V.T

        return y_pred, np.nan


@gin.configurable
class WALSRecommender(BaseRecommender):

    def __init__(self, log_dir, batch_size=70, save_weights=True,
                 latent_dim=50, n_iter=20, l2_reg=10., init_scale=0.1,
                 alpha=1.0, beta=0.0, embeddings_path=None):
        """Matrix factorization recommender with weighted square loss, optimized using
        Alternating Least Squares optimization.

        Implementation focused on arbitrary dense weights. Supports Hu's original weighting scheme
        (negatives @ 1.0 and positives @ 1.0 + alpha) but not efficiently.
        Also supports experimental ranking loss-inducing weighting, in which both positives and
        negatives are weighted according to (an estimate of) the number of discordant pos-neg pairs
        they are part of, attenuated with exponent `discordance_weighting` in (0, 1].

        Parameters:
        - log_dir (str): logging directory (a new timestamped directory will be created inside of it)
        - batch_size (int): evaluation batch_size
        - save_weights (bool): whether to save weights
        - latent_dim (int): the dimension of the user and item embeddings
        - n_iter (int): number of alternating least squares steps. Users and items are each updated
            once on each iteration
        - l2_reg (float): l2 regularization parameter
        - init_scale (float): draw initial user and item vectors using a standard-normal distribution
            with this scale
        - alpha (float): if beta == 0, use Hu's W-ALS weighting with this alpha
        - beta (float):
        - cache_statistics (bool): if True, use discordance weighting during item factor
            updates as well, using a cached statistics of pos and negative item scores for each user
        """
        self.save_weights = save_weights
        self.latent_dim = latent_dim
        self.n_iter = n_iter
        self.l2_reg = l2_reg
        self.init_scale = init_scale
        self.alpha = alpha
        self.beta = beta
        self.embeddings_path = embeddings_path
        self.epsilon = 1e-3

        self.U = None
        self.V = None
        self.user_stats = defaultdict(list)

        super().__init__(log_dir, batch_size=batch_size)

    def train(self, x_train, y_train, x_val, y_val):
        """Train and evaluate
        """
        if y_train is not None:
            warn("WALS is unsupervised, y_train will be ignored")

        t1 = time.perf_counter()
        print('Initializing U, V...')
        x_train = x_train.tocsr()
        if self.embeddings_path:
            npz_data = np.load(self.embeddings_path)
            self.U, self.V = npz_data['U'], npz_data['V']
        else:
            self.U = np.random.randn(x_train.shape[0], self.latent_dim) * self.init_scale
            self.V = np.random.randn(x_train.shape[1], self.latent_dim) * self.init_scale

        for i in range(self.n_iter):
            print(f'Evaluating...')
            self.evaluate(x_val, y_val, step=i)

            print(f'Iteration {i + 1}/{self.n_iter}')
            print('Updating alpha...')
            alpha = map_csr(self.item_alpha, x_train, self.U) if self.beta else self.alpha * x_train
            print(f'Updating user vectors...')
            xtx = self.V.T @ self.V
            self.U = solve_wols(self.V, yt=x_train, at=alpha, l2_reg=self.l2_reg, xtx=xtx).T
            print('Updating alpha...')
            alpha = map_csr(self.item_alpha, x_train, self.U) if self.beta else self.alpha * x_train
            print(f'Updating item vectors...')
            xtx = self.U.T @ self.U
            self.V = solve_wols(self.U, yt=x_train.T, at=alpha.T, l2_reg=self.l2_reg, xtx=xtx).T

            if self.save_weights and self.logger is not None:
                self.logger.save_weights(None, other={'U': self.U, 'V': self.V})

        dt = time.perf_counter() - t1
        metrics = self.evaluate(x_val, y_val, step=self.n_iter, other_metrics={'train_time': dt})

        return metrics

    def item_alpha(self, x, u):

        pos = (x > 0).toarray().flatten()
        if u is not None and self.beta > 0:
            y = u @ self.V.T
            neg = np.logical_not(pos)
            y_pos, y_neg = y[pos], y[neg]

            q1_neg, q2_neg, q3_neg = np.quantile(y_neg, [0.5, 0.625, 0.875])
            m_neg, s_neg = q1_neg, q3_neg - q2_neg
            p_pos = cauchy(m_neg, s_neg).pdf(y_pos)
            w_pos = p_pos / np.maximum(1. - y_pos, self.epsilon)
            w_pos = w_pos ** self.beta  # compress
            cap = np.sqrt(self.alpha + 1) * np.median(w_pos)
            w_pos[w_pos > cap] = cap  # cap
            w_pos = w_pos / (np.mean(w_pos) + self.epsilon)  # normalize
        else:
            w_pos = 1
        a = x.copy()
        a[0, pos] = self.alpha * w_pos

        return a

    def predict(self, x, y=None, n_iter=1):
        n_users = x.shape[0]
        u = [None] * n_users
        xtx = self.V.T @ self.V
        for _ in range(n_iter):
            sparse_alpha = map_csr(self.item_alpha, x, u, verbose=False)
            u = solve_wols(self.V, yt=x, at=sparse_alpha, l2_reg=self.l2_reg, xtx=xtx, verbose=False).T
        y_pred = u @ self.V.T

        return y_pred, np.nan


@gin.configurable
class WSLIMRecommender(BaseRecommender):

    def __init__(self, log_dir, batch_size=100, target_density=1.0, save_weights=True,
                 row_nnz=100, l2_reg=5, alpha=10., beta=1.0):
        """'Sparse + Low-rank' recommender. Models:
            X ~ U @ V.T + X @ S
        """
        self.target_density = target_density
        self.save_weights = save_weights
        self.row_nnz = row_nnz
        self.l2_reg = l2_reg
        self.alpha = alpha
        self.beta = beta
        self.S = None

        super().__init__(log_dir, batch_size=batch_size)

    def train(self, x_train, y_train, x_val, y_val):
        """Train and evaluate
        """
        if y_train is not None:
            warn("SLIM is unsupervised, y_train will be ignored")
        verbose = self.logger.verbose if self.logger else False
        clock = Clock(verbose=verbose)
        t1 = time.perf_counter()

        clock.interval('Solving for S, using sparse approximation')
        wt = (self.beta + x_col.toarray().flatten() * self.alpha for x_col in x_train.T)
        self.S = solve_wols(x_train, yt=x_train.T, wt=wt, l2_reg=self.l2_reg, row_nnz=self.row_nnz)

        if self.target_density < 1.0:
            self.S = prune_global(self.S, target_density=self.target_density)
        dt = time.perf_counter() - t1
        if self.save_weights and self.logger is not None:
            self.logger.save_weights(self.S)

        density = self.S.size / np.prod(self.S.shape)
        metrics = self.evaluate(x_val, y_val, other_metrics={'train_time': dt, 'weight_density': density})

        return metrics

    def predict(self, x, y=None, use_uv=True):
        """Predict scores

        NOTE: if you get a dimension mismatch here for the final multiplication,
        you probably ended up with S (or another var) a matrix (instead of array)
        """
        y_pred = x @ self.S

        return y_pred, np.nan


def solve_ols(x, y, l2_reg=100., zero_diag=False, verbose=False):
    """Compute argmin_b |XB - Y|^2
    """
    if verbose:
        print(f'    np.isfinite(x).mean() = {np.isfinite(x.data).mean()}')
        print(f'    np.isfinite(y).mean() = {np.isfinite(y.data).mean()}')
    xtx = x.T @ x
    xty = x.T @ y
    if issparse(xtx):
        xtx = xtx.toarray()
    if issparse(xty):
        xty = xty.toarray()
    diag_indices = np.diag_indices(xtx.shape[0])
    xtx[diag_indices] += l2_reg
    xtx_inv = np.linalg.inv(xtx)
    b = xtx_inv @ xty
    if zero_diag:
        b -= xtx_inv * (np.diag(b) / np.diag(xtx_inv))
    if verbose:
        print(f'    np.isfinite(b).mean() = {np.isfinite(b).mean()}')
        print(f'    np.abs(b).min() = {np.abs(b).min()}')
        print(f'    np.abs(b).max() = {np.abs(b).max()}')

    return b


def solve_wols(x, yt, at, l2_reg=100, xtx=None, verbose=True):
    """Solve the weighted Ordinary Least Squares problem:
        argmin_B | W * (X B - Y) |^2
    where * denotes element-wise multiplication.

    Or equivalently, compute for each column y_i of Y:
        argmin_b_i | W_i (X b_i - y_i) |^2 + reg
    where W_i := diagM(W[:, i]) a diagonal matrix,
    using the closed-form solution
        b_i = (X.T W_i X + l2_reg * I) X.T W_i y

    Use parameter `yt` and `at` to specify Y.T and W.T = at + 1
    Either of these may be a passed as a generator.
    """
    try:
        n_cols = yt.shape[0]
    except AttributeError:
        n_cols = None
    if xtx is None:
        xtx = x.T @ x
    yt = yt.tocsr()
    at = at.tocsr()
    gen = tqdm(zip(yt, at), total=n_cols, disable=not verbose)
    cols = [solve_wols_col(x, yt_i, at_i, l2_reg=l2_reg, xtx=xtx) for yt_i, at_i in gen]
    return np.hstack(cols)


def solve_wols_col(x, yt_i, at_i, l2_reg, xtx=None):
    """Solve the weighted Ordinary Least Squares problem
    for one row.
    """
    a_i, y_i = at_i.T, yt_i.T

    if xtx is None:
        xtx = x.T @ x
    sqrta_x = a_i.power(0.5).multiply(x)
    xtx = xtx + sqrta_x.T @ sqrta_x
    xty = x.T @ y_i

    nz = a_i.nonzero()[0]
    xty = xty + x[nz].T @ a_i[nz].multiply(y_i[nz])

    if issparse(xtx):
        xtx = xtx.toarray()
    diag_indices = np.diag_indices(xtx.shape[0])
    xtx[diag_indices] += l2_reg
    xtx_inv = np.linalg.inv(xtx)
    b = xtx_inv @ xty

    return np.asarray(b)


def get_gram_neighbors(x, k=100):
    """Return the top neighbors of an item given a user-item matrix X,
    based on the gram matrix X.T @ X, as a list of arrays (excl. self)
    """
    sparse_gram = prune_rows(x.T @ x, target_nnz=k)
    sparse_gram[np.diag_indices(sparse_gram.shape[0])] = 0.0
    neighbors = [row.nonzero()[1] for row in sparse_gram]

    return neighbors


def gram_matrix(x, w=None):

    if w is None:
        return x.T @ x

    return x.T.multiply(w) @ x if issparse(x) else (x.T * w) @ x


def map_csr(fn, *args, verbose=True):

    try:
        n_steps = args[0].shape[0]  # get number of rows from args[0]
    except AttributeError:
        n_steps = None
    gen = zip(*args) if n_steps is None else tqdm(zip(*args), total=n_steps, disable=not verbose)
    csr = vstack([csr_matrix(fn(*args_i)) for args_i in gen])

    return csr
