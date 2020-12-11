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
                 med_alpha=8, min_alpha=-1, max_alpha=26,
                 discordance_weighting=1.0, embeddings_path=None):
        """Matrix factorization recommender with weighted square loss, optimized using
        Alternating Least Squares optimization.

        Implementation focused on arbitrary dense weights. Supports Hu's original weighting scheme
        (negatives @ 1.0 and positives @ 1.0 + alpha) but not efficiently.
        Also supports experimental ranking loss-inducing weighting, in which both positives and
        negatives are weighted according to (an estimate of) the proportion of discordant pos-neg pairs
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
        - med_alpha (float): if beta == 0, use Hu's W-ALS weighting with this alpha
        - cache_statistics (bool): if True, use discordance weighting during item factor
            updates as well, using a cached statistics of pos and negative item scores for each user

        TODO: update params
        """
        self.save_weights = save_weights
        self.latent_dim = latent_dim
        self.n_iter = n_iter
        self.l2_reg = l2_reg
        self.init_scale = init_scale
        self.med_alpha = med_alpha
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.discordance_weighting = discordance_weighting
        self.embeddings_path = embeddings_path
        self.U = None
        self.V = None

        super().__init__(log_dir, batch_size=batch_size)
        assert self.logger is not None

    def train(self, x_train, y_train, x_val, y_val):
        """Train and evaluate
        """
        if y_train is not None:
            warn("WALS is unsupervised, y_train will be ignored")

        t1 = time.perf_counter()
        print('Initializing U, V...')
        x_train = x_train.tocsr()
        n_users, n_items = x_train.shape
        if self.embeddings_path:
            npz_data = np.load(self.embeddings_path)
            self.U, self.V = npz_data['U'], npz_data['V']
        else:
            self.U = np.random.randn(x_train.shape[0], self.latent_dim) * self.init_scale
            self.V = np.random.randn(x_train.shape[1], self.latent_dim) * self.init_scale

        best_ndcg = 0.0
        other_metrics = {}
        print('Evaluating...')
        metrics = self.evaluate(x_val, y_val, step=0, other_metrics=other_metrics)
        for i in range(self.n_iter):
            print(f'Iteration {i + 1}/{self.n_iter}')
            print('Updating user vectors...')
            alpha = self.med_alpha * x_train if self.med_alpha else None
            self.U = solve_ols(self.V, yt=x_train, at=alpha, l2_reg=self.l2_reg, n_cols=n_users).T

            print('Computing weights, y...')
            if self.med_alpha:
                alpha, y = self.compute_wy(x_train)
                other_metrics.update(describe_sparse_rows(alpha, prefix='alpha_'))
                other_metrics.update(describe_sparse_rows(y, prefix='y_'))
                at, yt = alpha.Τ, y.Τ
            else:
                at, yt = None, x_train.T
            print('Updating item vectors...')
            self.V = solve_ols(self.U, yt=yt, at=at, l2_reg=self.l2_reg, n_cols=n_items).T

            print('Evaluating...')
            dt = time.perf_counter() - t1
            metrics = self.evaluate(x_val, y_val, step=self.n_iter, other_metrics={'train_time': dt})
            if self.save_weights and metrics['ndcg'] > best_ndcg:
                self.logger.save_weights(None, other={'V': self.V})
                best_ndcg = metrics['ndcg']

        return metrics

    def compute_wy(self, x_train):

        y_rows = []
        alpha_rows = []
        for x, u in zip(x_train, tqdm(self.U)):
            pos = (x > 0).toarray().flatten()
            neg = np.logical_not(pos)

            uv_pos = u @ self.V[pos].T
            uv_neg = u @ self.V[neg].T
            q1_neg, q2_neg, q3_neg = np.quantile(uv_neg, [0.5, 0.625, 0.875])
            m_neg, s_neg = q1_neg, q3_neg - q2_neg

            f_pos = 1.0 - cauchy(m_neg, s_neg).cdf(uv_pos)  # target loss
            g_pos = -cauchy(m_neg, s_neg).pdf(uv_pos)  # first derivative
            y_pos = uv_pos - 2 * f_pos / g_pos
            w_pos = (g_pos ** 2) / (4 * f_pos)

            # construct sparse alpha, y
            a, y = x.copy(), x.copy()
            a[0, pos] = w_pos - 1.
            y[0, pos] = y_pos

            alpha_rows.append(a)
            y_rows.append(y)

        y = vstack(y_rows)
        alpha = vstack(alpha_rows)
        alpha.data = self.scale_weights(alpha.data + 1.) - 1.

        return alpha, y

    def scale_weights(self, w):

        w = (self.med_alpha + 1.) * (w / np.median(w)) ** self.discordance_weighting  # scale
        w[w > self.max_alpha + 1.] = self.max_alpha + 1.  # cap
        w[w < self.min_alpha + 1.] = self.min_alpha + 1.  # threshold

        return w

    def predict(self, x, y=None):

        alpha = self.med_alpha * x if self.med_alpha else None
        u = solve_ols(self.V, yt=x, at=alpha, l2_reg=self.l2_reg, verbose=False).T
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
        at = (self.beta + x_col.toarray().flatten() * self.alpha - 1. for x_col in x_train.T)
        self.S = solve_ols(x_train, yt=x_train.T, wt=at, l2_reg=self.l2_reg, row_nnz=self.row_nnz)

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


def solve_ols(x, yt, at=None, l2_reg=100., verbose=False, n_cols=None):
    """Solve the weighted Ordinary Least Squares problem:
        argmin_B | W * (X B - Y) |^2
    where * denotes element-wise multiplication.

    Or equivalently, compute for each column y_i of Y:
        argmin_b_i | w_i (X b_i - y_i) |^2 + reg
    using the closed form
        b_i = (X.T W_i X + l2_reg * I) X.T W_i y
    where W_i := diagM(W[:, i]) a diagonal matrix

    Use parameter `yt` and `at` to specify Y.T and W.T = at + 1
    Either of these may be a passed as a generator.
    """
    try:
        n_cols = yt.shape[0]
    except AttributeError:
        pass

    xtx = x.T @ x
    if at is None:
        xty = x.T @ yt.T
        if issparse(xtx):
            xtx = xtx.toarray()
        if issparse(xty):
            xty = xty.toarray()
        diag_indices = np.diag_indices(xtx.shape[0])
        xtx[diag_indices] += l2_reg
        xtx_inv = np.linalg.inv(xtx)
        b = xtx_inv @ xty
    else:
        xtx = x.T @ x
        gen = tqdm(zip(yt.tocsr(), at.tocsr()), total=n_cols, disable=not verbose)
        cols = [solve_wols_col(x, yt_i, at_i, l2_reg=l2_reg, xtx=xtx) for yt_i, at_i in gen]
        b = np.hstack(cols)

    return b


def solve_wols_col(x, yt_i, at_i, l2_reg, xtx=None):
    """Solve the weighted Ordinary Least Squares problem
        argmin_b | w (X b - y) |^2 + reg
    for one target.
    """
    a_i, y_i = at_i.T, yt_i.T

    if xtx is None:
        xtx = x.T @ x
    ax = a_i.multiply(x)
    xtx = xtx + x.T @ ax
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


def describe_sparse_rows(x, prefix=''):
    """Describe the rows of a sparse matrix.
    Includes explicit zeros.
    """
    min_nz = np.mean([np.min(row.data) for row in x])
    max_nz = np.mean([np.max(row.data) for row in x])
    mean_nz = np.mean([np.mean(row.data) for row in x])
    median_nz = np.mean([np.median(row.data) for row in x])
    allzero = np.mean([np.all(row.data == 0) for row in x])
    d = {f'{prefix}min': min_nz, f'{prefix}max': max_nz,
         f'{prefix}mean': mean_nz, f'{prefix}median': median_nz,
         f'{prefix}mean': mean_nz, f'{prefix}median': median_nz,
         f'{prefix}zero': allzero}

    return d
