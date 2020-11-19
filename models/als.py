"""Recommenders based on Alternating Least Squares optimization
"""
import time
from warnings import warn
from collections import defaultdict

import gin
import numpy as np
from scipy.sparse import issparse, hstack, csr_matrix
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
                 negative_target=0.0, alpha=1.0, discordance_weighting=0.0,
                 rectify_weights=False, cached_weights=False, embeddings_path=None):
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
        - alpha (float): if discordance_weighting == 0, use Hu's W-ALS weighting with this alpha
        - discordance_weighting (float):
        - cache_statistics (bool): if True, use discordance weighting during item factor
            updates as well, using a cached statistics of pos and negative item scores for each user
        """
        self.save_weights = save_weights
        self.latent_dim = latent_dim
        self.n_iter = n_iter
        self.l2_reg = l2_reg
        self.init_scale = init_scale
        self.alpha = alpha
        self.discordance_weighting = discordance_weighting
        self.cached_weights = cached_weights
        self.rectify_weights = rectify_weights
        self.negative_target = negative_target
        self.embeddings_path = embeddings_path

        self.U = None
        self.V = None
        self.user_stats = defaultdict(list)
        self.eps = 0.05 * np.sqrt(2 * np.pi) * self.init_scale ** 2 * np.sqrt(self.latent_dim) / 2
        # eps such that only 5 percent of scores is set to eps or -eps at initialization

        super().__init__(log_dir, batch_size=batch_size)

    def train(self, x_train, y_train, x_val, y_val):
        """Train and evaluate
        """
        if y_train is not None:
            warn("WALS is unsupervised, y_train will be ignored")
        n_users, n_items = x_train.shape

        t1 = time.perf_counter()
        if self.embeddings_path:
            npz_data = np.load(self.embeddings_path)
            self.U, self.V = npz_data['U'], npz_data['V']
        else:
            self.U = np.random.randn(x_train.shape[0], self.latent_dim) * self.init_scale
            self.V = np.random.randn(x_train.shape[1], self.latent_dim) * self.init_scale
        for i in range(self.n_iter):
            print(f'Iteration {i + 1}/{self.n_iter}')
            self.user_stats = defaultdict(list)

            print(f'Updating user vectors...')
            yt = map(self.make_y, x_train)
            wt = map(self.item_weights, x_train, self.U)
            self.U = solve_wols(self.V, yt=yt, wt=wt, l2_reg=self.l2_reg, n_steps=n_users, verbose=True).T

            print(f'Updating item vectors...')
            yt = map(self.make_y, x_train.T)
            wt = map(self.user_weights, x_train.T, self.V)
            self.V = solve_wols(self.U, yt=yt, wt=wt, l2_reg=self.l2_reg, n_steps=n_items, verbose=True).T

            print(f'Evaluating...')
            other_metrics = {'mean_' + k: np.mean(v) for k, v in self.user_stats.items()}
            self.evaluate(x_val, y_val, step=i+1, other_metrics=other_metrics)
            if self.save_weights and self.logger is not None:
                self.logger.save_weights(None, other={'U': self.U, 'V': self.V})

        dt = time.perf_counter() - t1
        metrics = self.evaluate(x_val, y_val, other_metrics={'train_time': dt})

        return metrics

    def item_weights(self, x, u, test_users=False):
        """
        TODO: Discordance weighting (with cache_scores = False) currently
        appears to cause w_neg < 1 most of the time. Why?
        """
        pos = (x > 0).toarray().flatten()
        if u is not None and self.discordance_weighting:
            y = u @ self.V.T
            w = self.compute_weights(y, pos, update_stats=not test_users, print_prob=0.00003)
        else:
            w = 1.0 + self.alpha * pos

        return w

    def user_weights(self, x, v):

        pos = (x > 0).toarray().flatten()
        if self.cached_weights:
            y = self.U @ v
            w = self.compute_weights(y, pos, user_stats=self.user_stats, print_prob=0.0003)
        else:
            w_pos = np.array(self.user_stats.get('w_pos', 1.0 + self.alpha))
            w_neg = np.array(self.user_stats.get('w_neg', 1.0))
            w = w_neg + (w_pos - w_neg) * pos

        return w

    def compute_weights(self, y, pos, user_stats=None, update_stats=False, print_prob=1e-4):

        neg = np.logical_not(pos)
        y_pos, y_neg = y[pos], y[neg]

        # estimate or retrieve parameters of pos, neg distributions
        if user_stats is None:
            n_pos, n_neg = pos.sum(), neg.sum()
            m_pos, s_pos = y_pos.mean(), y_pos.std()
            q1_neg, q2_neg, q3_neg = np.quantile(y_neg, [0.5, 0.625, 0.875])
            m_neg, s_neg = q1_neg, q3_neg - q2_neg
        else:
            keys = ['n_pos', 'n_neg', 'm_pos', 's_pos', 'm_neg', 's_neg']
            n_pos, n_neg, m_pos, s_pos, m_neg, s_neg = (np.array(self.user_stats[k]) for k in keys)
            n_pos, n_neg = n_pos[neg], n_neg[pos]  # see line w[pos] = ...

        p_pos = cauchy(m_neg, s_neg).pdf(y_pos)  # TODO: use numpy (params may be arrays)
        p_neg = norm(m_pos, s_pos).pdf(y_neg)  # TODO: use numpy (params may be arrays)
        w_pos = p_pos / self.apply_eps(1 - y_pos)
        w_neg = p_neg / self.apply_eps(y_neg - self.negative_target)
        w_pos = (n_neg / n_pos * w_pos) ** self.discordance_weighting
        w_neg = w_neg ** self.discordance_weighting

        if update_stats:
            d = {'n_pos': n_pos, 'n_neg': n_neg, 'm_pos': m_pos, 's_pos': s_pos, 'm_neg': m_neg, 's_neg': s_neg,
                 'p_pos': p_pos.mean(), 'p_neg': p_neg.mean(), 'w_pos': w_pos.mean(), 'w_neg': w_neg.mean()}
            if np.random.rand() < print_prob:
                print(d)
            self.update_user_stats(d)

        w = np.ones_like(y)
        w[pos] = w_pos
        w[neg] = w_neg

        return w

    def update_user_stats(self, user_stats):

        for k, v in user_stats.items():
            self.user_stats[k].append(v)

    def apply_eps(self, w):

        if self.rectify_weights:
            w[w < 0] = 0.0
        return np.sign(w) * np.maximum(np.abs(w), self.eps) + self.eps * (w == 0).astype(w.dtype)

    def make_y(self, x):

        if self.negative_target == 0:
            return x
        y = x.toarray() if issparse(x) else x.copy()
        y[y == 0] = self.negative_target

        return y

    def predict(self, x, y=None, n_iter=1):
        n_users = x.shape[0]
        u = [None] * n_users
        for _ in range(n_iter):
            yt = map(self.make_y, x)
            wt = (self.item_weights(x_, u_, test_users=True) for x_, u_ in zip(x, u))
            u = solve_wols(self.V, yt=yt, wt=wt, l2_reg=self.l2_reg, n_steps=n_users, verbose=False).T
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


def solve_wols(x, yt, wt, l2_reg=100, row_nnz=None, n_steps=None, verbose=0):
    """Solve the weighted Ordinary Least Squares problem:
        argmin_B | W * (X B - Y) |^2
    where * denotes element-wise multiplication

    Or equivalently, compute for each column y_i of Y:
        argmin_b_i | W_i (X b_i - y_i) |^2 + reg
    where W_i := diagM(W[:, i]) a diagonal matrix,
    using the closed-form solution
        b_i = (X.T W_i X + l2_reg * I) X.T W_i y

    Use parameter `yt` and `wt` to specify Y.T and W.T. Either of
    these may be a passed as a generator, so that (possibly dense)
    weights and targets can be computed on the fly, saving memory.

    If `row_nnz` is specified, use simple features selection to
    get 'neighbors' for each item i and constrain b_i to zero
    everywhere except in the positions given by neighbors_i---
    allowing for a big reduction in compute when Y has many cols
    If `row_nnz`, also exclude feature x_i when learning y_i,
    as common in SLIM and EASE^R (zero-diagonal constraint).

    Choose verbose == 1 to show a progress bar, verbose == 2 for
    more information (min, max, isfinite) on the learned weights

    NOTE: consider rewriting so that this inner loop refuses to do
    work and returns the existing b when
        np.sum(wt_i) < np.mean(np.sum(wt_i) for wt_i in wt "so far")
    NOTE: function doesn't currently know the existing b
    """
    if n_steps is None:
        try:
            n_steps = yt.shape[0]  # get number of items from Y.T
        except AttributeError:
            n_steps = len(wt)  # if Y.T is a generator, try W.T

    B_list = []
    for i, (wt_i, yt_i) in tqdm(enumerate(zip(wt, yt)), total=n_steps, disable=(verbose == 0)):
        wt_i = wt_i.toarray().flatten() if issparse(wt_i) else np.ravel(wt_i)
        yt_i = yt_i.toarray().flatten() if issparse(yt_i) else np.ravel(yt_i)

        xty = x.T @ (wt_i * yt_i)
        if row_nnz is None:
            xtx = gram_matrix(x, wt_i)
        else:
            # limit cols of x to row_nnz "neighbors" based on x.T @ w @ y
            feat_relevance = np.abs(xty)
            feat_relevance[i] = 0.0  # zero-diag (NOTE: only makes sense for learning similarities)
            feat_selection = np.argpartition(feat_relevance, kth=-row_nnz)[-row_nnz:]
            xtx = gram_matrix(x[:, feat_selection], w=wt_i)
            xty = xty[feat_selection]
        if issparse(xtx):
            xtx = xtx.toarray()

        diag_indices = np.diag_indices(xtx.shape[0])
        xtx[diag_indices] += l2_reg
        xtx_inv = np.linalg.inv(xtx)
        b = xtx_inv @ xty

        if row_nnz is not None:
            ii, jj = feat_selection, np.zeros_like(feat_selection)
            b = csr_matrix((b, (ii, jj)), shape=(x.shape[1], 1))
        B_list.append(b)

    B = np.vstack(B_list).T if row_nnz is None else hstack(B_list).T.tocsr()
    if verbose == 2:
        isfinite_B = np.isfinite(B.data).mean() if issparse(B) else np.isfinite(B).mean()
        print(f'    np.isfinite(B.data).mean() = {isfinite_B}')
        print(f'    np.abs(B).min() = {np.abs(B).min()}')
        print(f'    np.abs(B).max() = {np.abs(B).max()}')

    return B


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
