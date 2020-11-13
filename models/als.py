"""Recommenders based on Alternating Least Squares optimization
"""
import time
from warnings import warn
from collections import defaultdict

import gin
import numpy as np
from scipy.sparse import issparse, hstack, csr_matrix
from tqdm import tqdm

from models.base import BaseRecommender
from util import Clock, prune_global, prune_rows


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
                 alpha=1.0, discordance_weighting=0.0, n_pairs=None,
                 min_weight=1.0, cache_scores=False):
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
        - discordance_weighting (float): if discordance_weighting > 0, ignore alpha and weight
            positives and negatives according to (an estimate of) the number of discordant pos-neg
            pairs they are part of, attenuated with exponent `discordance_weighting` in (0, 1].
            Note: discordance weighting is used during user factor updates only, unless
            `cache_scores` is True.
        - n_pairs (int): maximum number of pos-neg pairs to check for each positive or negative
            when estimating the number of discordant pos-neg pairs
        - cache_scores (bool): if True, use discordance weighting during item factor
            updates as well, using a cached sample of pos and negative item scores for each user
        """
        self.save_weights = save_weights
        self.latent_dim = latent_dim
        self.n_iter = n_iter
        self.l2_reg = l2_reg
        self.init_scale = init_scale
        self.alpha = alpha
        self.discordance_weighting = discordance_weighting
        self.cache_scores = cache_scores
        self.min_weight = min_weight
        self.n_pairs = n_pairs
        self.user_stats = defaultdict(list)
        self.U = None
        self.V = None

        super().__init__(log_dir, batch_size=batch_size)

    def train(self, x_train, y_train, x_val, y_val):
        """Train and evaluate
        """
        if y_train is not None:
            warn("SLIM is unsupervised, y_train will be ignored")

        t1 = time.perf_counter()
        self.U = np.random.randn(x_train.shape[0], self.latent_dim) * self.init_scale
        self.V = np.random.randn(x_train.shape[1], self.latent_dim) * self.init_scale
        for i in range(self.n_iter):
            print(f'Iteration {i + 1}/{self.n_iter}')
            self.user_stats = defaultdict(list)

            wt = map(self.item_weights, x_train, self.U)
            self.U = solve_wols(self.V, yt=x_train, wt=wt, l2_reg=self.l2_reg, verbose=True).T
            wt = map(self.user_weights, x_train.T, self.V)
            self.V = solve_wols(self.U, yt=x_train.T, wt=wt, l2_reg=self.l2_reg, verbose=True).T
            other_metrics = {'mean_wpos': np.mean(self.user_stats['w_pos']),
                             'mean_wneg': np.mean(self.user_stats['w_neg']),
                             'mean_trailing_negatives': np.mean(self.user_stats['trailing_negatives'])}
            self.evaluate(x_val, y_val, step=i+1, other_metrics=other_metrics)
        dt = time.perf_counter() - t1

        metrics = self.evaluate(x_val, y_val, other_metrics={'train_time': dt})
        if self.save_weights and self.logger is not None:
            self.logger.save_weights(None, other={'U': self.U, 'V': self.V})

        return metrics

    def item_weights(self, x, u, print_prob=0.00003, cache_weights=True):
        """
        TODO: Discordance weighting (with cache_scores = False) currently
        appears to cause w_neg < 1 most of the time. Why?
        """
        pos = (x > 0).toarray().flatten()
        neg = np.logical_not(pos)
        if u is not None and self.discordance_weighting:
            n_items = x.shape[1]
            w = np.ones(n_items)
            y = u @ self.V.T
            y_pos, y_neg = y[pos], y[neg]

            y_neg_sample = np.percentile(y_neg, q=np.linspace(0, 100, self.n_pairs))
            discordance = y_pos[None, :] < y_neg_sample[:, None]
            # discordance = discordance * np.sum(neg) / self.n_pairs  # TODO: UNDO
            # w[pos] = (np.sum(discordance, axis=0) + self.min_weight) ** self.discordance_weighting
            w[pos] = self.alpha * np.sum(discordance, axis=0) ** self.discordance_weighting
            if self.cache_scores:
                self.user_stats['y_neg'].append(y_neg_sample)
                self.user_stats['n_neg'].append(np.sum(neg))

            y_pos_sample = np.percentile(y_pos, q=np.linspace(0, 100, self.n_pairs))
            discordance = y_neg[None, :] > y_pos_sample[:, None]
            # discordance = discordance * np.sum(pos) / self.n_pairs  # TODO: UNDO
            # w[neg] = (np.sum(discordance, axis=0) + self.min_weight) ** self.discordance_weighting
            w[neg] = 1.0 + np.sum(discordance, axis=0) ** self.discordance_weighting
            if self.cache_scores and self.neg_disc:
                self.user_stats['n_pos'].append(np.sum(pos))
                self.user_stats['y_pos'].append(y_pos_sample)
        else:
            w = 1.0 + self.alpha * pos
        if cache_weights:
            self.user_stats['w_pos'].append(np.median(w[pos]))
            self.user_stats['w_neg'].append(np.median(w[neg]))
            self.user_stats['trailing_negatives'].append(
                np.sum(w[neg] == self.min_weight ** self.discordance_weighting))
        if np.random.rand() < print_prob:
            rand_pos, rand_neg = np.random.choice(np.sum(pos), 5), np.random.choice(np.sum(neg), 5)
            print(f'for current user, w[neg]{rand_neg.tolist()} = {w[neg][rand_neg]}')
            print(f'for current user, w[pos]{rand_pos.tolist()} = {w[pos][rand_pos]}')

        return w

    def user_weights(self, x, v, print_prob=0.0003):

        pos = (x > 0).toarray().flatten()
        neg = np.logical_not(pos)
        if self.cache_scores:
            n_users = x.shape[1]
            w = np.ones(n_users)
            y = self.U @ v
            y_pos, y_neg = y[pos], y[neg]

            y_neg_sample = np.array(self.user_stats['y_neg'])[pos]
            discordance_sample = y_pos[:, None] < y_neg_sample  # (n_pos, None) x (n_pos, n_pairs)
            discordance = discordance_sample * np.array(self.user_stats['n_neg'])[pos, None] / self.n_pairs
            w[pos] = (np.sum(discordance, axis=1) + self.min_weight) ** self.discordance_weighting

            y_pos_sample = np.array(self.user_stats['y_pos'])[neg]
            discordance_sample = y_neg[:, None] > y_pos_sample  # (n_neg, None) x (n_neg, n_pairs)
            discordance = discordance_sample * np.array(self.user_stats['n_pos'])[neg, None] / self.n_pairs
            w[neg] = (np.sum(discordance, axis=1) + self.min_weight) ** self.discordance_weighting
        else:
            w_neg = np.array(self.user_stats['w_neg']) if self.user_stats['w_neg'] else 1.0
            w_pos = np.array(self.user_stats['w_pos']) if self.user_stats['w_pos'] else 1.0 + self.alpha
            w = w_neg * neg + w_pos * pos
        if np.random.rand() < print_prob:
            rand_pos, rand_neg = np.random.choice(np.sum(pos), 5), np.random.choice(np.sum(neg), 5)
            print(f'for current item, w[neg]{rand_neg.tolist()} = {w[neg][rand_neg]}')
            print(f'for current item, w[pos]{rand_pos.tolist()} = {w[pos][rand_pos]}')

        return w

    def predict(self, x, y=None, n_iter=1, print_prob=0.1):
        u = [None] * x.shape[0]
        for _ in range(n_iter):
            wt = (self.item_weights(x_col, None, cache_weights=False) for x_col in x)
            u = solve_wols(self.V, yt=x, wt=wt, l2_reg=self.l2_reg, verbose=False).T
        y_pred = u @ self.V.T
        if np.random.rand() < print_prob:
            i = np.random.choice(y_pred.shape[0])
            yi = y_pred[i]
            print(f'for current batch, y_pred[{i}]: min, median, max = {np.min(yi), np.median(yi), np.max(yi)}')

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
