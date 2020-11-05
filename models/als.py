"""global-local SLIM/EASE variants, such as models with item clusters and user clusters
"""
import time
from warnings import warn

import gin
import numpy as np
from scipy.sparse import issparse, hstack, csr_matrix
from tqdm import tqdm

from models.base import BaseRecommender
from util import Clock, prune_global, gen_batch_inds, prune_rows, load_weights


@gin.configurable
class SpLoRecommender(BaseRecommender):

    def __init__(self, log_dir, batch_size=100, weights_path=None, save_weights=True,
                 latent_dim=200, uv_l2_reg=0.01, uv_updates=3, init_scale=0.1,
                 update_s=True, l2_reg=100.0, row_nnz=None, target_density=1.0):
        """'Sparse + Low-rank' recommender. Models:
            X ~ U @ V.T + X @ S
        """
        self.save_weights = save_weights
        self.latent_dim = latent_dim
        self.uv_l2_reg = uv_l2_reg
        self.uv_updates = uv_updates
        self.init_scale = init_scale
        self.update_s = update_s
        self.l2_reg = l2_reg
        self.row_nnz = row_nnz
        self.target_density = target_density

        if weights_path:
            self.S, other = load_weights(weights_path)
            self.V = other['V']
            self.U = None
        else:
            self.S = None
            self.U = None
            self.V = None

        super().__init__(log_dir, batch_size=batch_size)

    def train(self, x_train, y_train, x_val, y_val):
        """Train and evaluate

        TODO: one more update for U just before S, to ensure we use the same U @ V.T as we'd have at prediction time?
        TODO: weights W = X so that S focuses on re-ranking? Will be slower, but worth an experiment
        """
        verbose = self.logger.verbose if self.logger else False
        clock = Clock(verbose=verbose)
        t1 = time.perf_counter()

        if y_train is not None:
            warn("SLIM is unsupervised, y_train will be ignored")
        n_users, n_items = x_train.shape

        if self.S is None:
            self.S = csr_matrix((n_items, n_items))
        if self.V is None:
            self.V = np.random.randn(n_items, self.latent_dim) * self.init_scale
        self.U = np.zeros((n_users, self.latent_dim))  # in case uv_updates = 0

        # update U, V
        for _ in range(self.uv_updates):
            clock.interval('Solving for U')
            batches = gen_batch_inds(n_users, batch_size=self.batch_size)
            y_batched = (x_train[inds].toarray() - x_train[inds].toarray() @ self.S for inds in batches)
            u_batched = [solve_ols(self.V, y_batch.T, l2_reg=self.uv_l2_reg).T for y_batch in y_batched]
            self.U = np.vstack(u_batched)
            clock.interval('Solving for V')
            batches = gen_batch_inds(n_items, batch_size=self.batch_size)
            y_batched = (x_train[:, inds].toarray() - x_train @ self.S[:, inds].toarray() for inds in batches)
            v_batched = [solve_ols(self.U, y_batch, l2_reg=self.uv_l2_reg).T for y_batch in y_batched]
            self.V = np.vstack(v_batched)
            clock.interval('Evaluating')
            metrics = self.evaluate(x_val, y_val)

        # update S
        if self.update_s and self.row_nnz is None:
            clock.interval('Computing XtX')
            xtx = x_train.T @ x_train
            clock.interval('Computing XtX^-1')
            diag_indices = np.diag_indices(xtx.shape[0])
            xtx_reg = xtx.copy()
            xtx_reg[diag_indices] += self.l2_reg
            xtx_inv = np.linalg.inv(xtx_reg.toarray())
            clock.interval('Computing XtY')
            xtu = x_train.T @ self.U
            xty = xtx - xtu @ self.V.T  # y = (x - u v.T) => x.T y = x.T x - X.T u v.T
            clock.interval('Computing XtX^-1 @ XtY')
            b = xtx_inv @ xty
            clock.interval('Computing S (zero-diag)')
            self.S = b - xtx_inv * np.diag(b) / np.diag(xtx_inv)  # zero-diag, see Steck 2019
            clock.interval('Pruning S')
            self.S = prune_global(self.S, target_density=self.target_density).tocsc()
        elif self.update_s:
            clock.interval('Solving for S, using sparse approximation')
            yt = (x.toarray().flatten() - self.U @ v for x, v in zip(x_train.T, self.V))
            wt = np.ones(x_train.shape[1])
            self.S = solve_wols(x_train, yt=yt, wt=wt, l2_reg=self.l2_reg, row_nnz=self.row_nnz)
            if self.target_density < 1.0:
                self.S = prune_global(self.S, target_density=self.target_density)
            self.S = self.S.tocsc()

        dt = time.perf_counter() - t1
        if self.save_weights and self.logger is not None:
            self.logger.save_weights(self.S, other={'V': self.V})

        density = self.S.size / np.prod(self.S.shape)
        metrics = self.evaluate(x_val, y_val, other_metrics={'train_time': dt, 'weight_density': density})

        return metrics

    def predict(self, x, y=None,):
        """Predict scores
        """
        u = solve_ols(self.V, x.toarray().T, l2_reg=self.uv_l2_reg).T
        y_pred = x @ self.S + u @ self.V.T

        return y_pred, np.nan


@gin.configurable
class ALSRecommender(BaseRecommender):

    def __init__(self, log_dir, batch_size=70, save_weights=True,
                 latent_dim=50, n_iter=20, l2_reg=10., init_scale=0.1):
        """Gobal-local EASE v0
        - low-rank and high-rank joint linear model
        - R ~ U @ V + R @ S
        - no clustering
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
        NOTE: if you get a dimension mismatch here for the final multiplication,
        you probably ended up with S (or another var) a matrix (instead of array)
        """
        u = solve_ols(self.V, x.T, l2_reg=self.l2_reg, verbose=False).T
        y_pred = u @ self.V.T

        return y_pred, np.nan


@gin.configurable
class WALSRecommender(BaseRecommender):

    def __init__(self, log_dir, batch_size=70, save_weights=True,
                 latent_dim=50, n_iter=20, l2_reg=10., init_scale=0.1,
                 imposter_weight=0.0, alpha=1.0, beta=1.0,
                 temperature=None):
        self.save_weights = save_weights
        self.latent_dim = latent_dim
        self.n_iter = n_iter
        self.l2_reg = l2_reg
        self.init_scale = init_scale
        self.imposter_weight = imposter_weight
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
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
            wt = map(self.row_weights, x_train, self.U)
            self.U = solve_wols(self.V, yt=x_train, wt=wt, l2_reg=self.l2_reg, verbose=True).T
            wt = map(self.col_weights, x_train.T, self.V)
            self.V = solve_wols(self.U, yt=x_train.T, wt=wt, l2_reg=self.l2_reg, verbose=True).T
            self.evaluate(x_val, y_val, step=i+1)
        dt = time.perf_counter() - t1

        metrics = self.evaluate(x_val, y_val, other_metrics={'train_time': dt})
        if self.save_weights and self.logger is not None:
            self.logger.save_weights(None, other={'U': self.U, 'V': self.V})

        return metrics

    def row_weights(self, x, e, col_weights=False, asym=False):

        q_pos = (x > 0).toarray().flatten()
        q_neg = np.logical_not(q_pos)
        imp = 0.0
        if self.imposter_weight or self.temperature is not None:
            g = self.U @ e if col_weights else e @ self.V.T
        if self.imposter_weight:
            thr = np.median(g[q_pos]) if np.sum(q_pos) else 0.0
            imp = q_neg * (g >= thr)
        if self.temperature is not None:
            q = np.exp(-g / self.temperature)
            q_neg = q_neg.astype(np.float) * q / np.mean(q)
        if asym:
            q_pos[g > 1.0] = 0.0  # TODO: test (maybe even simplify everything first)

        return self.beta * q_neg + (1.0 + self.alpha) * q_pos + self.imposter_weight * imp

    def col_weights(self, x, e):

        return self.row_weights(x, e, col_weights=True)

    def predict(self, x, y=None):
        """Predict scores
        NOTE: if you get a dimension mismatch here for the final multiplication,
        you probably ended up with S (or another var) a matrix (instead of array)
        """
        wt = (1.0 + x_col.toarray().flatten() * self.alpha for x_col in x)
        u = solve_wols(self.V, yt=x, wt=wt, l2_reg=self.l2_reg, verbose=False).T
        y_pred = u @ self.V.T
        if np.random.rand() < 0.1:
            # one in ten validation batches, print some numbers
            row = y_pred[np.random.choice(y_pred.shape[0])]
            print(f'min, median, max, mean, std = {np.min(row), np.median(row), np.max(row), np.mean(row), np.std(row)}')

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
