"""Recommenders based on Alternating Least Squares optimization
"""
import time
from warnings import warn
from functools import partial

import gin
import numpy as np
from tqdm import tqdm
from scipy.sparse import issparse, vstack
from scipy.stats import cauchy, norm, logistic
from scipy.special import expit

from models.base import BaseRecommender
from extensions import least_squares_cg, rank_least_squares_cg
# from opt import least_squares_cg, rank_least_squares_cg
from util import Clock, gen_batch_inds, prune_global, prune_rows


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
                 alpha=8., weight_scale=1., min_weight=0, max_weight=50., min_target=None, max_target=None,
                 dynamic_weights=False, dynamic_targets=False, conjugate_gradient=True,
                 embeddings_path=None, score_dist=None, loss=None, cg_steps=3):
        """Matrix factorization recommender with weighted square loss, optimized using
        Alternating Least Squares optimization.

        Implementation focused on arbitrary dense weights. Supports Hu's original weighting scheme
        (negatives @ 1.0 and positives @ 1.0 + alpha) but not efficiently.
        Also supports experimental ranking loss-inducing weighting, in which both positives and
        negatives are weighted according to (an estimate of) the proportion of discordant pos-neg pairs
        they are part of.

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
        - cache_statistics (bool): if True, use discordance weighting during item factor
            updates as well, using a cached statistics of pos and negative item scores for each user

        TODO: update params
        """
        self.save_weights = save_weights
        self.latent_dim = latent_dim
        self.n_iter = n_iter
        self.l2_reg = l2_reg
        self.init_scale = init_scale
        self.embeddings_path = embeddings_path
        self.dynamic_weights = dynamic_weights
        self.dynamic_targets = dynamic_targets
        self.alpha = alpha
        self.weight_scale = weight_scale
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.score_dist = score_dist
        self.loss = loss
        self.min_target = min_target
        self.max_target = max_target
        self.conjugate_gradient = conjugate_gradient
        self.cg_steps = cg_steps
        self.U = None
        self.V = None

        super().__init__(log_dir, batch_size=batch_size)
        assert self.logger is not None

    def train(self, x_train, y_train, x_val, y_val):
        """Train and evaluate

        TODO: instead of conditionals in train loop, subclass WALSRecommender for CG case?
        """
        t1 = time.perf_counter()

        x_train = x_train.tocsr()
        n_users, n_items = x_train.shape
        self.init_embeddings(n_users, n_items)

        best_ndcg, other_metrics = 0.0, {}
        fast_track = self.conjugate_gradient
        conf_u = x_train.multiply(self.alpha).astype(np.float32).tocsr() if self.alpha else None
        conf_v = conf_u.T.tocsr()
        for i in tqdm(range(self.n_iter), disable=not fast_track):
            if not fast_track:
                other_metrics['train_time'] = time.perf_counter() - t1
                metrics = self.evaluate(x_val, y_val, best_ndcg, other_metrics, step=self.n_iter)
                print(f'Iteration {i + 1}/{self.n_iter}')

            if self.conjugate_gradient:
                if self.score_dist == 'cauchy' and self.loss == 'hinge':
                    raise NotImplementedError()
                # update U (in-place)
                least_squares_cg(conf_u, self.U, self.V, regularization=self.l2_reg, cg_steps=self.cg_steps)
                # update V (in-place)
                if self.dynamic_weights and self.dynamic_targets:
                    mu, sigma = self.compute_mu_sigma(x_train)
                    rank_least_squares_cg(conf_v, self.V, self.U, mu, sigma, self.l2_reg,
                                          min_weight=self.min_weight, max_weight=self.max_weight,
                                          min_target=self.min_target, max_target=self.max_target,
                                          cg_steps=self.cg_steps, cauchy=(self.score_dist == 'cauchy'),
                                          hinge_loss=(self.loss == 'hinge'))
                elif not self.dynamic_weights and not self.dynamic_targets:
                    least_squares_cg(conf_v, self.V, self.U, regularization=self.l2_reg, cg_steps=self.cg_steps)
                else:
                    raise NotImplementedError()
                continue

            print('Updating user vectors...')
            self.U = solve_ols(self.V, yt=x_train, at=conf_v, l2_reg=self.l2_reg, n_cols=n_users).T

            print('Computing weights, y...')
            conf_vt, y = self.compute_wy(x_train)
            conf_v, yt = None, y.T.tocsr()
            if conf_v is not None:
                other_metrics.update(describe_sparse_rows(conf_vt, prefix='alpha_'))
                other_metrics.update(describe_sparse_rows(y, prefix='y_'))
                conf_v = conf_vt.T.tocsr()
            print('Updating item vectors...')
            self.V = solve_ols(self.U, yt=yt, at=conf_v, l2_reg=self.l2_reg, n_cols=n_items).T

        other_metrics['train_time'] = time.perf_counter() - t1
        metrics = self.evaluate(x_val, y_val, best_ndcg, other_metrics, step=self.n_iter)

        return metrics

    def evaluate(self, x, y, best_ndcg=0.0, other_metrics={}, step=None, test=False):

        print('Evaluating...')
        metrics = super().evaluate(x, y, other_metrics=other_metrics, step=step, test=test)
        if not test and self.save_weights and metrics['ndcg'] > best_ndcg:
            self.logger.save_weights(None, other={'V': self.V})
            best_ndcg = metrics['ndcg']

        return metrics

    def init_embeddings(self, n_users, n_items):

        if self.embeddings_path:
            npz_data = np.load(self.embeddings_path)
            self.U, self.V = npz_data.get('U', None), npz_data['V']
            assert self.V.shape[1] == self.latent_dim, "Embeddings loaded but latent_dim doesn't match."
        else:
            self.V = np.random.randn(n_items, self.latent_dim) * self.init_scale
        if self.U is None and self.conjugate_gradient:
            self.U = np.random.randn(n_users, self.latent_dim) * self.init_scale

        if self.conjugate_gradient:
            self.U = self.U.astype(np.float32)
            self.V = self.V.astype(np.float32)

    def compute_mu_sigma(self, x_train, n_sample=200):

        n_users, n_items = x_train.shape
        mu, sigma = np.zeros(n_users), np.zeros(n_users)
        for inds in gen_batch_inds(n_users, progress_bar=False):
            sample = np.random.choice(n_items, n_sample)
            scores = self.U[inds] @ self.V[sample].T
            if self.score_dist == 'cauchy':
                q1_neg, q2_neg, q3_neg = np.quantile(scores, [0.5, 0.625, 0.875], axis=1)
                mu[inds], sigma[inds] = q1_neg, q3_neg - q2_neg
            elif self.score_dist == 'logistic':
                mu[inds] = np.mean(scores, axis=1)
                sigma[inds] = np.sqrt(3) / np.pi * np.std(sample, axis=1)
            else:
                raise NotImplementedError()

        return mu, sigma

    def compute_wy(self, x_train, u=None, verbose=True):

        # don't waste time computing scores uv when we only need static w, y
        if not self.dynamic_weights and not self.dynamic_targets:
            alpha = x_train.multiply(self.alpha)
            y = x_train
            return y, x_train

        y_rows = []
        alpha_rows = []
        u = self.U if u is None else u
        for inds in gen_batch_inds(x_train.shape[0], self.batch_size, progress_bar=verbose):
            x_batch = x_train[inds]
            uv_batch = u[inds] @ self.V.T

            for x, uv in zip(x_batch, uv_batch):
                pos = (x > 0).toarray().flatten()
                neg = np.logical_not(pos)
                uv_pos, uv_neg = uv[pos], uv[neg]
                w_pos, y_pos = self.fit_wy(uv_pos, uv_neg)

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

    def fit_wy(self, uv_pos, uv_neg):
        """Given a set of positive scores `uv_pos` and negatives scores `uv_neg`,
        compute weights and targets for positive items such that the resulting
        square loss approximates a certain ranking loss (see get_loss).

        TODO: make work for batches of uv, x?
        """
        loss, gradient = self.get_loss(uv_pos, uv_neg)
        if self.dynamic_weights and self.dynamic_targets:
            w_pos = (gradient ** 2) / (4 * loss)
            y_pos = uv_pos - 2 * loss / gradient
        elif self.dynamic_targets:
            y_pos = uv_pos - gradient / 2.
            w_pos = np.ones_like(uv_pos)
        elif self.dynamic_weights:
            w_pos = gradient / (uv_pos - 1) / 2
            y_pos = np.ones_like(uv_pos)
        else:
            raise NotImplementedError()

        if self.min_y is not None:
            y_pos = np.maximum(y_pos, self.min_y)
        if self.max_y is not None:
            y_pos = np.minimum(y_pos, self.max_y)

        return w_pos, y_pos

    def get_loss(self, uv_pos, uv_neg):
        """Given a set of positive scores `uv_pos` and negatives scores `uv_neg`,
        compute target loss and its first derivative for each positive,
        assuming negative scores follow a certain distribution (see get_density).
        """
        if self.loss == 'discordance':
            pdf, cdf, _ = self.get_density(uv_neg)
            f_pos = 1. - cdf(uv_pos)
            g_pos = -pdf(uv_pos)
        elif self.loss == 'hinge':
            margin = 1.
            pdf, cdf, ccdf = self.get_density(uv_neg + margin)
            f_pos = ccdf(-uv_pos)  # TODO: not used
            g_pos = -cdf(uv_pos)
        else:
            raise NotImplementedError()

        return f_pos, g_pos

    def get_density(self, sample):
        """TODO: subclass each dist instead?
        """
        ccdf = None  # if loss (e.g. hinge) requires \int cdf, use logistic
        if self.score_dist == 'cauchy':
            q1_neg, q2_neg, q3_neg = np.quantile(sample, [0.5, 0.625, 0.875], axis=-1)
            dist = cauchy(q1_neg, q3_neg - q2_neg)
        elif self.score_dist == 'normal':
            m_neg, s_neg = np.mean(sample, axis=-1), np.std(sample, axis=-1)
            dist = norm(m_neg, s_neg)
        elif self.score_dist == 'logistic':
            m_neg, s_neg = np.mean(sample, axis=-1), np.sqrt(3) / np.pi * np.std(sample, axis=-1)
            dist = logistic(m_neg, s_neg)
            ccdf = partial(softplus, m=m_neg, s=s_neg)
        else:
            raise NotImplementedError()

        return dist.pdf, dist.cdf, ccdf

    def scale_weights(self, w):

        w = (self.alpha + 1.) * (w / np.nanmedian(w))
        w[w > self.max_weight] = self.max_weight  # cap
        w[w < self.min_weight] = self.min_weight  # threshold
        w[np.isnan(w)] = self.max_alpha + 1.  # cap where nan

        return w

    def predict(self, x, y=None):

        yt, at = x, self.alpha * x if self.alpha else None
        u = solve_ols(self.V, yt=yt, at=at, l2_reg=self.l2_reg, n_cols=x.shape[0], verbose=False).T
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
        self.beta = beta  # not the same beta as in WALSRecommender
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


def solve_ols(x, yt, at=None, l2_reg=100., verbose=True, n_cols=None):
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

    if at is None:
        xtx = x.T @ x
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
    if np.all(b == 0):
        print('DEBUG: all-zero embedding in solve_wols_col')
        import pdb
        pdb.set_trace()

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
    if x is None:
        return {}
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


def softplus(x, m, s):

    z = (x - m) / s
    return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0)


def sigmoid(x, m, s):

    return expit((x - m) / s)
