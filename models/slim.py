
import time
from warnings import warn

import gin
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, issparse, eye

from models.base import BaseRecommender
from util import prune_global, load_weights, get_pruning_threshold, sparse_info


@gin.configurable
class LinearRecommender(BaseRecommender):

    def __init__(self, log_dir, reg=500, density=1.0, batch_size=100, save_weights=True,
                 incremental=False, head_tail=False):
        """Linear recommender based on Harald Steck's closed form variant [1]
        of Sparse Linear Methods (SLIM) [2].

        [1] Harald Steck, Embarrassingly shallow auto-encoders. WWW 2019
        https://arxiv.org/pdf/1905.03375.pdf

        [3] Xia Ning and George Karypis, SLIM: Sparse Linear Methods for
        Top-N Recommender Systems. ICDM 2011
        http://glaros.dtc.umn.edu/gkhome/node/774
        """
        self.reg = reg
        self.density = density
        self.save_weights = save_weights
        self.incremental = incremental
        self.head_tail = head_tail
        self.weights = None
        super().__init__(log_dir, batch_size=batch_size)

    def train(self, x_train, y_train, x_val, y_val):
        """Train and evaluate"""
        if y_train is not None:
            warn("SLIM is unsupervised, y_train will be ignored")

        t1 = time.perf_counter()
        if self.incremental and self.head_tail:
            self.weights = block_slim(x_train, l2_reg=self.reg, target_density=self.density)
            if self.density < 1.0:
                self.weights = prune_global(self.weights, self.density)
        elif self.incremental:
            self.weights = incremental_slim(x_train, batch_size=self.batch_size, target_density=self.density)
            # self.weights = naive_incremental_slim(x_train, batch_size=self.batch_size, target_density=self.density)
        elif self.head_tail:
            self.weights = head_tail_slim(x_train, l2_reg=self.reg, target_density=self.density)
            if self.density < 1.0:
                self.weights = prune_global(self.weights, self.density)
        else:
            gramm = (x_train.T @ x_train).toarray()
            self.weights = closed_form_slim(gramm, l2_reg=self.reg)
            if self.density < 1.0:
                self.weights = prune_global(self.weights, self.density)
        dt = time.perf_counter() - t1
        if self.logger is not None:
            self.logger.log_config(gin.operative_config_str())
            if self.save_weights:
                self.logger.save_weights(self.weights)

        print('Evaluating...')
        metrics = self.evaluate(x_val, y_val)
        density = self.weights.size / np.prod(self.weights.shape)
        metrics['train_time'] = dt
        metrics['weight_density'] = density

        return metrics

    def predict(self, x, y=None):
        """Predict scores"""
        y_pred = x @ self.weights

        return y_pred, np.nan


@gin.configurable
class WoodburyRecommender(LinearRecommender):

    def __init__(self, log_dir, reg=500, density=1.0, batch_size=100,
                 approx=False, n_models=1,
                 save_weights=True):

        self.approx = approx
        self.n_models = n_models
        super().__init__(log_dir, reg=reg, density=density, batch_size=batch_size, save_weights=save_weights)

    def train(self, x_train, y_train, x_val, y_val):
        """Train and evaluate"""
        if y_train is not None:
            warn("SLIM is unsupervised, y_train will be ignored")

        t0 = time.perf_counter()
        if self.approx and self.n_models > 1:
            self.weights = woodbury_ensemble(x_train, n_models=self.n_models,
                                             batch_size=self.batch_size,
                                             target_density=self.density)
        elif self.n_models > 1:
            raise NotImplementedError('approx=True and n_models>1 not supported')
        elif self.approx:
            self.weights = woodbury_slim_approx(x_train, batch_size=self.batch_size,
                                                target_density=self.density)
        else:
            self.weights = woodbury_slim(x_train, batch_size=self.batch_size,
                                         target_density=self.density)

        dt = time.perf_counter() - t0
        if self.logger is not None: 
            self.logger.log_config(gin.operative_config_str())
            if self.save_weights:
                self.logger.save_weights(self.weights)

        print('Evaluating...')
        metrics = self.evaluate(x_val, y_val)
        density = self.weights.size / np.prod(self.weights.shape)
        metrics['train_time'] = dt
        metrics['weight_density'] = density

        return metrics


@gin.configurable
class LinearRecommenderFromFile(BaseRecommender):

    def __init__(self, log_dir=None, path=None, batch_size=100):
        """Pretrained linear recommender.

        Given a file containing 2d weights and optional 1d biases,
        predict item scores.
        """
        weights, biases = load_weights(path)
        if biases is None:
            print('LinearRecommenderFromFile: no intercepts loaded...')
            biases = np.zeros((1, weights.shape[1]))
        else:
            biases = biases.reshape(1, -1)
        self.weights = weights
        self.biases = biases
        super().__init__(log_dir=log_dir, batch_size=batch_size)

    def predict(self, x, y=None):
        """Predict scores"""
        y_pred = np.asarray(x @ self.weights + self.biases)

        return y_pred, np.nan


def closed_form_slim(gramm, l2_reg=500):

    print(f'gramm.max() = {gramm.max()}')
    raise ValueError()
    if issparse(gramm):
        gramm = gramm.toarray()
    print(f'  computing slim weights of shape {gramm.shape}')
    diag_indices = np.diag_indices(gramm.shape[0])
    gramm[diag_indices] += l2_reg
    inv_gramm = np.linalg.inv(gramm)
    weights = inv_gramm / (-np.diag(inv_gramm))
    weights[diag_indices] = 0.0

    return weights


def distilled_slim(gramm, fn_name='closed_form', n_distillations=1, l2_reg=500):

    slim_fns = {'closed_form': closed_form_slim}
    
    slim_fn = slim_fns[fn_name]
    weights = slim_fn(gramm, l2_reg=l2_reg)
    for i in range(n_distillations):
        print('DISTILL ITERATION {i}...')
        gramm = weights @ gramm @ weights
        weights = slim_fn(weights, l2_reg=l2_reg)

    return weights


def head_tail_slim(x, l2_reg=500, target_density=0.01, gramm_mass_thr=0.1):

    n_users, n_items = x.shape
    t = time.perf_counter()

    print('- computing gramm matrix...')
    gramm = x.T @ x
    B = csr_matrix(gramm.shape)
    print(f'  elapsed: {time.perf_counter() - t}')
    t = time.perf_counter()

    print('- computing head and tail...')
    head, tail = get_head_tail(gramm, gramm_mass_thr)
    print(f'  elapsed: {time.perf_counter() - t}')
    t = time.perf_counter()

    G_head = gramm[head][:, head].toarray()
    print(f'- computing head weights - shape {G_head.shape}...')
    diag_indices = np.diag_indices(len(head))
    G_head[diag_indices] += l2_reg
    P_head = np.linalg.inv(G_head)
    B_head = P_head / (-np.diag(P_head))
    B_head[diag_indices] = 0.0
    block_density = min(1.0, target_density * np.prod(B.shape) / np.prod(B_head.shape))
    B_head = prune_global(B_head, target_density=block_density).tocoo()
    B += csr_matrix((B_head.data, (head[B_head.row], head[B_head.col])), shape=B.shape)
    print(f'  elapsed: {time.perf_counter() - t}')
    t = time.perf_counter()

    G_head_tail = gramm[head][:, tail].toarray()
    print(f'- computing head x tail weights - shape {G_head_tail.shape}...')
    B_tail = P_head @ G_head_tail
    block_density = min(1.0, target_density * np.prod(B.shape) / np.prod(B_tail.shape))
    B_tail = prune_global(B_tail, target_density=block_density).tocoo()
    dB = csr_matrix((B_tail.data, (head[B_tail.row], tail[B_tail.col])), shape=B.shape)
    B += dB + dB.T
    print(f'  elapsed: {time.perf_counter() - t}')

    return B


def get_head_tail(gramm, thr=0.1):

    col_sum = np.ravel(gramm.sum(axis=0))
    items = np.argsort(-col_sum)

    tot_gramm = gramm.sum()
    cum_gramm = np.cumsum([gramm[item, item] + 2 * gramm[items[i:], item].sum() for i, item in enumerate(items)])
    i = np.nonzero(cum_gramm > (1 - thr) * tot_gramm)[0][0]

    return items[:i], items[i:]


def block_slim(x, l2_reg=500, target_density=0.01, block_size=3000, mass_thr=0.00):
    """Compute SLIM weights using Steck's closed form solution,
    but in (large) batches.
    """
    t = time.perf_counter()
    print('- computing gramm matrix...')
    gramm = x.T @ x
    gramm_edges = gramm.copy()
    gramm_edges.data = np.ones_like(gramm_edges.data, dtype=int)
    row_degrees = np.ravel(gramm_edges.sum(axis=0))
    tail = np.argsort(-row_degrees)

    B = csr_matrix(gramm.shape)
    thr = 0.0
    gramm_mass = gramm.sum()
    remaining_mass = gramm_mass
    while remaining_mass / gramm_mass > mass_thr:
        head, tail = tail[:block_size], tail[block_size:]
        
        G_head = gramm[head][:, head].toarray()
        print(f'  elapsed: {time.perf_counter() - t}...')
        t = time.perf_counter()
        print(f'- computing weights block of shape {G_head.shape}...')
        diag_indices = np.diag_indices(len(head))
        G_head[diag_indices] += l2_reg
        P_head = np.linalg.inv(G_head)
        B_head = P_head / (-np.diag(P_head))
        B_head[diag_indices] = 0.0
        print(f'  elapsed: {time.perf_counter() - t}...')
        t = time.perf_counter()
        print('- updating weights...')
        B_head[np.abs(B_head) < thr] = 0
        B_head = coo_matrix(B_head)
        dB = csr_matrix((B_head.data, (head[B_head.row], head[B_head.col])), shape=B.shape)
        B += dB
        print(f'  elapsed: {time.perf_counter() - t}...')
        t = time.perf_counter()
        print('- pruning weights')

        thr = get_pruning_threshold(B, target_density=target_density)
        B.data[np.abs(B.data) < thr] = 0.0
        B.eliminate_zeros()

        G_tail = gramm[tail][:, tail]
        remaining_mass = G_tail.sum()
        print(f'  remaining gramm mass = {remaining_mass / gramm_mass}...')

        if len(tail) == 0:
            continue

        G_head_tail = gramm[head][:, tail].toarray()
        print(f'  elapsed: {time.perf_counter() - t}...')
        t = time.perf_counter()
        print(f'- computing weights block of shape {G_head_tail.shape}...')
        B_head_tail = P_head @ G_head_tail
        print(f'  elapsed: {time.perf_counter() - t}...')
        t = time.perf_counter()
        print('- updating weights...')
        B_head_tail[np.abs(B_head_tail) < thr] = 0
        B_head_tail = coo_matrix(B_head_tail)
        dB = csr_matrix((B_head_tail.data, (head[B_head_tail.row], tail[B_head_tail.col])), shape=B.shape)
        B = B + dB + dB.T

    return B


def woodbury_slim(x, l2_reg=500, batch_size=100, drop_inactive_items=False, target_density=1.0):
    """Closed-form SLIM with incremental computation of the inverse gramm matrix

    we use the Woodbury matrix identity:
        inv(A + U C V) = inv(A) - inv(A) U inv(inv(c) + V inv(A) U) V inv(A)
    or if P is the inverse of gramm matrix G and if X is a batch of NEW users:
        inv(G + X.T X) = P X.T inv(I + X P X.T) X P
    and avoid some multiplication overhead by
    - precomputing X P and P X.T = (X P).T
    - where possible, drop rows and cols for items not involved in the update
    """
    n_users, n_items = x.shape
    inv_gramm = np.eye(n_items) / l2_reg
    for x_batch in gen_batches(x, batch_size=batch_size):
        print('  small multiplications...')
        if drop_inactive_items:
            inds = np.unique(x_batch.tocoo().col)
            x_batch = x_batch[:, inds]
            XP = x_batch @ inv_gramm[inds]
            XPX = XP[:, inds] @ x_batch.T
        else:
            XP = x_batch @ inv_gramm
            XPX = XP @ x_batch.T
        print(f'  inverse of rank {x_batch.shape[0]}...')
        B = np.linalg.inv(np.eye(x_batch.shape[0]) + XPX)
        print('  big multiplication...')
        update = XP.T @ B @ XP
        print('  update...')
        inv_gramm -= update

    diag_indices = np.diag_indices(n_items)
    weights = - inv_gramm / inv_gramm.diagonal()
    weights[diag_indices] = 0.0
    if target_density < 1.0:
        weights = prune_global(weights, target_density=target_density)

    return weights


def woodbury_slim_approx(x, l2_reg=500, batch_size=100, target_density=1.0,
                         extra_reg=500, strict_sparsity=True):
    """Closed-form SLIM with incremental computation of the inverse gramm matrix,
    and a useful approximation.

    TODO stabilize this computation: diverges for some batch sizes / seeds
    - additional regularization (diag += reg) to bound the lowest eigenvalue away from zero?
    - found that leaving the diagonal untouched in the pruning step is not enough
    - may be inevitable after 100+ matrix inversions

    we use the Woodbury matrix identity:
        inv(A + U C V) = inv(A) - inv(A) U inv(inv(c) + V inv(A) U) V inv(A)
    or if P is the inverse of gramm matrix G and if X is a batch of NEW users:
        inv(G + X.T X) = P - P X.T inv(I + X P X.T) X P
    we also
    - precompute X P which also gets us P X.T = (X P).T
    - avoid some multiplication overhead by dropping rows and cols for items not involved in `x_batch`
        for most operations this does not affect the solution, however
        in our last step, it does, making this method an approximation
    - optionally prune each update to conserve memory
    - optionally add extra regularization to stabilize training
    """
    n_users, n_items = x.shape
    P = eye(n_items).tocsr() / l2_reg
    for x_batch in gen_batches(x, batch_size=batch_size):
        print('  subsetting...')
        inds = np.unique(x_batch.tocoo().col)
        x_batch = x_batch.tocsc()[:, inds]
        SPS = P[inds][:, inds]
        print('  matmuls...')
        XSPS = (x_batch @ SPS).toarray()
        XSPSX = XSPS @ x_batch.T
        print('  inverse...')
        B = np.linalg.inv(np.eye(x_batch.shape[0]) * (1.0 + extra_reg) + XSPSX)
        print('  computing update...')
        dSPS = - XSPS.T @ B @ XSPS
        SPS_new = SPS.toarray() + dSPS
        print('  pruning...')
        thr = get_pruning_threshold(P.tocsr(), target_density=target_density)
        SPS_new[np.abs(SPS_new) < thr] = 0.0
        print('  constructing sparse update...')
        dSPS = (csr_matrix(SPS_new) - SPS).tocoo()
        dP = csr_matrix((dSPS.data, (inds[dSPS.row], inds[dSPS.col])), shape=P.shape)
        print('  updating...')
        P += dP
        print(f'  max(abs(P)) = {np.max(np.abs(P))}')

    diag_indices = np.diag_indices(n_items)
    inv_diag = 1./P.diagonal()
    inv_diag[P.diagonal() == 0] = 1.0
    weights = - P.multiply(inv_diag).tocsr()
    weights[diag_indices] = 0.0
    if target_density < 1.0 and strict_sparsity:
        weights = prune_global(weights, target_density=target_density)

    return weights


def incremental_slim(x, batch_reg=5, batch_size=100, target_density=1.0):
    """Closed-form SLIM with incremental computation of the inverse gramm matrix,
    and a useful approximation.

    Compute a SLIM matrix for each batch (over 'active items' only)
    and average after pruning
    """
    n_users, n_items = x.shape
    n_batches = int(np.ceil(n_users / batch_size))
    P = eye(n_items).tocsr() / batch_reg * n_batches
    for x_batch in gen_batches(x, batch_size=batch_size):
        print('  submatrix...')
        X, cols = drop_empty_cols(x_batch)
        print('  computing update...')
        B = np.linalg.inv(np.eye(x_batch.shape[0]) * batch_reg + X @ X.T)
        XBX = X.T @ B @ X
        P_sub = - XBX / batch_reg
        print('  updating...')
        P = add_submatrix(P, P_sub, where=(cols, cols), target_density=target_density)

    diag_indices = np.diag_indices(n_items)
    inv_diag = 1./P.diagonal()
    inv_diag[P.diagonal() == 0] = 1.0
    weights = - P.multiply(inv_diag).tocsr()
    weights[diag_indices] = 0.0

    return weights


def naive_incremental_slim(x, batch_reg=5, batch_size=100, target_density=1.0):
    """Closed-form SLIM with incremental computation of the inverse gramm matrix,
    and a useful approximation.

    Compute a SLIM matrix for each batch (over 'active items' only)
    and average after pruning
    """
    n_users, n_items = x.shape
    B = csr_matrix((n_items, n_items))
    for x_batch in gen_batches(x, batch_size=batch_size):
        print('  submatrix...')
        XS, cols = drop_empty_cols(x_batch)
        print('  computing slim B for batch...')
        B_sub = closed_form_slim(XS.T @ XS, l2_reg=batch_reg)
        print('  updating...')
        B = add_submatrix(B, B_sub, where=(cols, cols), target_density=target_density)

    return B


def woodbury_ensemble(x, n_models=30, l2_reg=500, batch_size=100, target_density=1.0, strict_sparsity=True):

    weights = 0
    samples_per_model = int(np.ceil(x.shape[0] / n_models))
    for x_split in gen_batches(x, batch_size=samples_per_model):
        print(f'{x_split.shape[0]} users in split')
        w = woodbury_slim_approx(x_split, l2_reg=l2_reg, batch_size=batch_size, target_density=target_density)
        weights = weights + w

    weights = weights / n_models
    if target_density < 1.0 and strict_sparsity:
        weights = prune_global(weights, target_density=target_density)

    return weights


def drop_empty_cols(X):

    active_cols = np.unique(X.tocoo().col)
    X_sub = X.tocsc()[:, active_cols]

    return X_sub, active_cols


def add_submatrix(A, sub, where, target_density=1.0, verbose=False):
    """Add submatrix `sub` to `A` at indices `where = (rows, cols)`.
    Optionally ensure the result is sparse with density `target_density`
    """
    rows, cols = where
    if target_density < 1.0:
        thr = get_pruning_threshold(A.tocsr(), target_density=target_density)
        sub_before = A[rows][:, cols]
        sub_after = sub_before.toarray() + sub
        sub_after[np.abs(sub_after) < thr] = 0.0
        sub = (csr_matrix(sub_after) - sub_before).tocoo()
    dA = csr_matrix((sub.data, (rows[sub.row], cols[sub.col])), shape=A.shape)
    A += dA
    if target_density < 1.0:
        A.data[np.abs(A.data) < thr] = 0.0
        A.eliminate_zeros()
    if verbose:
        print(f'  update size = ({len(rows)}, {len(cols)})')
        print(f'  max(abs(A)) = {np.max(np.abs(A))}')

    return A


def gen_batches(x, batch_size=100):

    n_examples = x.shape[0]
    n_batches = int(np.ceil(n_examples / batch_size))
    for i_batch, start in enumerate(range(0, n_examples, batch_size)):
        print('batch {}/{}...'.format(i_batch + 1, n_batches))
        end = min(start + batch_size, n_examples)
        yield x[start:end]

