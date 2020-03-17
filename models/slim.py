
import time
from warnings import warn

import gin
import numpy as np
from scipy.sparse import issparse, csr_matrix, lil_matrix, coo_matrix, eye

from models.base import BaseRecommender
from util import prune_global, load_weights, get_pruning_threshold


@gin.configurable
class LinearRecommender(BaseRecommender):

    def __init__(self, log_dir, reg=500, density=1.0, batch_size=100, save_weights=True):
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
        self.weights = None
        super().__init__(log_dir, batch_size=batch_size)

    def train(self, x_train, y_train, x_val, y_val):
        """Train and evaluate"""
        if y_train is not None:
            warn("SLIM is unsupervised, y_train will be ignored")

        t1 = time.perf_counter()
        gramm = gramm_matrix(x_train).toarray()
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

    def __init__(self, log_dir, reg=500, density=1.0, batch_size=100, approx=False, save_weights=True):

        self.approx = approx
        super().__init__(log_dir, reg=reg, density=density, batch_size=batch_size, save_weights=save_weights)

    def train(self, x_train, y_train, x_val, y_val):
        """Train and evaluate"""
        if y_train is not None:
            warn("SLIM is unsupervised, y_train will be ignored")

        t0 = time.perf_counter()
        if self.approx:
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

    if issparse(gramm):
        gramm = gramm.toarray()
    print(f'  computing slim weights of shape {gramm.shape}')
    diag_indices = np.diag_indices(gramm.shape[0])
    gramm[diag_indices] += l2_reg
    inv_gramm = np.linalg.inv(gramm)
    weights = inv_gramm / (-np.diag(inv_gramm))
    weights[diag_indices] = 0.0

    return weights


def block_closed_form_slim(x, l2_reg=500, batch_size=100, density=1.0):

    n_users, n_items = x.shape
    weights = coo_matrix((n_items, n_items))
    for x_batch in gen_batches(x, batch_size=batch_size):
        gramm = gramm_matrix(x)
        gramm.eliminate_zeros()
        nonzero_rows = np.unique(gramm.tocoo().row)

        sub_gramm = gramm[nonzero_rows][:, nonzero_rows]
        sub_weights = csr_matrix(closed_form_slim(sub_gramm, l2_reg=l2_reg))

        rows = lil_matrix((sub_weights.shape[0], gramm.shape[1]))
        rows[:, nonzero_rows] = lil_matrix(sub_weights)
        batch_weights = lil_matrix(gramm.shape)
        batch_weights[nonzero_rows] = rows
        if density < 1.0:
            batch_weights = prune_global(batch_weights, density)
        weights += batch_weights.tocoo()
    n_batches = np.ceil(n_users / batch_size)
    weights = weights.tocsr() / n_batches

    return weights


def woodblocks(x, l2_reg=500, batch_size=100, density=1.0, eps=0.001, sparse_gramm=False):
    """Batched computation of the inverse gramm matrix

    steps:
        G is the gramm matrix of the updates
        U, V index the submatrix C of G such that C = V @ G @ U

        C = V @ G @ U
        B = inv(C) + V @ W @ U
        W = W - W @ U @ inv(B) @ V @ W
    """
    n_users, n_items = x.shape
    n_batches = np.ceil(n_users / batch_size)
    inv_gramm = eye(n_items).tocsr() if sparse_gramm else np.eye(n_items)
    inv_gramm /= l2_reg
    for x_batch in gen_batches(x, batch_size=batch_size):
        
        inds = np.unique(x_batch.tocoo().col)
        rank = len(inds)

        gramm_batch_sub = gramm_matrix(x_batch[:, inds])
        gramm_batch_sub.eliminate_zeros()
        if eps > 0.0:
            diag_indices = np.diag_indices(rank)
            gramm_batch_sub[diag_indices] += eps * l2_reg / n_batches

        inv_gramm_sub = inv_gramm[inds][:, inds]
        print(f'  inverting dense array of rank {rank}...')
        if sparse_gramm:
            inv_gramm_sub = inv_gramm_sub.toarray()
        inv_rel_update_sub = np.linalg.inv(gramm_batch_sub.toarray()) + inv_gramm_sub
        print(f'  inverting another dense array of rank {rank}...')
        rel_update_sub = np.linalg.inv(inv_rel_update_sub)
        print(f'  multiplying inverse gramm with relative update...')  # expensive
        if sparse_gramm:
            rel_update_sub = csr_matrix(rel_update_sub)
        update = inv_gramm[:, inds] @ rel_update_sub @ inv_gramm[inds]
        print(f'  updating the inverse gramm...')
        inv_gramm -= update
            
    diag_indices = np.diag_indices(n_items)
    weights = - inv_gramm / inv_gramm.diagonal()
    if sparse_gramm:
        weights = csr_matrix(weights)
    weights[diag_indices] = 0.0

    return weights


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


def woodbury_slim_approx(x, l2_reg=500, batch_size=100, target_density=1.0):
    """Closed-form SLIM with incremental computation of the inverse gramm matrix,
    and a useful approximation.

    TODO stabilize this computation: diverges for some batch sizes / seeds
    - additional regularization (diag += reg) to bound the lowest eigenvalue away from zero?
    - found that leaving the diagonal untouched in the pruning step is not enough
    - may be inevitable after 100+ matrix inversions

    we use the Woodbury matrix identity:
        inv(A + U C V) = inv(A) - inv(A) U inv(inv(c) + V inv(A) U) V inv(A)
    or if P is the inverse of gramm matrix G and if X is a batch of NEW users:
        inv(G + X.T X) = P X.T inv(I + X P X.T) X P
    and avoid some multiplication overhead by
    - precomputing X P and P X.T = (X P).T
    - drop rows and cols for items not involved in `x_batch`...
        - in various places where this does not affect the solution
        - in the final update, too, making this method an approximation
    """
    n_users, n_items = x.shape
    if target_density < 1.0:
        P = eye(n_items).tocsr() / l2_reg
    else:
        P = np.eye(n_items) / l2_reg
    for x_batch in gen_batches(x, batch_size=batch_size):
        print('  subsetting...')
        inds = np.unique(x_batch.tocoo().col)
        x_batch = x_batch.tocsc()[:, inds]
        SPS = P[inds][:, inds]
        print('  matmuls...')
        XSPS = (x_batch @ SPS).toarray()
        XSPSX = XSPS @ x_batch.T
        print('  inverse...')
        B = np.linalg.inv(np.eye(x_batch.shape[0]) + XSPSX)
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
    weights = prune_global(weights, target_density=target_density)

    return weights


def gen_batches(x, batch_size=100):

    n_examples = x.shape[0]
    n_batches = int(np.ceil(n_examples / batch_size))
    for i_batch, start in enumerate(range(0, n_examples, batch_size)):
        print('batch {}/{}...'.format(i_batch + 1, n_batches))
        end = min(start + batch_size, n_examples)
        yield x[start:end]


def gramm_matrix(x):

    return x.T.dot(x)
