
import time
from warnings import warn

import gin
import numpy as np
from scipy.special import expit
from scipy.sparse import csr_matrix, vstack, issparse, eye, find

from models.base import BaseRecommender
from models.slim import LinearRecommender, batched_gramm, closed_form_slim, add_submatrix
from util import Clock, gen_batches, gen_batch_inds, prune_global, prune_rows
from metric import binary_crossentropy_from_logits


@gin.configurable
class UserFactorRecommender(LinearRecommender):
    """High-dimensional matrix factorization recommender
    
    The model learns user factors U and item factors B
    such that U @ B ~ X
    where:
    U has the shape of a user-item matrix and is regularized
    - by constraining its sparsity pattern to that of X
    - with a l2 penalty ~ self.l2_reg
    B is an item-item weights matrix, regularized
    - by constraining its diagonal to zero
    - by constraining its sparsity pattern to that of X.T @ X
    - with a l2 penalty ~ weights_fn.l2_reg
    
    This model only does one iteration of computing the user and item factors,
    concretely, it uses the weights of a trained EASE^R model for B,
    and only then compute the user factors.
    For a model that trains the user factor and item factors jointly,
    see LogisticMFRecommender.
    """
    def __init__(self, log_dir, weights_fn=None, target_density=1.0, batch_size=100, save_weights=True,
                 als_iterations=0, l2_reg=1.0):

        self.l2_reg = l2_reg
        self.als_iterations = als_iterations
        self.weights_fn = closed_form_slim if weights_fn is None else weights_fn
        super().__init__(log_dir=log_dir, weights_fn=weights_fn, target_density=target_density,
                         batch_size=batch_size, save_weights=save_weights)

    def train(self, x_train, y_train, x_val, y_val):

        for i in range(self.als_iterations):
            self.weights = self.weights_fn(x_train)
            x_train = vstack(self.user_vector(x) for x, _ in
                             gen_batches(x_train, batch_size=1, print_interval=100))

        return super().train(x_train, y_train, x_val, y_val)

    def user_vector(self, x):
        """Predict a user vector from a row x of X

        Procedure
        - B is subset to retain only rows for which x is nonzero:
            B' = B[x > 0]
        - we then use B' to solve a variant of the lasso:
            argmin_u |u' @ B' - x| + l2_reg |u'|
          where u' is similarly a subset (of u, the desired user vector)
            u' = u[:, x > 0]
          it has closed-form solution:
            u' = x @ B't @ (B' @ B't + reg @ Ihh)^-1
          with
            Ihh = I[x > 0, x > 0]
        - the regularization term however is tweaked to keep u close to x rather than 0
            argmin_u |u' @ B' - x| + l2_reg |u - x'|
          with solution
            x @ (B'.T + reg @ Ih) @ (B' @ B't + reg @ Ihh)^-1
          where
            Ih = I[x > 0]
        """
        n_items = x.shape[1]
        ones, hist, _ = find(x)
        b = self.weights[hist]
        Ihh = eye(len(hist))
        Ihn = eye(n_items).tocsr()[hist]
        bbt = b @ b.T + Ihh * self.l2_reg
        xbt = x @ (b + Ihn * self.l2_reg).T
        if issparse(xbt):
            bbt, xbt = bbt.toarray(), xbt.toarray()
        else:
            bbt, xbt = np.asarray(bbt), np.asarray(xbt)
        u = xbt @ np.linalg.inv(bbt)

        return csr_matrix((u.flatten(), (ones, hist)), shape=x.shape)

    def predict(self, X, y=None):
        """Predict scores"""
        user_vectors = vstack(self.user_vector(x) for x, _ in
                              gen_batches(X, batch_size=1, print_interval=None))
        y_pred = user_vectors @ self.weights

        return y_pred, np.nan

 
@gin.configurable
class LogisticMFRecommender(BaseRecommender):

    def __init__(self, log_dir, latent_dim=100, batch_size=100, n_epochs=10, epoch_n_val=0,
                 save_weights=True):
        """Logistic weighted matrix factorization.
        See `sparse_logistic_mf`.
        """
        self.latent_dim = latent_dim
        self.n_epochs = n_epochs
        self.epoch_n_val = epoch_n_val
        self.save_weights = save_weights
        self.item_embeddings = None
        super().__init__(log_dir, batch_size=batch_size)

    def train(self, x_train, y_train, x_val, y_val):
        """Train and evaluate"""
        if y_train is not None:
            warn("Logistic MF is unsupervised, y_train will be ignored")

        print('Training...')
        logger = self.logger
        self.logger = None  # disable logging (give it back later)
        t1 = time.perf_counter()
        if self.latent_dim is None:
            user_embeddings, self.item_embeddings = sparse_inits(x_train, mode="cosine")
        else:
            user_embeddings, self.item_embeddings = dense_inits(x_train)
        for epoch in range(self.n_epochs):
            print(f'* epoch {epoch + 1}/{self.n_epochs}')
            if self.epoch_n_val:
                print(f'  computing validation metrics...')
                metrics = self.evaluate(x_val[:self.epoch_n_val], y_val[:self.epoch_n_val])
                print(f'  metrics at start of epoch {epoch + 1}:\n  {metrics}')
            if self.latent_dim is None:
                user_embeddings, self.item_embeddings = sparse_logistic_mf(x_train, user_embeddings,
                                                                           self.item_embeddings)
            else:
                user_embeddings, self.item_embeddings = logistic_mf(x_train, user_embeddings, self.item_embeddings)
        train_time = time.perf_counter() - t1
        self.logger = logger

        if self.logger is not None:
            self.logger.log_config(gin.operative_config_str())
            if self.save_weights:
                self.logger.save_weights(self.item_embeddings)

        print('Evaluating...')
        density = self.item_embeddings.size / np.prod(self.item_embeddings.shape)
        metrics = self.evaluate(x_val, y_val, other_metrics={'train_time': train_time, 'weight_density': density})

        return metrics

    def predict(self, x, y=None):
        """Predict scores"""
        for epochs in range(self.n_epochs):
            if self.latent_dim is None:
                user_embeddings, _ = sparse_logistic_mf(x, V=self.item_embeddings, lr_v=0.0,
                                                        batch_size=None, print_loss=False)
            else:
                user_embeddings, _ = logistic_mf(x, k=self.latent_dim, V=self.item_embeddings,
                                                 lr_v=0.0, batch_size=None, print_loss=False)
        y_pred = user_embeddings @ self.item_embeddings.T  # no sigmoid needed

        return y_pred, np.nan


def sparse_inits(X, col_nnz=200, target_density=0.01, init_scale=1.0, mode="cosine"):
    if mode == "slim":
        G = closed_form_slim(X)
        if target_density < 1.0:
            G = prune_global(G, target_density=target_density)
        if col_nnz is not None:
            G = prune_rows(G.T, target_nnz=col_nnz).T
    elif mode == "cosine":
        print('  computing Gramm matrix...')
        G = batched_gramm(X, cosine=True, row_nnz=col_nnz, target_density=target_density, batch_size=2020).T
        G[np.diag_indices_from(G)] = 0.0
    elif mode == "gramm":
        print('  computing Gramm matrix...')
        G = batched_gramm(X, cosine=False, row_nnz=col_nnz, target_density=target_density, batch_size=2020).T
        G[np.diag_indices_from(G)] = 0.0
    else:
        raise ValueError("init options are: slim, cosine, gramm")
    U = init_scale * X.copy().tocsr()
    V = init_scale * G.tocsr()

    return U, V


def dense_inits(X, k=100, init_scale=1.0):

    n_users, n_items = X.shape
    U = init_scale * np.random.randn(n_users, k)
    V = init_scale * np.random.randn(n_items, k)

    return U, V


@gin.configurable
def sparse_logistic_mf(X, U=None, V=None, l2_reg=1.0, alpha_w=0.0, alpha_s=0.5,
                       lr_u=0.001, lr_v=0.001, batch_size=100,
                       print_loss=True, verbose=True):
    """Fit sparse user and item factors to a preference matrix for one epoch.
    See `logistic_mf`.

    In this sparse version, we constrain U to have the same sparsity structure as X
    and V to have the sparsity structure of the co-occurence matrix X.T @ X
    - optionally weighted and pruned (as in standard item-knn recommenders)
    - with an additional zero constraint on the diagonal (as in EASE^R)
    The nonzero entries of U and V are learned with minibatch SGD.

    TODO: should we print loss at all?
    NOTE: technically, loss across all of y_hat is inf due to the -log(y_hat) term
    """
    n_users, n_items = X.shape
    if U is None:
        U = X.copy().tocsr()
    if lr_u == 0 and lr_v == 0:
        return U, V
    if lr_v > 0:
        V_bin = sparse_binarize(V.copy())

    s = 2 * np.log(1 / alpha_s - 1)
    clock = Clock(verbose=verbose)
    for batch in gen_batch_inds(n_users, batch_size=batch_size, shuffle=True):
        clock.interval('computing y_pred and dL/dz')
        print(f'    mean, std U.data = {np.mean(U.data):.3f}, {np.std(U.data):.3f}')
        print(f'    mean, std V.data = {np.mean(V.data):.3f}, {np.std(V.data):.3f}')
        x, u = X[batch], U[batch]
        y = sparse_binarize(x)
        z = u @ V.T
        y_pred = sparse_sigmoid(z, a=s, b=-s/2)
        dldz = y_pred + alpha_w * y_pred.multiply(x) - sparse_binarize(x) - alpha_w * x
        dldz = s * dldz  # account for scaling in sparse_sigmoid
        if print_loss:
            clock.interval('computing loss')
            bce = sparse_bce_from_logits(z, y, a=s, b=s/2)
            print(f'    BCE (with scale {s:.3f}) = {np.mean(bce):.3f}')

        print(f'    mean y = {np.sum(y.data) / np.prod(y.shape):.3f}')
        print(f'    mean y_pred = {np.sum(y_pred.data) / np.prod(y_pred.shape):.3f}')
        print(f'    mean dldz = {np.sum(dldz.data) / np.prod(dldz.shape):.3f}')
        if lr_u > 0:
            clock.interval('updating U')
            dldu = dldz @ V + 2 * l2_reg * u
            dldu = dldu.multiply(y)  # constrain to x.nonzero()
            U = add_submatrix(U, - lr_u * dldu, where=(batch, None))  # gradient step
            print(f'    mean dldu = {np.sum(dldu.data) / np.prod(dldu.shape):.3f}')
            print(f'    mean, std dldu.data = {np.mean(dldu.data):.3f}, {np.std(dldu.data):.3f}')
            print(f'    mean, std U.data = {np.mean(U.data):.3f}, {np.std(U.data):.3f}')
        if lr_v > 0:
            clock.interval('updating V')
            dldV = dldz.T @ u + 2 * l2_reg * V
            dldV = dldV.multiply(V_bin)  # constrain to V.nonzero()
            V = add_submatrix(V, - lr_v * dldV)  # gradient step
            print(f'    mean dldV = {np.sum(dldV.data) / np.prod(dldV.shape):.3f}')
            print(f'    mean, std dldV.data = {np.mean(dldV.data):.3f}, {np.std(dldV.data):.3f}')
            print(f'    mean, std V.data = {np.mean(V.data):.3f}, {np.std(V.data):.3f}')

    return U, V


@gin.configurable
def logistic_mf(X, U=None, V=None, l2_reg=1.0, alpha=10.0, lr_u=0.001, lr_v=0.001, batch_size=100,
                k=100, init_scale=1.0, print_loss=True, verbose=True):
    """Fit user and item factors to a preference matrix for one epoch,
    using a weighted logistic matrix factorization objective:
        loss = WBCE(W, Y, Y_hat) + l2_reg * (|U|^2 + |V|^2)
    where
        Y = X > 0
        Y_hat = sigm(U @ V.T)
        WBCE is "weighted binary cross-entropy"
            - sum( W * Y * log(Y_hat) + W * (1-Y) * log(1-Y_hat)) )
        W = 1 + alpha * X
        U, V the user and item vectors
    """
    n_users, n_items = X.shape
    if U is None:
        U = init_scale * np.random.randn(n_users, k)
    if lr_u == 0 and lr_v == 0:
        return U, V

    clock = Clock(verbose=verbose)
    for batch in gen_batch_inds(n_users, batch_size=batch_size, shuffle=True):
        clock.interval('computing y_pred and dL/dz')
        x, u = X[batch].toarray(), U[batch]
        y = (x > 0).astype(float)
        y_pred = expit(u @ V.T)
        if print_loss:
            clock.interval('computing loss')
            bce = binary_crossentropy_from_logits(u @ V.T, y)
            print(f'    batch BCE = {np.mean(bce):.3f}')
        
        dldz = y_pred + alpha * y_pred * x - y - alpha * x
        if lr_u > 0:
            clock.interval('updating U')
            dldu = dldz @ V + 2 * l2_reg * u
            U[batch] = U[batch] - lr_u * dldu  # gradient step
        if lr_v > 0:
            clock.interval('updating V')
            dldV = dldz.T @ u + 2 * l2_reg * V
            V = V - lr_v * dldV  # gradient step

    return U, V


def sparse_sigmoid(z, a=1.0, b=0.0):
    """Apply sigmoid to the nonzero entries of sparse matrix Z.

    NOTE: sigmoid(0.0) = 0.5 whereas this function maps exact zeros to 0.0.
    Calculations using this function will only make sense if all offending
    entries are masked in later operations.
    """
    z.data = expit(a * z.data + b)
    return z


def sparse_binarize(x):

    x.data = np.ones_like(x.data)
    return x


def sparse_bce_from_logits(z, y, a=1.0, b=0.0):
    """Sparse binary crossentropy from logits

    log(y_hat) = log(exp(z) / (1 + exp(z)))
               = z - log(1 + exp(z))
    log(1 - y_hat) = log(1 - exp(z) / (1 + exp(z)))
                   = log(1 / (1 + exp(z)))
                   = - log(1 + exp(z))
    y log(y_hat) + (1 - y) log(1- y_hat) = yz - y log(1 + exp(z)) - log(1 + exp(z)) + y log(1 + exp(z))
                                         = yz - log(1 + exp(z))
    bce = - sum(yz - log(1 + exp(z)), axis=1) / n_items
    """
    yazb = y.multiply(a * z) + y * b  # y(az+b)
    ll_pos = yazb.sum(axis=1)
    ll_neg_zeros = -np.log(1 + np.exp(b)) * (z.shape[1] - z.getnnz(axis=1))  # - log(1 + exp(b)) for every zero in z
    ll_neg_nonzeros = a * z
    ll_neg_nonzeros.data = -np.log(1 + np.exp(ll_neg_nonzeros.data + b))  # - log(1 + exp(az + b)) for all nonzeros
    ll_neg_nonzeros = ll_neg_nonzeros.sum(axis=1)
    ll = ll_pos + ll_neg_zeros + ll_neg_nonzeros

    return -ll / y.shape[1]  # average nll over classes
