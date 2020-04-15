
import time
from copy import deepcopy
from warnings import warn

import gin
import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, issparse, eye
from sklearn.decomposition import TruncatedSVD

from models.base import BaseRecommender
from util import load_weights, prune_global, prune_rows, gen_batches


@gin.configurable
class LinearRecommender(BaseRecommender):

    def __init__(self, log_dir, weights_fn=None, target_density=1.0, batch_size=100, save_weights=True):
        """Linear recommender based on Harald Steck's closed form variant [1]
        of Sparse Linear Methods (SLIM) [2].

        [1] Harald Steck, Embarrassingly shallow auto-encoders. WWW 2019
        https://arxiv.org/pdf/1905.03375.pdf

        [3] Xia Ning and George Karypis, SLIM: Sparse Linear Methods for
        Top-N Recommender Systems. ICDM 2011
        http://glaros.dtc.umn.edu/gkhome/node/774
        """
        self.weights_fn = weights_fn
        self.target_density = target_density
        self.save_weights = save_weights

        self.weights = None
        if self.weights_fn is None:
            self.weights_fn = closed_form_slim
        super().__init__(log_dir, batch_size=batch_size)

    def train(self, x_train, y_train, x_val, y_val):
        """Train and evaluate"""
        if y_train is not None:
            warn("SLIM is unsupervised, y_train will be ignored")

        t1 = time.perf_counter()
        self.weights = self.weights_fn(x_train)
        if self.target_density < 1.0:
            self.weights = prune_global(self.weights, target_density=self.target_density)
        dt = time.perf_counter() - t1
        if self.logger is not None:
            self.logger.log_config(gin.operative_config_str())
            if self.save_weights:
                self.logger.save_weights(self.weights)

        print('Evaluating...')
        density = self.weights.size / np.prod(self.weights.shape)
        metrics = self.evaluate(x_val, y_val, other_metrics={'train_time': dt, 'weight_density': density})

        return metrics

    def predict(self, x, y=None):
        """Predict scores"""
        y_pred = x @ self.weights

        return y_pred, np.nan


@gin.configurable
class EmbeddingRecommender(BaseRecommender):

    def __init__(self, log_dir, embedding_fn=None, item_nnz=100, batch_size=100, save_embeddings=True):
        """Embedding recommender.
        """
        self.embedding_fn = embedding_fn
        self.item_nnz = item_nnz
        self.save_embeddings = save_embeddings

        self.embeddings = None
        self.priors = None
        if self.embedding_fn is None:
            self.embedding_fn = cholesky_embeddings
        super().__init__(log_dir, batch_size=batch_size)

    def train(self, x_train, y_train, x_val, y_val):
        """Train and evaluate"""
        if y_train is not None:
            warn("SLIM is unsupervised, y_train will be ignored")

        t1 = time.perf_counter()
        self.embeddings, self.priors = self.embedding_fn(x_train)
        if self.item_nnz is not None:
            self.embeddings = prune_rows(self.embeddings, target_nnz=self.item_nnz)
        dt = time.perf_counter() - t1
        if self.logger is not None:
            self.logger.log_config(gin.operative_config_str())
            if self.save_embeddings:
                self.logger.save_weights(self.embeddings)

        print('Evaluating...')
        density = self.embeddings.size / np.prod(self.embeddings.shape)
        if issparse(self.embeddings):
            item_nnz = np.mean(np.ravel(self.embeddings.getnnz(axis=1)))
        else:
            item_nnz = self.embeddings.shape[1]
        other_metrics = {'train_time': dt, 'embedding_density': density, 'item_nnz': item_nnz}
        metrics = self.evaluate(x_val, y_val, other_metrics=other_metrics)

        return metrics

    def predict(self, x, y=None):
        """Predict scores"""
        h = x @ self.embeddings
        y = h @ self.embeddings.T
        if self.priors is not None:
            priors = np.reshape(self.priors, (1, -1))
            y = y.multiply(priors).tocsr() if issparse(y) else y * priors

        return y, np.nan


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


class Clock(object):

    def __init__(self):
        self.t0 = time.perf_counter()

    def tic(self):
        self.t0 = time.perf_counter()

    def toc(self):
        elapsed = time.perf_counter() - self.t0
        print(f'    elapsed: {elapsed}')

    def print_interval(self):
        self.toc()
        self.tic()


def closed_form_slim_from_gramm(gramm, l2_reg=500):

    if issparse(gramm):
        gramm = gramm.toarray()
    diag_indices = np.diag_indices(gramm.shape[0])
    gramm[diag_indices] += l2_reg
    inv_gramm = np.linalg.inv(gramm)
    weights = inv_gramm / (-np.diag(inv_gramm))
    weights[diag_indices] = 0.0

    return weights


@gin.configurable
def closed_form_slim(x, l2_reg=500):

    return closed_form_slim_from_gramm(x.T @ x, l2_reg=l2_reg)


@gin.configurable
def cholesky_slim(x, l2_reg=500, beta=2.0, row_nnz=None, sort_by_nn=False, zero_diag=True):

    E, prior = cholesky_embeddings(x, l2_reg=l2_reg, beta=beta, row_nnz=row_nnz, sort_by_nn=sort_by_nn)
    B = E @ E.T.multiply(prior.reshape(1, -1))
    if zero_diag:
        B[np.diag_indices(E.shape[0])] = 0.0

    return B


@gin.configurable
def cholesky_embeddings(x, l2_reg=500, beta=2.0, row_nnz=None, sort_by_nn=False):
    """Cholesky factors of -P + beta * I

    If we're only interested in recommending new items, changes
    to the diagonal of B are not going to matter, so we might as well do the thing where we
    apply cholesky to x.T x, invert the resulting lower factor,
    and feed its transpose back into cholesky_update_AAt with beta = np.max(diag_P) * beta
    to get E (if that makes sense).
    """
    clock = Clock()
    assert beta > 1.0

    print('computing gramm matrix G')
    G = x.T @ x
    diag_indices = np.diag_indices(G.shape[0])
    G[diag_indices] += l2_reg
    clock.print_interval()
    if sort_by_nn:
        print('sorting items by number of number of neighbors')
        items_by_nn = np.argsort(np.ravel(G.getnnz(axis=0)))  # nn = non-zero co-counts
        G = G[items_by_nn][:, items_by_nn]
        clock.print_interval()
    print('computing inverse P')
    P = np.linalg.inv(G.toarray())
    diag_P = np.diag(P)
    clock.print_interval()

    print('factorizing -P + beta * diag_P')
    A = -P
    A[diag_indices] += beta * diag_P
    if G.nnz / np.prod(G.shape) > 0.15:
        E = sp.linalg.cholesky(A, lower=True)
    else:
        from sksparse.cholmod import cholesky
        A = csc_matrix((A[G.nonzero()], G.nonzero()), shape=G.shape)
        E = cholesky(A, ordering_method='natural').L()
    clock.print_interval()
    print('pruning factors')
    E = prune_rows(E, target_nnz=row_nnz)
    clock.print_interval()

    print('computing priors')
    prior = 1 / diag_P
    prior[diag_P == 0] = 0.0
    if sort_by_nn:
        print('undo sort items')
        original_order = np.argsort(items_by_nn)
        prior = prior[original_order]
        E = E[original_order]

    return E, prior


@gin.configurable
def svd_slim(x, k=100, l2_reg=500.0):

    svd = TruncatedSVD(n_components=k).fit(x)
    Sx, Vt = svd.singular_values_, svd.components_

    Sp = 1 / (Sx**2 + l2_reg)
    Dp = (Vt.T**2 * Sp).sum(axis=1)

    return - Vt.T * Sp @ Vt / Dp


@gin.configurable
def block_slim(x, l2_reg=1.0, row_nnz=1000, target_density=0.01, r_blanket=0.5, max_iter=None):
    """Sparse but approximate 'block-wise' variant of the closed-form slim algorithm.
    Both algorithms due to Steck.

    POTENTIAL IMPROVEMENTS:
    - still takes a lot of iterations - currently ~10K blocks
        i.e. possibly only 4x faster than 40K sparse regressions?
        PLAN:
        - instead of sub_list, use a sparse matrix of n_blocks x n_items
          (with some abuse of explicit zeros)
        - we could then swap this out for a sparse matrix U that decomposes A ~ U.T @ U
    - still need to store a lot of values:
        paper implementation does not avoid the large memory footprint of the combined blocks
    - still need to compute the whole gramm, or worse, the dense correlation matrix
        should read the paper about how an offset mu doesn't affect some outcome
    - averaging is not particularly sound
        though admittedly it's close in performance to rr = 0.0 which avoids averaging
    """
    print('  computing gramm matrix G')
    G = (x.T @ x).tocsr().astype(np.float32)
    print('  computing sparsity pattern A')
    A = prune_global(G, target_density=target_density)
    A = prune_rows(A, target_nnz=row_nnz).tocsr()

    B = csr_matrix(G.shape)
    blocks = csr_matrix(G.shape)
    for sub, weights in gen_submatrices(A, r_blanket=r_blanket, max_iter=max_iter):
        print(f'  computing weights for block of size {len(sub)}...')
        block = A[sub][:, sub].tocoo().T  # T: limit col nnz rather than row
        block.data = weights[block.col]
        B_sub = closed_form_slim_from_gramm(G[sub][:, sub], l2_reg=l2_reg)
        B_sub = coo_matrix((B_sub[block.nonzero()], block.nonzero()))
        B_sub.data = B_sub.data * weights[B_sub.col]
        B = add_submatrix(B, B_sub, where=(sub, sub))
        blocks = add_submatrix(blocks, block, where=(sub, sub))

    print(f'  scaling B by number of blocks summed...')
    B[B.nonzero()] = B[B.nonzero()] / blocks[B.nonzero()]

    return B


@gin.configurable
def block_slim_steck(train_data, alpha=0.75, threshold=50, rr=0.5, maxInColumn=1000, L2reg=1.0,
                     max_iter=None, sparse_gramm=True):
    """Sparse but approximate 'block-wise' variant of the closed-form slim algorithm.
    Both algorithms due to Steck.
    This implementation is a close adaptation of Steck's code released with [1].
    It "implements section 3.2 in the paper".

    [1] Steck, Harald. "Markov Random Fields for Collaborative Filtering."
    Advances in Neural Information Processing Systems. 2019.
    """
    print("computing gramm matrix XtX and sparsity pattern AA")
    myClock = Clock()
    if sparse_gramm:
        # gramm matrix XtX as a sp.sparse matrix to preserve memory and
        # pruned gramm matrix  AA based on XtX rather than the correlation matrix [JVB]
        XtX = train_data.T.dot(train_data).tocsr()
        ii_diag = np.diag_indices(XtX.shape[0])
        XtXdiag = np.asarray(XtX[ii_diag]).flatten()
        AA = XtX.tocsc().astype(np.float32)
        AA.data[np.abs(AA.data) <= threshold] = 0.0
        AA.eliminate_zeros()
    else:
        userCount = train_data.shape[0]
        XtX = np.asarray(train_data.T.dot(train_data).todense(), dtype=np.float32)
        del train_data  # only the item-item data-matrix XtX is needed in the following

        mu = np.diag(XtX) / userCount   # the mean of the columns in train_data (for binary train_data)
        # variances of columns in train_data (scaled by userCount)
        variance_times_userCount = np.diag(XtX) - mu * mu * userCount

        # standardizing the data-matrix XtX (if alpha=1, then XtX becomes the correlation matrix)
        XtX -= mu[:, None] * (mu * userCount)
        rescaling = np.power(variance_times_userCount, alpha / 2.0)
        scaling = 1.0 / rescaling
        XtX = scaling[:, None] * XtX * scaling

        XtXdiag = deepcopy(np.diag(XtX))
        ii_diag = np.diag_indices(XtX.shape[0])
        print("    number of items: {}".format(len(mu)))
        print("    number of users: {}".format(userCount))

        print("    sparsifying the data-matrix (section 3.1 in the paper) ...")
        myClock.tic()
        # apply threshold
        ix = np.where(np.abs(XtX) > threshold)
        AA = csc_matrix((XtX[ix], ix), shape=XtX.shape, dtype=np.float32)

    # enforce maxInColumn, see section 3.1 in paper
    countInColumns = AA.getnnz(axis=0)
    iiList = np.where(countInColumns > maxInColumn)[0]
    print("    number of items with more than {} entries in column: {}".format(maxInColumn, len(iiList)))
    for ii in iiList:
        jj = AA[:, ii].nonzero()[0]
        kk = np.argpartition(-np.abs(np.asarray(AA[jj, ii].todense()).flatten()), maxInColumn)[maxInColumn:]
        AA[jj[kk], ii] = 0.0
    AA.eliminate_zeros()
    print("    resulting sparsity of AA: {}".format(AA.nnz*1.0 / AA.shape[0] / AA.shape[0]))
    myClock.toc()

    XtX[ii_diag] = XtXdiag + L2reg

    # list L in the paper, sorted by item-counts per column
    # ties broken by item-popularities as reflected by np.diag(XtX)
    AAcountInColumns = AA.getnnz(axis=0)  # [re-use countInColumns? more accurate?; JVB]
    sortedList = np.argsort(AAcountInColumns + XtXdiag / 2.0 / np.max(XtXdiag))[::-1]  # [re-use XtXdiag; JVB]

    print("iterating through steps 1,2, and 4 in section 3.2 of the paper ...")
    myClock.tic()
    todoIndicators = np.ones(AAcountInColumns.shape[0])
    blockList = []  # list of blocks. Each block is a list of item-indices, to be processed in step 3 of the paper
    for ii in sortedList:
        if todoIndicators[ii] == 1:
            nn, _, vals = sp.sparse.find(AA[:, ii])  # step 1 in paper: set nn contains item ii and its neighbors N
            if len(nn) < 2:
                continue

            kk = np.argsort(np.abs(vals))[::-1]
            nn = nn[kk]
            blockList.append(nn)  # list of items in the block, to be processed in step 3 below
            # remove possibly several items from list L, as determined by parameter rr (r in the paper)
            dd_count = max(1, int(np.ceil(len(nn)*rr)))
            dd = nn[:dd_count]  # set D, see step 2 in the paper
            todoIndicators[dd] = 0  # step 4 in the paper
    myClock.toc()

    print("now step 3 in section 3.2 of the paper: iterating ...")
    # now the (possibly heavy) computations of step 3:
    # given that steps 1,2,4 are already done, the following for-loop could be implemented in parallel.
    myClock.tic()
    BBlist_ix1, BBlist_ix2, BBlist_val = [], [], []
    for i, nn in enumerate(blockList[:max_iter]):    # [max_iter; JVB]
        print(f'  iter {i+1}/{len(blockList)}: {len(nn)}x{len(nn)}')  # [JVB]
        # calculate dense solution for the items in set nn
        BBblock = np.linalg.inv(XtX[nn][:, nn].toarray())  # [adapt to sparse: more efficient slicing, toarray; JVB]
        BBblock /= -np.diag(BBblock)
        # determine set D based on parameter rr (r in the paper)
        dd_count = max(1, int(np.ceil(len(nn)*rr)))
        dd = nn[:dd_count]  # set D in paper
        # store the solution regarding the items in D
        blockix = np.meshgrid(dd, nn)
        BBlist_ix1.extend(blockix[1].flatten().tolist())
        BBlist_ix2.extend(blockix[0].flatten().tolist())
        BBlist_val.extend(BBblock[:, :dd_count].flatten().tolist())
    myClock.toc()

    print("final step: obtaining the sparse matrix BB by averaging the solutions regarding the various sets D ...")
    myClock.tic()
    BBsum = csc_matrix((BBlist_val, (BBlist_ix1, BBlist_ix2)), shape=XtX.shape, dtype=np.float32)
    BBcnt = csc_matrix((np.ones(len(BBlist_ix1), dtype=np.float32), (BBlist_ix1, BBlist_ix2)),
                       shape=XtX.shape, dtype=np.float32)
    b_div = sp.sparse.find(BBcnt)[2]
    b_3 = sp.sparse.find(BBsum)
    BBavg = csc_matrix((b_3[2] / b_div, (b_3[0], b_3[1])), shape=XtX.shape, dtype=np.float32)
    BBavg[ii_diag] = 0.0
    myClock.toc()

    print("forcing the sparsity pattern of AA onto BB ...")
    myClock.tic()
    BBavg = csr_matrix((np.asarray(BBavg[AA.nonzero()]).flatten(), AA.nonzero()),
                       shape=BBavg.shape, dtype=np.float32)
    print("    resulting sparsity of learned BB: {}".format(BBavg.nnz * 1.0 / AA.shape[0] / AA.shape[0]))
    myClock.toc()

    return BBavg


@gin.configurable
def woodbury_slim(x, l2_reg=500, batch_size=100, target_density=1.0, extra_reg=0.0):
    """Closed-form SLIM with incremental computation of the inverse gramm matrix,
    using the Woodbury matrix identity.

    TODO stabilize this computation: diverges for some batch sizes / seeds
    - additional regularization (diag += reg) to bound the lowest eigenvalue away from zero?
    - found that leaving the diagonal untouched in the pruning step is not enough
    - may be inevitable after 100+ matrix inversions

    We use the Woodbury matrix identity:
        inv(A + U C V) = inv(A) - inv(A) U inv(inv(c) + V inv(A) U) V inv(A)
    or if P is the inverse of gramm matrix G and if X is a batch of NEW users:
        inv(G + X.T X) = P - P X.T inv(I + X P X.T) X P
    We also
    - precompute X P which also gets us P X.T = (X P).T
    - avoid some multiplication overhead by dropping rows and cols for items not involved in `x_batch`
        for most operations this does not affect the solution, however
        in our last step, it does, making this method an approximation
    - optionally prune after updating, to conserve memory
    - optionally add extra regularization to stabilize training
    """
    n_users, n_items = x.shape
    P = eye(n_items).tocsr() / l2_reg
    for x_batch, _ in gen_batches(x, batch_size=batch_size):
        print('  subsetting...')
        inds = np.unique(x_batch.tocoo().col)
        x_batch = x_batch.tocsc()[:, inds]
        SPS = P[inds][:, inds]
        print('  matmuls...')
        XSPS = (x_batch @ SPS).toarray()
        XSPSX = XSPS @ x_batch.T
        print('  inverse...')
        B = np.linalg.inv(np.eye(x_batch.shape[0]) * (1.0 + extra_reg) + XSPSX)
        print('  updating...')
        dSPS = - XSPS.T @ B @ XSPS
        P = add_submatrix(P, dSPS, where=(inds, inds), target_density=target_density)
        print(f'  max(abs(P)) = {np.max(np.abs(P))}')

    diag_indices = np.diag_indices(n_items)
    inv_diag = 1./P.diagonal()
    inv_diag[P.diagonal() == 0] = 1.0
    weights = - P.multiply(inv_diag).tocsr()
    weights[diag_indices] = 0.0

    return weights


@gin.configurable
def naive_incremental_slim(x, batch_size=50, l2_reg=10, target_density=1.0):
    """Closed-form SLIM with incremental computation of the inverse gramm matrix,
    and a useful approximation.

    Compute a SLIM matrix for each batch (over 'active items' only)
    and average after pruning
    """
    n_users, n_items = x.shape
    B = csr_matrix((n_items, n_items))
    for x_batch, _ in gen_batches(x, batch_size=batch_size):
        print('  submatrix...')
        XS, cols = drop_empty_cols(x_batch)
        print('  computing slim B for batch...')
        G_upd = XS.T @ XS
        B_upd = closed_form_slim_from_gramm(G_upd, l2_reg=l2_reg)
        B_upd = coo_matrix((B_upd[G_upd.nonzero()], G_upd.nonzero()), shape=B.shape)
        print('  updating...')
        B = add_submatrix(B, B_upd, where=(cols, cols), target_density=target_density)

    return B


def gen_submatrices(A, r_blanket=0.5, max_iter=None):
    """Generates square submatrices, represented as sets of items,
    plus binary weights indicating which items are in the present
    submatrix' "markov blanket" as defined by Steck in [1]--
    this will be 1 item if r_blanket = 0, or all if r_blanket = 1

    Using a list comprehension rather than np.setdiff1d at the end
    since set diff doesn't preserve order.

    [1] Steck, Harald. "Markov Random Fields for Collaborative Filtering."
    Advances in Neural Information Processing Systems. 2019.
    """
    sort_scores = A.getnnz(axis=1) + 0.5 * A.diagonal() / A.diagonal().max()
    ind_list = np.argsort(-sort_scores)

    i = 0
    while len(ind_list) and (max_iter is None or i < max_iter):
        _, sub, vals = sp.sparse.find(A[ind_list[0]])
        if len(sub) < 2:
            ind_list = ind_list[1:]
            continue
        thr = np.percentile(np.abs(vals), 100 * r_blanket)
        markov_blanket = vals >= thr
        yield sub, markov_blanket
        drop = set(sub[markov_blanket])
        ind_list = [i for i in ind_list if i not in drop]
        print(f'  {len(ind_list)} remaining...')
        i += 1


def add_submatrix(A, dA, where=None, target_density=1.0, max_density=None):
    """Add submatrix `sub` to `A` at indices `where = (rows, cols)`.
    Optionally prune the result down to density `target_density`

    NOTE: do not use with target_density < 1.0 when summing
    matrices before normalizing the sum by the number of
    matrices summed -- pruning won't be correct
    """
    if max_density is None:
        max_density = 3 * target_density
    if dA.size > max_density * np.prod(A.shape):
        thr = np.min(np.abs(A.data[np.abs(A.data) > 0]))
        dA[np.abs(dA) < thr] = 0.0
    dA = coo_matrix(dA)
    if where is not None:
        rows, cols = where
        dA = csr_matrix((dA.data, (rows[dA.row], cols[dA.col])), shape=A.shape)
    A += dA
    if A.nnz > max_density * np.prod(A.shape):
        print(f'  density > max_density: pruning result')
        A = prune_global(A, target_density=target_density, copy=False)

    return A


def drop_empty_cols(X):

    active_cols = np.unique(X.tocoo().col)
    X_sub = X.tocsc()[:, active_cols]

    return X_sub, active_cols
