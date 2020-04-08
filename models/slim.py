
import time
from warnings import warn

import gin
import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, issparse, eye
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import NMF

from models.base import BaseRecommender
from util import load_weights, prune_global, prune_rows, get_pruning_threshold


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


@gin.configurable
def distilled_slim(x, n_distillations=1, head_tail=False):

    gramm = x.T @ x
    if head_tail:
        weights = head_tail_slim_from_gramm(gramm)
    else:
        weights = closed_form_slim_from_gramm(gramm)
    if issparse(weights):
        weights = weights.toarray()
    for i in range(n_distillations):
        print(f'Distilling ({i+1}/{n_distillations})')

        print('recomputing gramm matrix...')
        t = time.perf_counter()
        gramm = weights @ gramm.toarray() @ weights
        print(f'  elapsed: {time.perf_counter() - t}')

        print('recomputing distilled weights...')
        if head_tail:
            w_distilled = head_tail_slim_from_gramm(gramm)
        else:
            w_distilled = closed_form_slim_from_gramm(gramm)

        print('multiplying weights...')
        t = time.perf_counter()
        if issparse(w_distilled):
            w_distilled = w_distilled.toarray()
        weights = weights @ w_distilled
        print(f'  elapsed: {time.perf_counter() - t}')

    return weights


@gin.configurable
def closed_form_slim(x, l2_reg=500):

    return closed_form_slim_from_gramm(x.T @ x, l2_reg=l2_reg)


@gin.configurable
def head_tail_slim(x, l2_reg=500, target_density=0.01, gramm_mass_thr=0.1):
    
    return head_tail_slim_from_gramm(x.T @ x, l2_reg=l2_reg, target_density=target_density, gramm_mass_thr=0.1)


def closed_form_slim_from_gramm(gramm, l2_reg=500):

    if issparse(gramm):
        gramm = gramm.toarray()
    diag_indices = np.diag_indices(gramm.shape[0])
    gramm[diag_indices] += l2_reg
    inv_gramm = np.linalg.inv(gramm)
    weights = inv_gramm / (-np.diag(inv_gramm))
    weights[diag_indices] = 0.0

    return weights


def head_tail_slim_from_gramm(gramm, l2_reg=500, target_density=0.01, gramm_mass_thr=0.1):

    B = csr_matrix(gramm.shape)
    t = time.perf_counter()

    print('- computing head and tail...')
    head, tail = get_head_tail(gramm, gramm_mass_thr)
    print(f'  elapsed: {time.perf_counter() - t}')
    t = time.perf_counter()

    print(f'- computing head weights - shape {(len(head), len(head))}...')
    G_head = gramm[head][:, head]
    if issparse(G_head):
        G_head = G_head.toarray()
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

    print(f'- computing head x tail weights - shape {(len(head), len(tail))}...')
    G_head_tail = gramm[head][:, tail]
    if issparse(G_head):
        G_head_tail = G_head_tail.toarray()
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


@gin.configurable
def block_slim_steck(x, l2_reg=1.0, row_nnz=1000, target_density=0.01, r_blanket=0.5, max_iter=None):
    """Sparse but approximate 'block-wise' variant of
    the closed-form slim algorithm. Both algorithms due to Steck.

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
        block = A[sub][:, sub].tocoo()
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
def online_slim(x, l2_reg=1.0, batch_size=1000, row_nnz=1000, target_density=0.01, r_visit=0.95, max_iter=1000):
    """
    Online structure learning for sparse linear recommenders

    goal:
    given a matrix x of interactions between M users and N items
    learn a sparse NxN matrix B such that X ~= X@B
    with a memory footprint O(K*N + k*M) with K < 1000
    i.e.:
    - dataset X should not have to fit in memory
    - and neither should the full, dense, B
    """
    x = x.tocsc()
    n_users, n_items = x.shape
    A = csr_matrix((n_items, n_items))
    B = csr_matrix((n_items, n_items))
    blocks = csr_matrix((n_items, n_items))
    for x_batch in gen_batches(x, batch_size):
        print('  updating A...')
        x_batch, cols = drop_empty_cols(x_batch)
        A = add_submatrix(A, x_batch.T @ x_batch, where=(cols, cols), target_density=target_density)
        print('  pruning the rows of A...')
        A[cols] = prune_rows(A[cols], target_nnz=row_nnz).tocsr()  # slow but important...

        print('  computing matrix block to update...')
        r_visited = (blocks[A.nonzero()] > 0).mean()
        print(f'  visited: {r_visited}...')
        min_visits = blocks[A.nonzero()].min()
        print(f'  min_visits = {min_visits}...')
        least_visited = np.asarray(blocks[A.nonzero()] == min_visits).flatten()
        least_visited = csr_matrix((least_visited, A.nonzero()), shape=A.shape)
        next_best_row = np.asarray(least_visited.sum(axis=1)).flatten()
        col = np.argmax(next_best_row)  # ...or this gets stuck
        print(f'  getting neighbors for item {col}...')
        _, sub, vals = sp.sparse.find(A[col])
        if len(sub) < 2:
            print(f'len(sub) = {len(sub)} < 2, skipping...')
            if len(sub):
                blocks[sub[0], sub[0]] += 1
            continue
        if len(sub) > row_nnz:
            sub = sub[np.argpartition(np.abs(vals), -row_nnz)[-row_nnz:]]

        G_sub = A[sub][:, sub]
        block = A[sub][:, sub]
        block.data = np.ones_like(block.data)
        print(f'  computing weights for block of size {len(sub)}...')
        B_sub = closed_form_slim_from_gramm(G_sub, l2_reg=l2_reg)
        B_sub = coo_matrix((B_sub[block.nonzero()], block.nonzero()))
        print('  updating...')
        B = add_submatrix(B, B_sub, where=(sub, sub), target_density=target_density)
        blocks = add_submatrix(blocks, block, where=(sub, sub))
        print(f'  dens(B) = {B.nnz / np.prod(B.shape)}...')

    print(f'  scaling B by number of blocks summed...')
    B[B.nonzero()] = B[B.nonzero()] / blocks[B.nonzero()]

    return B


def gen_submatrices(A, r_blanket=0.5, max_iter=None):
    """Generates square submatrices, represented as sets of items,
    plus binary weights indicating which items are in the present
    submatrix' "markov blanket" as defined by Steck in [1]--
    this will be 1 item if r_blanket = 0, or all if r_blanket = 1

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
        ind_list = [i for i in ind_list if i not in drop]  # np.setdiff1d doesn't preserve order
        print(f'  {len(ind_list)} remaining...')
        i += 1


def gen_submatrices_from_users(x, n_submatrices=1000, row_nnz=1000):

    col_count = np.asarray(x.sum(axis=1)).flatten()
    if n_submatrices is not None:
        # sample = np.random.choice(x.shape[0], n_submatrices, replace=False)
        sample = np.argpartition(col_count, -n_submatrices)[-n_submatrices:]
        x = x[sample]
    x = prune_rows(x, target_nnz=row_nnz)
    for user in x:
        _, sub, vals = sp.sparse.find(user)
        yield sub, vals


class Clock(object):

    def __init__(self):
        self.t0 = 0

    def tic(self):
        self.t0 = time.perf_counter()

    def toc(self):
        elapsed = time.perf_counter() - self.t0
        print(f'    elapsed: {elapsed}')

    def print_interval(self):
        self.toc()
        self.tic()


@gin.configurable
def sparse_parameter_estimation(train_data, alpha=0.75, threshold=50, rr=0.5, maxInColumn=1000, L2reg=1.0,
                                max_iter=None):
    # this implements section 3.2 in the paper

    myClock = Clock()  # JVB

    # userCount = train_data.shape[0]
    # XtX = np.asarray(train_data.T.dot(train_data).todense(), dtype=np.float32)
    # del train_data  # only the item-item data-matrix XtX is needed in the following

    # mu = np.diag(XtX) / userCount   # the mean of the columns in train_data (for binary train_data)
    # # variances of columns in train_data (scaled by userCount)
    # variance_times_userCount = np.diag(XtX) - mu * mu * userCount

    # # standardizing the data-matrix XtX (if alpha=1, then XtX becomes the correlation matrix)
    # XtX -= mu[:, None] * (mu * userCount)
    # rescaling = np.power(variance_times_userCount, alpha / 2.0)
    # scaling = 1.0 / rescaling
    # XtX = scaling[:, None] * XtX * scaling

    # XtXdiag = deepcopy(np.diag(XtX))
    # ii_diag = np.diag_indices(XtX.shape[0])

    # print("number of items: {}".format(len(mu)))
    # print("number of users: {}".format(userCount))

    # print("sparsifying the data-matrix (section 3.1 in the paper) ...")
    # myClock.tic()
    # # apply threshold
    # ix = np.where(np.abs(XtX) > threshold)
    # AA = csc_matrix((XtX[ix], ix), shape=XtX.shape, dtype=np.float32)

    # <JVB>
    print("computing gramm matrix XtX and sparsity pattern AA")
    XtX = train_data.T.dot(train_data).tocsr()
    ii_diag = np.diag_indices(XtX.shape[0])
    XtXdiag = np.asarray(XtX[ii_diag]).flatten()
    AA = XtX.tocsc().astype(np.float32)
    AA.data[np.abs(AA.data) <= threshold] = 0.0
    AA.eliminate_zeros()
    # </JVB>

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
def cluster_slim(x, l2_reg=100, target_density=0.01, n_head=None, n_clusters=None):
    """Sparse approximate *clustering-based* variant of
    the closed-form slim algorithm.

    Avoids computing the full gramm matrix.
    """
    if n_head is None:
        n_head = int(x.shape[1] * target_density ** 0.5)
    if n_clusters is None:
        n_clusters = int(target_density ** -0.5)
    head = np.argpartition(np.asarray(x.sum(axis=0)).flatten(), -n_head)[-n_head:]
    clusters = cluster_items(x.tocsr(), train_items=head, n_clusters=n_clusters)

    _, n_items = x.shape
    B = csr_matrix((n_items, n_items))
    for i in range(n_clusters):
        cluster_i = np.where(clusters == i)[0]
        sub = np.union1d(head, cluster_i)
        print(f'  getting submatrix {i+1}/{n_clusters} of size {len(sub)}...')
        x_sub = x.tocsc()[:, sub]
        print('  computing slim weights B_sub for submatrix...')
        B_sub = closed_form_slim(x_sub, l2_reg=l2_reg)
        print('  updating B...')
        B = add_submatrix(B, B_sub, where=(sub, sub), target_density=target_density)
    B.data = B.data / n_clusters

    return B


def block_slim_batches(x, l2_reg=500, target_density=0.01, block_size=3000, mass_thr=0.00):
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
        inv(G + X.T X) = P - P X.T inv(I + X P X.T) X P
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


def woodbury_slim_approx(x, l2_reg=500, batch_size=100, target_density=100.0, extra_reg=0.0):
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

    return weights


@gin.configurable
def incremental_slim(x, l2_reg=10, batch_size=50, target_density=0.01, impose_gramm_sparsity=False):
    """Closed-form SLIM with incremental computation of the inverse gramm matrix,
    and a useful approximation.

    Compute a SLIM matrix for each batch (over 'active items' only)
    and average after pruning
    """
    n_users, n_items = x.shape
    n_batches = int(np.ceil(n_users / batch_size))
    P = eye(n_items).tocsr() / l2_reg * n_batches
    for x_batch in gen_batches(x, batch_size=batch_size):
        print('  submatrix...')
        X, cols = drop_empty_cols(x_batch)
        print('  computing update...')
        B = np.linalg.inv(np.eye(x_batch.shape[0]) * l2_reg + X @ X.T)
        print('  updating...')
        XBX = np.asarray(X.T @ B @ X)
        if impose_gramm_sparsity:
            G_sub = X.T @ X
            ii, jj = G_sub.nonzero()
            P_sub_vals = - XBX[ii, jj] / l2_reg
            P_sub = coo_matrix((P_sub_vals, (ii, jj)), shape=(n_items, n_items))
        else:
            P_sub = - XBX / l2_reg
        P = add_submatrix(P, P_sub, where=(cols, cols), target_density=target_density)

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
    for x_batch in gen_batches(x, batch_size=batch_size):
        print('  submatrix...')
        XS, cols = drop_empty_cols(x_batch)
        print('  computing slim B for batch...')
        G_upd = XS.T @ XS
        B_upd = closed_form_slim_from_gramm(G_upd, l2_reg=l2_reg)
        B_upd = coo_matrix((B_upd[G_upd.nonzero()], G_upd.nonzero()), shape=B.shape)
        print('  updating...')
        B = add_submatrix(B, B_upd, where=(cols, cols), target_density=target_density)

    return B


def drop_empty_cols(X):

    active_cols = np.unique(X.tocoo().col)
    X_sub = X.tocsc()[:, active_cols]

    return X_sub, active_cols


def add_submatrix(A, dA, where=None, prune_sub=False, target_density=1.0, max_density=None):
    """Add submatrix `sub` to `A` at indices `where = (rows, cols)`.
    Optionally ensure the result is sparse with density `target_density`

    TODO: this doesn't cut it for block_slim_steck where we
    add to a matrix that contains summed submatrices before
    normalizing by the number of matrices summed--as a result
    values are not comparable across all of B
    """
    if max_density is None:
        max_density = 3 * target_density
    dA = coo_matrix(dA)
    if where is not None:
        rows, cols = where
        dA = csr_matrix((dA.data, (rows[dA.row], cols[dA.col])), shape=A.shape)
    A += dA
    if A.nnz > max_density * np.prod(A.shape):
        print(f'  density > max_density: pruning result')
        thr = get_pruning_threshold(A.tocsr(), target_density=target_density)
        A.data[np.abs(A.data) < thr] = 0.0
        A.eliminate_zeros()

    return A


def gen_batches(x, batch_size=100):

    n_examples = x.shape[0]
    n_batches = int(np.ceil(n_examples / batch_size))
    for i_batch, start in enumerate(range(0, n_examples, batch_size)):
        print('batch {}/{}...'.format(i_batch + 1, n_batches))
        end = min(start + batch_size, n_examples)
        yield x[start:end]


def cluster_items(x, train_items, affinity='cosine', n_clusters=10):
    """Use spectral clustering to assign items to clusters
    """
    if affinity == 'cosine':
        x_norm = np.asarray(x.power(2).sum(axis=0)) ** 0.5
        x_std = x.multiply(1./x_norm).tocsc()
        affinity_to_train_items = x_std.T @ x_std[:, train_items]
    elif affinity == 'gramm':
        affinity_to_train_items = x.T @ x[:, train_items]
    else:
        raise ValueError('affinity must be `cosine` or `gramm`')

    train_affinities = affinity_to_train_items[train_items]
    mod = SpectralClustering(n_clusters, affinity='precomputed', assign_labels='discretize')
    train_item_clusters = mod.fit_predict(train_affinities)
    closest_train_items = np.asarray(affinity_to_train_items.argmax(axis=1)).flatten()
    clusters = train_item_clusters[closest_train_items]
    
    return clusters
