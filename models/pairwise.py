
import gin
import numpy as np
import tensorflow as tf

from util import load_weights_biases


@gin.configurable
class PairwiseSLIM(object):
    """SLIM-like model with sparsity constraint and pairwise loss

    TODO / try next:
    - find the right scale to start fine-tuning (e.g. way down) / balance with reg_term
    - track train loss, somehow, to know if we're overfitting
    - different loss functions over the pairs e.g.
        relu(neg - pos)
        (neg - pos) ** 2
    """
    def __init__(self, batch_size, n_items, weights_path=None, zero_diag=True, row_nnz=100,
                 n_hist=100, keep_prob=1.0, lam=0.01, lr=3e-4, random_seed=None,
                 randomize=False, n_pairs=100, Optimizer=tf.compat.v1.train.AdamOptimizer,
                 rescale=False):
        w_init, b_init = load_weights_biases(weights_path)  # w**2.sum() ~ 330
        if randomize:
            w_init.data = 0.01 * np.random.randn(*w_init.data.shape)
            b_init = np.zeros(n_items)
        if b_init is None:
            b_init = np.zeros(n_items)
        if rescale:  # logit(5.89 x - 2.94) maps to x = 0 to 0.05 and x = 1 to 0.95
            w_init *= 5.89
            b_init -= 2.94

        # make init from (single) weights file, break if weights not square
        assert w_init.shape[0] == w_init.shape[1] == n_items
        inds, vals = sparse_inds_vals(w_init, row_nnz=row_nnz, zero_diag=zero_diag)
        self.batch_size = batch_size
        self.n_items = n_items
        self.n_hist = n_hist
        self.n_pairs = n_pairs
        self.lam = lam
        self.lr = lr
        self.random_seed = random_seed

        # placeholders and weights
        self.input_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[batch_size, w_init.shape[1]])
        self.label_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[batch_size, w_init.shape[1]])
        self.keep_prob_ph = tf.compat.v1.placeholder_with_default(keep_prob, shape=None)
        self.w_inds = tf.Variable(inds.astype(np.int64))
        self.w_vals = tf.Variable(vals.astype(np.float32))
        self.bias = tf.Variable(b_init.astype(np.float32))

        # build graph
        self.logits = self.forward()
        self.loss = self.pairwise_loss(self.logits) + self.reg_term()
        self.train_op = Optimizer(self.lr).minimize(self.loss)
        self.saver = tf.compat.v1.train.Saver()

    def forward(self):
        """Construct forward graph (up until logits)

        TODO: figure out why our more efficient embedding lookup doesn't work:
            vals, inds = tf.math.top_k(x, k=self.n_hist)
            hist_emb = sparse_embeddings(self.w_inds, self.w_vals, d=self.n_items, inds=inds)
            user_emb = tf.sparse.reduce_sum(hist_emb * tf.expand_dims(vals, -1), axis=1)
        """
        x = tf.nn.dropout(self.input_ph, rate=1-self.keep_prob_ph)

        w = csr_matrix(self.w_inds, self.w_vals, n_cols=self.n_items)
        y = tf.transpose(tf.sparse.sparse_dense_matmul(w, x, adjoint_a=True, adjoint_b=True))
        if self.bias is not None:
            y += tf.expand_dims(self.bias, 0)

        return y

    def pairwise_loss(self, logits):
        """Pairwise loss from logits
        """
        labels_pos = tf.cast(tf.equal(self.label_ph, 1.), dtype=np.float32)
        labels_neg = tf.cast(tf.equal(self.label_ph, 0.), dtype=np.float32)
        pos = tf.random.categorical(tf.math.log(labels_pos), self.n_pairs)
        neg = tf.random.categorical(tf.math.log(labels_neg), self.n_pairs)
        y_pos = tf.gather(logits, pos, batch_dims=-1)
        y_neg = tf.gather(logits, neg, batch_dims=-1)

        # log loss over pairs, ~ a soft relu of y_neg - y_pos
        # pairwise_losses = tf.math.log(1 + tf.math.exp(y_neg - y_pos))
        # losses = tf.reduce_sum(pairwise_losses, axis=1)
        # loss = tf.reduce_mean(losses)
        loss = tf.nn.l2_loss(1 - (y_pos - y_neg))  # TODO: revert (experiment)

        return loss

    def reg_term(self):

        # apply regularization to weights
        return self.lam * tf.nn.l2_loss(self.w_vals)

    def save(self, sess, log_dir):
        """TODO subclass AutoEncoder so we don't need this?
        """
        self.saver.save(sess, '{}/model'.format(log_dir))

    def restore(self, sess, log_dir):
        """TODO subclass AutoEncoder so we don't need this?
        """
        self.saver.restore(sess, '{}/model'.format(log_dir))


def sparse_inds_vals(init, row_nnz, zero_diag=False):

    init = init.tocoo()
    if zero_diag:
        init.setdiag(0.0)
        init.eliminate_zeros()

    inds = []
    vals = []
    for row in init.tocsr():
        if len(row.data) > row_nnz:
            thr = np.partition(np.abs(row.data), kth=-row_nnz)[-row_nnz]
            row.data[np.abs(row.data) < thr] = 0.0
            row.eliminate_zeros()
        ii = np.argsort(-row.data)[:row_nnz]
        padding = [0, row_nnz - len(ii)]
        inds.append(np.pad(row.indices[ii], padding))
        vals.append(np.pad(row.data[ii], padding))
    inds = np.asarray(inds)
    vals = np.asarray(vals)

    return inds, vals


def csr_matrix(inds, vals, n_cols):
    """Construct sparse float32 matrix from indices and values in a simple CSR-like format

    - inds: 2d-tensor of int64
        inds[i] holds the column inds for row i of the CSR matrix
    - vals: 2d-tensor of float32
        vals[i] holds the values for row i and cols inds[i] of the CSR matrix
    - n_cols: number of colums in the CSR_matrix
        number of rows is inds.shape[0]
    """
    n_rows, row_nnz = vals.shape
    row_inds = tf.repeat(tf.range(n_rows, dtype=np.int64), row_nnz)
    col_inds = tf.reshape(inds, [-1])
    inds = tf.transpose(tf.stack([row_inds, col_inds]))
    vals = tf.reshape(vals, [-1])

    return tf.sparse.SparseTensor(inds, vals, dense_shape=[n_rows, n_cols])


def csr_tensor(inds, vals, d2):
    """Construct 3d sparse tensor from 3d indices and 3d values in a CSR-like format

    ~ stack([csr_matrix(i, v), 0) for i, v in zip(inds, vals)])
    except that we cannot iterate over inds and vals like this

    - inds: 3d-tensor of type np.int64
        inds[i, j] holds the inds corresponding to non-zeros in out[i, j, :]
    - vals: 3d-tensor
        inds[i, j] holds the non-zeros values in out[i, j, :]
    - d2: last dimension of the 3d-tensor
        first two dimensions will be inds.shape[:2]
    """
    d0, d1, nnz = inds.shape

    # fn = lambda args: csr_matrix(args[0], args[1], d2)
    # out_sig = tf.SparseTensorSpec(shape=[d1, d2], dtype=np.float32)
    # out = tf.map_fn(fn, (inds, vals), fn_output_signature=out_sig)

    inds0, inds1, _ = tf.meshgrid(tf.range(d0, dtype=np.int64),
                                  tf.range(d1, dtype=np.int64),
                                  tf.range(nnz, dtype=np.int64))
    inds = tf.transpose(tf.stack([tf.reshape(i, [-1]) for i in [inds0, inds1, inds]]))
    out = tf.sparse.SparseTensor(inds, tf.reshape(vals, [-1]), dense_shape=[d0, d1, d2])

    return out


def sparse_embeddings(w_inds, w_vals, d, inds):

    w_inds = tf.gather(w_inds, inds)
    w_vals = tf.gather(w_vals, inds)

    return csr_tensor(w_inds, w_vals, d2=d)
