
from collections import defaultdict

import gin
import time
import numpy as np
import tensorflow as tf
from scipy.sparse import hstack, vstack

from losses import mse, neg_ll, pairwise_loss
from models.base import BaseRecommender
from models.pairwise import PairwiseSLIM
# from models.rigl import MaskedAutoencoder
from util import Logger, load_weights_biases, gen_batches, to_float32, prune_rows

gin.external_configurable(tf.compat.v1.train.GradientDescentOptimizer)
gin.external_configurable(tf.compat.v1.train.AdamOptimizer)

tf.compat.v1.disable_eager_execution()


@gin.configurable
class TFRecommender(BaseRecommender):

    def __init__(self, log_dir=None, Model=None, batch_size=100, exact_batches=False, n_epochs=10):
        """Build a TF-based auto-encoder model with given initial weights.
        """
        self.log_dir = log_dir
        self.Model = Model
        self.batch_size = batch_size
        self.exact_batches = exact_batches
        self.n_epochs = n_epochs
        if Model is None:
            self.Model = AutoEncoder

        self.logger = None
        self.sess = None

    def train(self, x_train, y_train, x_val, y_val):
        """Train a tensorflow recommender"""

        t1 = time.perf_counter()
        tf.compat.v1.reset_default_graph()
        self.model = self.Model(n_items=x_train.shape[1])
        best_ndcg = 0.0
        with tf.compat.v1.Session() as self.sess:
            self.logger = TFLogger(self.log_dir, self.sess)
            self.logger.log_config(gin.operative_config_str())
            self.sess.run(self.model.init_ops)

            metrics = self.evaluate(x_val, y_val)
            for epoch in range(self.n_epochs):
                print(f'Training. Epoch = {epoch + 1}/{self.n_epochs}')
                self.train_one_epoch(x_train, y_train)
                print('Evaluating...')
                metrics = self.evaluate(x_val, y_val, other_metrics={'epoch': epoch})

                if metrics['ndcg'] > best_ndcg:
                    best_ndcg = metrics['ndcg']
                    self.model.save(self.sess, log_dir=self.logger.log_dir)

        self.sess = None
        metrics['train_time'] = time.perf_counter() - t1

        return metrics

    def train_one_epoch(self, x_train, y_train, x_val=None, y_val=None,
                        print_interval=1):
        for x, y in gen_batches(x_train, y_train, batch_size=self.batch_size, exact=self.exact_batches):
            x, y = self.prepare_batch(x, y)
            feed_dict = {self.model.input_ph: x, self.model.label_ph: y}
            _, train_loss = self.sess.run([self.model.train_op, self.model.loss], feed_dict=feed_dict)
            self.logger.log_to_tensorboard({'train_loss': train_loss})

    def predict(self, x, y=None):
        """Predict scores.
        If y is not None, also return a loss.
        """
        x, y = self.prepare_batch(x, y)
        feed_dict = {self.model.input_ph: x, self.model.label_ph: y, self.model.keep_prob_ph: 1.0}
        y_pred, loss = self.sess.run([self.model.logits, self.model.loss], feed_dict=feed_dict)

        return y_pred, loss

    def prepare_batch(self, x, y=None):
        """Convert a batch of x and y to a sess.run-compatible format"""
        x = to_float32(x, to_dense=True)
        if y is None:
            y = to_float32(x, to_dense=True)
        else:
            y = to_float32(y, to_dense=True)

        return x, y

    def evaluate(self, x_val, y_val, other_metrics=None, test=False):
        """Wrapper around BaseRecommender.evaluate that restores the model
        from the log directory if the session is None.

        Also compute and report model sparsity.
        """
        if other_metrics is None:
            other_metrics = {}
        if self.sess is None:
            with tf.compat.v1.Session() as self.sess:
                self.model.restore(self.sess, log_dir=self.logger.log_dir)
                other_metrics['weight_density'] = self.model.get_density(self.sess)
                results = super().evaluate(x_val, y_val, other_metrics, test)
            self.sess = None
        else:
            other_metrics['weight_density'] = self.model.get_density(self.sess)
            results = super().evaluate(x_val, y_val, other_metrics, test)

        return results


@gin.configurable
class AutoEncoder(object):
    """Simple Autoencoder

    TODO: make lam ~ 1/batch_size (now done manually in gin files)
    """
    def __init__(self, n_items, n_layers=2, latent_dim=100, use_biases=True,
                 normalize_inputs=False, tanh=True, loss="nll",
                 keep_prob=0.5, lam=0.01, lr=3e-4, random_seed=None,
                 Optimizer=tf.compat.v1.train.AdamOptimizer):

        self.n_items = n_items
        self.n_layers = n_layers
        self.latent_dim = latent_dim
        self.use_biases = use_biases
        self.normalize_inputs = normalize_inputs
        self.tanh = tanh
        self.loss = loss
        self.lam = lam
        self.lr = lr
        self.random_seed = random_seed

        # loss
        loss_functions = {"mse": mse,
                          "nll": neg_ll,
                          "cxe": neg_ll,
                          "bce": tf.nn.sigmoid_cross_entropy_with_logits,
                          "bxe": tf.nn.sigmoid_cross_entropy_with_logits,
                          "pairwise": pairwise_loss}
        loss_fn = loss_functions[self.loss]

        # placeholders and weights
        self.input_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, n_items])
        self.label_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, n_items])
        self.keep_prob_ph = tf.compat.v1.placeholder_with_default(keep_prob, shape=None)
        self.weights, self.biases = self.construct_weights()

        # build graph
        self.logits = self.forward_pass()
        self.loss = tf.reduce_mean(
            loss_fn(labels=self.label_ph, logits=self.logits)) + self.reg_term()
        self.train_op = Optimizer(self.lr).minimize(self.loss)
        self.saver = tf.compat.v1.train.Saver()
        self.init_ops = [tf.compat.v1.global_variables_initializer()]

    def construct_weights(self):

        assert self.n_items is not None
        dims = [self.n_items] + [self.latent_dim] * (self.n_layers - 1) + [self.n_items]
        weights = []
        biases = []

        # define weights
        for i in range(self.n_layers):
            weight_key = "weight_{}to{}".format(i, i+1)
            bias_key = "bias_{}".format(i+1)

            shape = (dims[i], dims[i + 1])
            initializer = tf.contrib.layers.xavier_initializer(seed=self.random_seed)
            w = tf.compat.v1.get_variable(name=weight_key, shape=shape, initializer=initializer)
            weights.append(w)
            tf.compat.v1.summary.histogram(weight_key, w)

            if self.use_biases:
                shape = (dims[i + 1],)
                initializer = tf.truncated_normal_initializer(stddev=0.001, seed=self.random_seed)
                b = tf.compat.v1.get_variable(name=bias_key, shape=shape, initializer=initializer)
                biases.append(b)
                tf.compat.v1.summary.histogram(bias_key, biases[-1])

        return weights, biases

    def forward_pass(self):
        # construct forward graph
        if self.normalize_inputs:
            h = tf.nn.l2_normalize(self.input_ph, 1)
        else:
            h = self.input_ph

        if self.keep_prob_ph != 1.0:
            h = tf.nn.dropout(h, rate=1-self.keep_prob_ph)

        for i, w in enumerate(self.weights):
            h = tf.matmul(h, w)
            if len(self.biases):
                h = h + self.biases[i]
            if self.tanh and i != len(self.weights) - 1:
                h = tf.nn.tanh(h)

        return h

    def reg_term(self):

        reg_var = self.lam * tf.add_n([tf.nn.l2_loss(w) for w in self.weights + self.biases])

        return 2 * reg_var

    def get_density(self, sess=None):

        return 1.0

    def save(self, sess, log_dir):

        self.saver.save(sess, '{}/model'.format(log_dir))

    def restore(self, sess, log_dir):

        self.saver.restore(sess, '{}/model'.format(log_dir))


@gin.configurable
class SparseAutoEncoder(object):
    """Autoencoder from a list of initial weights matrices. Sparse by default.
    """
    def __init__(self, n_items, weights_path=None, n_layers=1,
                 row_nnz=None, randomize_inits=False, use_biases=True,
                 normalize_inputs=False, shared_weights=False, loss="mse",
                 keep_prob=1.0, lam=0.01, lr=3e-4, random_seed=None,
                 init_alpha=None, Optimizer=tf.compat.v1.train.AdamOptimizer):
        weights, biases = load_weights_biases(weights_path)

        # make init from (single) weights file break if n_layers > 1 and weights not square
        assert weights.shape[0] == n_items, f"Weights don't match dataset. weights.shape[0] = {weights.shape[0]}, n_items = {n_items}"
        assert n_layers > 2 or weights.shape[0] == weights.shape[1]
        w_inits = [weights] * n_layers
        b_inits = [biases] * n_layers

        self.w_inits = w_inits
        print(f'b_inits: {b_inits}')
        print(f'w_inits: {w_inits}')
        self.b_inits = [None] * len(w_inits) if b_inits is None else b_inits
        self.row_nnz = row_nnz
        self.randomize_inits = randomize_inits
        self.use_biases = use_biases
        self.normalize_inputs = normalize_inputs
        self.shared_weights = shared_weights
        self.loss = loss
        self.lam = lam
        self.lr = lr
        self.random_seed = random_seed
        self.init_b = 0. if init_alpha is None else - np.log(1. / init_alpha - 1.)
        self.init_a = 1. if init_alpha is None else - np.log(1. / (1. - init_alpha) - 1.) - self.init_b

        # loss
        loss_functions = {"mse": mse,
                          "nll": neg_ll,
                          "cxe": neg_ll,
                          "bce": tf.nn.sigmoid_cross_entropy_with_logits,
                          "bxe": tf.nn.sigmoid_cross_entropy_with_logits,
                          "pairwise": pairwise_loss}
        loss_fn = loss_functions[self.loss]

        # placeholders and weights
        self.input_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, w_inits[0].shape[1]])
        self.label_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, w_inits[0].shape[1]])
        self.keep_prob_ph = tf.compat.v1.placeholder_with_default(keep_prob, shape=None)
        self.weights, self.biases = self.construct_weights()

        # build graph
        self.logits = self.forward_pass()
        self.loss = tf.reduce_mean(
            loss_fn(labels=self.label_ph, logits=self.logits)) + self.reg_term()
        self.train_op = Optimizer(self.lr).minimize(self.loss)
        self.init_ops = [tf.compat.v1.global_variables_initializer()]
        self.saver = tf.compat.v1.train.Saver()

    def construct_weights(self):

        weights = []
        biases = []

        # define weights
        for i, (w_init, b_init) in enumerate(zip(self.w_inits, self.b_inits)):
            weight_key = "weight_{}to{}".format(i, i+1)
            bias_key = "bias_{}".format(i+1)
            w_init *= self.init_a
            if i == 0 or not self.shared_weights:
                w = sparse_tensor_from_init(w_init, randomize=self.randomize_inits,
                                            zero_diag=True, name=weight_key, row_nnz=self.row_nnz)
            weights.append(w)

            if self.use_biases:
                if b_init is None:
                    b_init = np.zeros(w_init.shape[-1])
                else:
                    if self.randomize_inits:
                        b_init = mul_noise(b_init)
                b_init += self.init_b
                b = tf.Variable(b_init.astype(np.float32), name=bias_key)
                tf.compat.v1.summary.histogram(bias_key, b)
                biases.append(b)
        print([w.values.shape for w in weights])
        print([b.shape for b in biases])

        return weights, biases

    def forward_pass(self):
        # construct forward graph

        h = self.process_inputs()
        h = self.encoder(h)

        return self.decoder(h)

    def process_inputs(self):

        h = self.input_ph
        if self.normalize_inputs:
            h = tf.nn.l2_normalize(self.input_ph, 1)
        if self.keep_prob_ph != 1.0:
            h = tf.nn.dropout(h, rate=1-self.keep_prob_ph)

        return h

    def encoder(self, h):

        for i, w in enumerate(self.weights[:-1]):
            h = tf.transpose(tf.sparse.sparse_dense_matmul(w, h, adjoint_a=True, adjoint_b=True))
            if len(self.biases):
                h = h + self.biases[i]
            h = tf.nn.tanh(h)

        return h

    def decoder(self, h):

        # single sparse layer
        h = tf.transpose(tf.sparse.sparse_dense_matmul(self.weights[-1], h, adjoint_a=True, adjoint_b=True))
        if len(self.biases):
            h = h + self.biases[-1]

        return h

    def reg_term(self):

        reg_var = self.lam * tf.add_n([tf.nn.l2_loss(w) for w in [w.values for w in self.weights] + self.biases])

        return 2 * reg_var

    def get_density(self, sess=None):

        nnzs, sizes = [], []
        for w_init, b_init in zip(self.w_inits, self.b_inits):
            nnzs.extend([w_init.nnz, np.size(b_init)])
            sizes.extend([np.size(w_init), np.size(b_init)])

        return np.sum(nnzs) / np.sum(sizes)

    def save(self, sess, log_dir):
        """TODO subclass AutoEncoder so we don't need this?
        """
        self.saver.save(sess, '{}/model'.format(log_dir))

    def restore(self, sess, log_dir):
        """TODO subclass AutoEncoder so we don't need this?
        """
        self.saver.restore(sess, '{}/model'.format(log_dir))


@gin.configurable
class GraphUNet(SparseAutoEncoder):

    def __init__(self, n_items, weights_path=None,
                 latent_dim=2000, n_channels=10, n_conv_layers=1, shared_weights=False,
                 randomize_inits=False, normalize_inputs=False, loss="mse",
                 keep_prob=1.0, lam=0.01, lr=3e-4, random_seed=None,
                 Optimizer=tf.compat.v1.train.AdamOptimizer):
        """Graph neural network with bottleneck, from a list of initial weights matrices.
        Sparse by default.

        NOTE: in current implementation, any pretrained biases are discarded
        """
        self.latent_dim = latent_dim
        self.n_channels = n_channels
        self.n_conv_layers = n_conv_layers
        super().__init__(n_items, weights_path=weights_path, n_layers=1,
                         randomize_inits=randomize_inits, use_biases=True,
                         normalize_inputs=normalize_inputs, shared_weights=False, loss=loss,
                         keep_prob=keep_prob, lam=lam, lr=lr, random_seed=random_seed,
                         init_alpha=None, Optimizer=tf.compat.v1.train.AdamOptimizer)
        self.diffuser = None
        if shared_weights:
            pruned_graph = prune_rows(self.w_inits[0], target_nnz=3).tocoo()
            inds = np.vstack([pruned_graph.row, pruned_graph.col]).T.astype(np.int64)
            vals = np.ones_like(pruned_graph.data)
            self.diffuser = tf.sparse.SparseTensor(tf.constant(inds), tf.constant(vals), dense_shape=pruned_graph.shape)
            self.diffuser = tf.sparse.reorder(self.diffuser)

    def construct_weights(self):
        """Take the single scipy-sparse weight init matrix w_sp = w_inits[0] and
        tile w_sp[:, :latent_dim] `n_channels` times in the column dimension.
        These `n_channels` copies can then be stacked in a third dimension once the weight tensors
        are constructed from scipy.sparse inits (which are necessarily 2D).

        See function tile_sparse_weights for more on how the tiling is done.
        """
        assert len(self.w_inits) == 1
        w_tiled = tile_sparse_weights(self.w_inits[0], max_cols=self.latent_dim, tile_cols=self.n_channels)
        self.w_inits = [0.01 * w_tiled, 100. * w_tiled.T]
        self.b_inits = [None, None]

        return super().construct_weights()

    def encoder(self, h):

        if self.shared_weights:
            h = h @ self.diffuser
        super().encoder(h)
        if self.n_conv_layers == 0:
            return h

        # apply conv layers
        h = tf.transpose(tf.reshape(h, [-1, self.n_channels, self.latent_dim]), [0, 2, 1])
        for _ in range(self.n_conv_layers):
            h = tf.compat.v1.layers.conv1d(h, filters=self.n_channels, kernel_size=1,
                                           use_bias=True, activation=tf.nn.relu)
        h = tf.reshape(tf.transpose(h, [0, 2, 1]), [-1, self.latent_dim * self.n_channels])

        return h


class TFLogger(Logger):

    def __init__(self, base_dir, sess):
        """Logger class that also writes summaries to Tensorboard"""
        super().__init__(base_dir)

        self.sess = sess
        self.variables = {}
        self.summaries = {}
        self.summary_writer = tf.compat.v1.summary.FileWriter(self.log_dir, graph=tf.compat.v1.get_default_graph())
        self.history = defaultdict(list)

    def log_metrics(self, metrics, config=None, step=None, test=False):
        """Log a dictionary of metrics to a tf.compat.v1.summary.FileWriter
        """
        super().log_metrics(metrics, config=config, step=step, test=test)
        if not test:
            self.log_to_tensorboard(metrics)

    def log_to_tensorboard(self, metrics, step=None):

        feed_dict = {}
        summaries = []
        with tf.compat.v1.get_default_graph().as_default():
            for name, value in metrics.items():
                if name not in self.variables:
                    self.variables[name] = tf.Variable(0.0, name=name)
                    self.summaries[name] = tf.compat.v1.summary.scalar(name, self.variables[name])
                    self.sess.run(tf.compat.v1.variables_initializer([self.variables[name]]))
                summaries.append(self.summaries[name])
                feed_dict[self.variables[name]] = value
            summaries = self.sess.run(summaries, feed_dict=feed_dict)
            for name, summary in zip(metrics.keys(), summaries):
                self.history[name].append(summary)
                if step is None:
                    step = len(self.history[name])
                self.summary_writer.add_summary(summary, global_step=step)


def tile_sparse_weights(w_sp, max_cols=None, tile_cols=1, max_rows=None, tile_rows=1, row_nnz=None):
    """Tile sparse weights.

    First selects the first `max_rows` rows and `max_cols` rows and cols,
    then, if tile_rows > 1, tiles w_sp `tile_rows` times, vertically
    and if tile_cols > 1, tiles w_sp `tile_cols` times, horizontally
    """
    if max_rows is not None or max_cols is not None:
        w_sp = w_sp[:max_rows, :max_cols]
    if tile_rows > 1:
        w_sp = vstack([w_sp] * tile_rows)
    if tile_cols > 1:
        w_sp = hstack([w_sp] * tile_cols)

    return w_sp


def sparse_tensor_from_init(w_sp, name='sparse_weight', randomize=False, zero_diag=False, row_nnz=None):
    """Create a sparse tensor from a sparse scipy array w_sp.
    """
    w_sp = w_sp.tocoo()
    if zero_diag:
        w_sp.setdiag(0.0)
        w_sp.eliminate_zeros()
    if row_nnz is not None:
        w_sp = prune_rows(w_sp, target_nnz=row_nnz).tocoo()
    if randomize:
        w_sp.data = mul_noise(w_sp.data)
    inds = list(zip(w_sp.row, w_sp.col))
    data = w_sp.data.astype(np.float32)

    w_inds = tf.convert_to_tensor(inds, dtype=np.int64)
    w_data = tf.Variable(data, name=name + '_data')
    w = tf.SparseTensor(w_inds, tf.identity(w_data), dense_shape=w_sp.shape)
    w = tf.sparse.reorder(w, name=name)  # as suggested here:
    # https://www.tensorflow.org/api_docs/python/tf/sparse/SparseTensor?version=stable

    #  summary for tensorboard
    tf.compat.v1.summary.histogram(name, w_data)

    return w


def mul_noise(x, eps=0.001):

    return eps * np.maximum(np.minimum(np.random.randn(*x.shape), 3.), -3.)
    # return np.random.exponential(eps) * x
