
from collections import defaultdict

import gin
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import apply_regularization, l2_regularizer

from models.base import BaseRecommender
from util import Logger, load_weights, gen_batches, to_float32

gin.external_configurable(tf.train.GradientDescentOptimizer)
gin.external_configurable(tf.train.AdamOptimizer)


@gin.configurable
class TFRecommender(BaseRecommender):

    def __init__(self, log_dir=None, weights_path=None, n_layers=1,
                 batch_size=100, n_epochs=10):
        """Build a TF-based wide auto-encoder model with given initial weights.

        TODO:
        - model snapshots (check lines containing "best_ndcg" in Liang's notebook)
        """
        weights, biases = load_weights(weights_path)
        w_inits = [weights] * n_layers
        b_inits = [biases] * n_layers

        tf.reset_default_graph()
        self.model = WAE(w_inits, b_inits=b_inits)
        self.log_dir = log_dir
        self.batch_size = batch_size
        self.n_epochs = n_epochs

        self.logger = None
        self.sess = None

    def train(self, x_train, y_train, x_val, y_val):
        """Train a tensorflow recommender"""
        with tf.Session() as self.sess:
            self.logger = TFLogger(self.log_dir, self.sess)
            self.logger.log_config(gin.operative_config_str())

            init = tf.global_variables_initializer()
            self.sess.run(init)

            metrics = self.evaluate(x_val, y_val)
            for epoch in range(self.n_epochs):
                print(f'Training. Epoch = {epoch + 1}/{self.n_epochs}')
                self.train_one_epoch(x_train, y_train)
                print('Evaluating...')
                metrics = self.evaluate(x_val, y_val)

        return metrics

    def train_one_epoch(self, x_train, y_train, x_val=None, y_val=None,
                        print_interval=1):
        for x, y in gen_batches(x_train, y_train, batch_size=self.batch_size):
            x, y = self.prepare_batch(x, y)
            feed_dict = {self.model.input_ph: x, self.model.label_ph: y}
            summary_train, _ = self.sess.run([self.model.summaries, self.model.train_op],
                                             feed_dict=feed_dict)
            self.logger.log_summaries({'summary': summary_train})

    def predict(self, x, y=None):
        """Predict scores.
        If y is not None, also return a loss.
        """
        x, y = self.prepare_batch(x, y)
        if y is not None:
            feed_dict = {self.model.input_ph: x, self.model.label_ph: y}
            y_pred, loss = self.sess.run([self.model.logits, self.model.loss],
                                         feed_dict=feed_dict)
        else:
            feed_dict = {self.model.input_ph: x}
            y_pred = self.sess.run(self.model.logits, feed_dict=feed_dict)
            loss = None

        return y_pred, loss

    def prepare_batch(self, x, y=None):
        """Convert a batch of x and y to a sess.run-compatible format"""
        x = to_float32(x, to_dense=True)
        if y is not None:
            y = to_float32(y, to_dense=True)

        return x, y


@gin.configurable
class WAE(object):

    def __init__(self, w_inits, b_inits=None,
                 randomize_inits=False, use_biases=True,
                 normalize_inputs=False, shared_weights=False, loss="mse",
                 keep_prob=1.0, lam=0.01, lr=3e-4, random_seed=None,
                 Optimizer=tf.train.AdamOptimizer):

        self.w_inits = w_inits
        self.b_inits = [None] * len(w_inits) if b_inits is None else b_inits
        self.randomize_inits = randomize_inits
        self.use_biases = use_biases
        self.normalize_inputs = normalize_inputs
        self.shared_weights = shared_weights
        self.loss = loss
        self.lam = lam
        self.lr = lr
        self.random_seed = random_seed

        # loss
        loss_functions = {"mse": mse,
                          "nll": neg_ll,
                          "bce": tf.nn.sigmoid_cross_entropy_with_logits}
        loss_fn = loss_functions[self.loss]

        # placeholders and weights
        self.input_ph = tf.placeholder(dtype=tf.float32, shape=[None, w_inits[0].shape[1]])
        self.label_ph = tf.placeholder(dtype=tf.float32, shape=[None, w_inits[0].shape[1]])
        self.keep_prob_ph = tf.placeholder_with_default(keep_prob, shape=None)
        self.weights, self.biases = self.construct_weights()

        # build graph
        self.logits = self.forward_pass()
        self.loss = tf.reduce_mean(
            loss_fn(labels=self.label_ph, logits=self.logits)) + self.reg_term()
        self.train_op = Optimizer(self.lr).minimize(self.loss)
        self.saver = tf.train.Saver()

        # add summary statistics
        tf.summary.scalar('loss', self.loss)
        self.summaries = tf.summary.merge_all()

    def construct_weights(self):

        weights = []
        biases = []

        # define weights
        for i, (w_init, b_init) in enumerate(zip(self.w_inits, self.b_inits)):
            weight_key = "weight_{}to{}".format(i, i+1)
            bias_key = "bias_{}".format(i+1)

            if i == 0 or not self.shared_weights:
                w = sparse_tensor_from_init(w_init, randomize=self.randomize_inits,
                                            zero_diag=True, name=weight_key)
            weights.append(w)

            if self.use_biases:
                if b_init is None:
                    b_init = np.zeros([w_init.shape[0]])
                elif self.randomize_inits:
                    b_init = mul_noise(b_init)
                biases.append(tf.Variable(b_init.astype(np.float32), name=bias_key))
                tf.summary.histogram(bias_key, biases[-1])

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
            h = tf.transpose(tf.sparse.sparse_dense_matmul(w, h, adjoint_a=True, adjoint_b=True))
            if len(self.biases):
                h = h + self.biases[i]
            if i != len(self.weights) - 1:
                h = tf.nn.tanh(h)

        return h

    def reg_term(self):

        # apply regularization to weights
        reg = l2_regularizer(self.lam)
        reg_var = apply_regularization(reg, [w.values for w in self.weights] + self.biases)

        # tensorflow l2 regularization multiply 0.5 to the l2 norm
        # multiply 2 so that it is back in the same scale
        return 2 * reg_var


class TFLogger(Logger):

    def __init__(self, base_dir, sess):
        """Logger class that also writes summaries to Tensorboard"""
        super().__init__(base_dir)

        self.sess = sess
        self.variables = {}
        self.summaries = {}
        self.summary_writer = tf.summary.FileWriter(self.log_dir, graph=tf.get_default_graph())
        self.history = defaultdict(list)

    def log_metrics(self, metrics, config=None):
        """Log a dictionary of metrics to a tf.summary.FileWriter
        """
        super().log_metrics(metrics, config=config)

        feed_dict = {}
        summaries = []
        for name, value in metrics.items():
            if name not in self.variables:
                self.variables[name] = tf.Variable(0.0, name=name)
                self.summaries[name] = tf.summary.scalar(name, self.variables[name])
            summaries.append(self.summaries[name])
            feed_dict[self.variables[name]] = value
        summaries = self.sess.run(summaries, feed_dict=feed_dict)
        summaries_dict = dict(zip(metrics.keys(), summaries))
        self.log_summaries(summaries_dict)

    def log_summaries(self, summaries, step=None):

        for name, summary in summaries.items():
            self.history[name].append(summary)
            if step is None:
                step = len(self.history[name])
            self.summary_writer.add_summary(summary, global_step=step)


def sparse_tensor_from_init(init, name='sparse_weight', randomize=False, zero_diag=False):

    init = init.tocoo()
    if randomize:
        init.data = mul_noise(init.data)
    if zero_diag:
        init.setdiag(0.0)
        init.eliminate_zeros()

    inds = list(zip(init.row, init.col))
    data = init.data.astype(np.float32)
    w_inds = tf.convert_to_tensor(inds, dtype=np.int64)
    w_data = tf.Variable(data, name=name)
    w = tf.SparseTensor(w_inds, tf.identity(w_data),
                        dense_shape=init.shape)
    w = tf.sparse.reorder(w)  # as suggested here:
    # https://www.tensorflow.org/api_docs/python/tf/sparse/SparseTensor?version=stable

    #  summary for tensorboard
    tf.summary.histogram(name, w_data)

    return w


def mul_noise(x, eps=0.01):

    # return = eps * np.random.randn(*x.shape)
    return np.random.exponential(eps) * x


def neg_ll(logits, labels):

    log_softmax_var = tf.nn.log_softmax(logits)
    return -tf.reduce_sum(log_softmax_var * labels, axis=1)


def mse(labels, logits):

    return tf.square(labels - logits)
