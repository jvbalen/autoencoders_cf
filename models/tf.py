
import os
import datetime
from collections import defaultdict

import gin
import numpy as np
import tensorflow as tf
from scipy.sparse import issparse
from tensorflow.contrib.layers import apply_regularization, l2_regularizer

from metric import ndcg_binary_at_k_batch, recall_at_k_batch, binary_crossentropy_from_logits
from util import load_weights


@gin.configurable
class TFRecommender(object):

    def __init__(self, log_dir=None, n_layers=1, weights_path=None,
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

    def train(self, x_train, y_train, x_val, y_val):
        """Train a tensorflow recommender"""
        with tf.Session() as sess:
            logger = MetricLogger(self.log_dir, sess) if self.log_dir is not None else None

            init = tf.global_variables_initializer()
            sess.run(init)

            metrics = self.evaluate(sess, x_val, y_val, logger=logger)
            print('Validation NDCG = {}'.format(metrics))

            for epoch in range(self.n_epochs):
                print('Training. Epoch = {}/{}'.format(epoch + 1, self.n_epochs))
                self.train_one_epoch(sess, x_train, y_train, logger=logger)

                metrics = self.evaluate(sess, x_val, y_val, logger=logger)
                print('Validation NDCG = {}'.format(metrics))

        return metrics

    def train_one_epoch(self, sess, x_train, y_train, x_val=None, y_val=None,
                        print_interval=1, logger=None):

        n_train = x_train.shape[0]
        train_inds = list(range(n_train))

        np.random.shuffle(train_inds)
        for i_batch, start in enumerate(range(0, n_train, self.batch_size)):
            end = min(start + self.batch_size, n_train)
            if i_batch % print_interval == 0:
                print('batch {}/{}...'.format(i_batch + 1, int(n_train / self.batch_size)))

            x = x_train[train_inds[start:end]]
            y = y_train[train_inds[start:end]]
            feed_dict = {self.model.input_ph: prepare_batch(x),
                         self.model.label_ph: prepare_batch(y)}
            summary_train, _ = sess.run([self.model.summaries, self.model.train_op],
                                        feed_dict=feed_dict)

            if logger is not None:
                logger.log_summaries({'summary': summary_train})

    def evaluate(self, sess, x_val, y_val, logger=None):
        """Evaluate model on observed and unobserved validation data x_val, y_val
        """
        n_val = x_val.shape[0]
        val_inds = list(range(n_val))

        loss_list = []
        ndcg_list = []
        r100_list = []
        bce_list = []
        for i_batch, start in enumerate(range(0, n_val, self.batch_size)):
            print('validation batch {}/{}...'.format(i_batch + 1, int(n_val / self.batch_size)))

            end = min(start + self.batch_size, n_val)
            x = x_val[val_inds[start:end]]
            y = y_val[val_inds[start:end]]

            feed_dict = {self.model.input_ph: prepare_batch(x),
                         self.model.label_ph: prepare_batch(y)}
            y_pred, ae_loss = sess.run([self.model.logits, self.model.loss],
                                       feed_dict=feed_dict)

            # exclude examples from training and validation (if any) and run rank metrics
            y_pred[x.nonzero()] = -np.min(y_pred) - 1.0
            ndcg_list.append(ndcg_binary_at_k_batch(y_pred, y, k=100))
            r100_list.append(recall_at_k_batch(y_pred, y, k=100))
            bce_list.append(binary_crossentropy_from_logits(y_pred, y))
            loss_list.append(ae_loss)

        val_ndcg = np.concatenate(ndcg_list).mean()  # mean over n_val
        val_r100 = np.concatenate(r100_list).mean()
        val_bce = np.concatenate(bce_list).mean()
        val_loss = np.mean(loss_list)  # mean over batches

        metrics = {'val_ndcg': val_ndcg, 'val_r100': val_r100, 'val_bce': val_bce,
                   'val_loss': val_loss}
        if logger is not None:
            logger.log_metrics(metrics)

        return metrics


@gin.configurable
class WAE(object):

    def __init__(self, w_inits, b_inits=None,
                 randomize_inits=False, use_biases=True,
                 normalize_inputs=False, shared_weights=False, loss="mse",
                 keep_prob=1.0, lam=0.01, lr=3e-4, random_seed=None):

        self.w_inits = w_inits
        self.b_inits = [None] * len(w_inits) if b_inits is None else b_inits
        self.randomize_inits = randomize_inits
        self.use_biases = use_biases
        self.normalize_inputs = normalize_inputs
        self.shared_weights = shared_weights
        self.lam = lam
        self.lr = lr
        self.random_seed = random_seed

        # loss
        loss_functions = {"mse": mse,
                          "nll": neg_ll,
                          "bce": tf.nn.sigmoid_cross_entropy_with_logits}
        loss_fn = loss_functions[loss]

        # placeholders and weights
        self.input_ph = tf.placeholder(dtype=tf.float32, shape=[None, w_inits[0].shape[1]])
        self.label_ph = tf.placeholder(dtype=tf.float32, shape=[None, w_inits[0].shape[1]])
        self.keep_prob_ph = tf.placeholder_with_default(keep_prob, shape=None)
        self.weights, self.biases = self.construct_weights()

        # build graph
        self.logits = self.forward_pass()
        self.loss = tf.reduce_mean(
            loss_fn(labels=self.label_ph, logits=self.logits)) + self.reg_term()
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.saver = tf.train.Saver()

        # add summary statistics
        tf.summary.scalar('loss', self.loss)
        self.summaries = tf.summary.merge_all()

    def construct_weights(self,):

        weights = []
        biases = []

        # define weights
        for i, (w_init, b_init) in enumerate(zip(self.w_inits, self.b_inits)):
            weight_key = "weight_{}to{}".format(i, i+1)
            bias_key = "bias_{}".format(i+1)

            if i == 0 or not self.shared_weights:
                w = sparse_tensor_from_init(w_init, randomize=self.randomize_inits,
                                            name=weight_key)
            weights.append(w)

            if self.use_biases:
                if b_init is None:
                    b_init = np.zeros([w_init.shape[0]])
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


class MetricLogger(object):

    def __init__(self, base_dir, sess, metric_name='ndcg_at_k_val'):

        self.sess = sess
        self.metric_name = metric_name

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(base_dir, timestamp)

        self.metrics = {}
        self.summaries = {}
        self.summary_writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())

        self.history = defaultdict(list)

    def log_metrics(self, metrics):
        """Log a dictionary of metrics to a tf.summary.FileWriter
        """
        feed_dict = {}
        summaries = []
        for name, value in metrics.items():
            if name not in self.metrics:
                self.metrics[name] = tf.Variable(0.0, name=name)
                self.summaries[name] = tf.summary.scalar(name, self.metrics[name])
            summaries.append(self.summaries[name])
            feed_dict[self.metrics[name]] = value
        summaries = self.sess.run(summaries, feed_dict=feed_dict)
        summaries_dict = dict(zip(metrics.keys(), summaries))
        self.log_summaries(summaries_dict)

    def log_summaries(self, summaries, step=None):

        for name, summary in summaries.items():
            self.history[name].append(summary)
            if step is None:
                step = len(self.history[name])
            self.summary_writer.add_summary(summary, global_step=step)


def sparse_tensor_from_init(init, name='sparse_weight', randomize=False, eps=0.001):

    init = init.tocoo()
    init_data = init.data
    if randomize:
        init_data = eps * np.random.randn(*init.data.shape)

    w_inds = tf.convert_to_tensor(list(zip(init.row, init.col)), dtype=np.int64)
    w_data = tf.Variable(init_data.astype(np.float32), name=name)
    w = tf.SparseTensor(w_inds, tf.identity(w_data),
                        dense_shape=init.shape)
    w = tf.sparse.reorder(w)  # as suggested here:
    # https://www.tensorflow.org/api_docs/python/tf/sparse/SparseTensor?version=stable

    #  summary for tensorboard
    tf.summary.histogram(name, w_data)

    return w


def neg_ll(logits, labels):

    log_softmax_var = tf.nn.log_softmax(logits)
    return -tf.reduce_sum(log_softmax_var * labels, axis=1)


def mse(labels, logits):

    return tf.square(labels, logits)


def prepare_batch(x):
    if issparse(x):
        x = x.toarray()

    return x.astype('float32')
