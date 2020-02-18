
import os
import datetime
from collections import defaultdict

import gin
import numpy as np
import tensorflow as tf
from scipy.sparse import issparse
from tensorflow.contrib.layers import apply_regularization, l2_regularizer

from metric import ndcg_binary_at_k_batch, recall_at_k_batch


@gin.configurable
class WAE(object):

    def __init__(self, w_inits, b_inits=None, use_biases=True,
                 normalize_inputs=False, shared_weights=False, loss="mse",
                 keep_prob=1.0, lam=0.01, lr=3e-4, random_seed=None):

        self.w_inits = w_inits
        self.b_inits = [None] * len(w_inits) if b_inits is None else b_inits
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
        self.keep_prob_ph = tf.placeholder_with_default(keep_prob, shape=None)
        self.weights, self.biases = self.construct_weights()

        # build graph
        self.logits = self.forward_pass()
        self.loss = tf.reduce_mean(
            loss_fn(labels=self.input_ph, logits=self.logits)) + self.reg_term()
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
                w = sparse_tensor_from_init(w_init, name=weight_key)
            weights.append(w)

            if self.use_biases:
                if b_init is None:
                    b_init = b_init or np.zeros([w_init.shape[0]])
                biases.append(tf.Variable(b_init.astype(np.float32), name=bias_key))
                tf.summary.histogram(bias_key, biases[-1])

        return weights, biases

    def forward_pass(self):
        # construct forward graph
        if self.normalize_inputs:
            h = tf.nn.l2_normalize(self.input_ph, 1)
        else:
            h = self.input_ph
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


def neg_ll(logits, labels):

    log_softmax_var = tf.nn.log_softmax(logits)
    return -tf.reduce_sum(log_softmax_var * labels, axis=1)


def mse(labels, logits):

    return tf.square(labels, logits)


def evaluate(model, sess, x_val, y_val, batch_size=100, metric_logger=None):
    """Evaluate model on observed and unobserved validation data x_val, y_val
    """
    n_val = x_val.shape[0]
    val_inds = list(range(n_val))

    loss_list = []
    ndcg_list = []
    r100_list = []
    for i_batch, start in enumerate(range(0, n_val, batch_size)):
        print('validation batch {}/{}...'.format(i_batch + 1, int(n_val / batch_size)))

        end = min(start + batch_size, n_val)
        x = x_val[val_inds[start:end]]
        y = y_val[val_inds[start:end]]

        if issparse(x):
            x = x.toarray()
        x = x.astype('float32')

        y_pred, ae_loss = sess.run([model.logits, model.loss], feed_dict={model.input_ph: x})
        # exclude examples from training and validation (if any)
        y_pred[x.nonzero()] = -np.inf
        ndcg_list.append(ndcg_binary_at_k_batch(y_pred, y, k=100))
        r100_list.append(recall_at_k_batch(y_pred, y, k=100))
        loss_list.append(ae_loss)

    val_ndcg = np.concatenate(ndcg_list).mean()  # mean over n_val
    val_r100 = np.concatenate(r100_list).mean()  # mean over n_val
    val_loss = np.mean(loss_list)  # mean over batches
    metrics = {'val_ndcg': val_ndcg, 'val_r100': val_r100, 'val_loss': val_loss}

    if metric_logger is not None:
        metric_logger.log_metrics(metrics)

    return metrics


def train_one_epoch(model, sess, x_train,
                    x_val=None, y_val=None, batch_size=100,
                    print_interval=1, metric_logger=None):

    n_train = x_train.shape[0]
    train_inds = list(range(n_train))

    np.random.shuffle(train_inds)
    for i_batch, start in enumerate(range(0, n_train, batch_size)):
        if i_batch % print_interval == 0:
            print('batch {}/{}...'.format(i_batch + 1, int(n_train / batch_size)))

        end = min(start + batch_size, n_train)
        x = x_train[train_inds[start:end]]

        if issparse(x):
            x = x.toarray()
        x = x.astype('float32')

        feed_dict = {model.input_ph: x}
        summary_train, _ = sess.run([model.summaries, model.train_op], feed_dict=feed_dict)

        if metric_logger is not None:
            metric_logger.log_summaries({'summary': summary_train})


@gin.configurable
def train(model, x_train, x_val, y_val, batch_size=100, n_epochs=10, log_dir=None):
    """Train a tensorflow recommender

    TODO: model snapshots (check lines containing "best_ndcg" in Liang's notebook)
    """

    with tf.Session() as sess:
        metric_logger = MetricLogger(log_dir, sess) if log_dir is not None else None

        init = tf.global_variables_initializer()
        sess.run(init)

        metrics = evaluate(model, sess, x_val, y_val, metric_logger=metric_logger)
        print('Validation NDCG = {}'.format(metrics))

        for epoch in range(n_epochs):
            print('Training. Epoch = {}/{}'.format(epoch + 1, n_epochs))
            train_one_epoch(model, sess, x_train, batch_size=batch_size,
                            metric_logger=metric_logger)

            metrics = evaluate(model, sess, x_val, y_val, batch_size=batch_size,
                               metric_logger=metric_logger)
            print('Validation NDCG = {}'.format(metrics))

    return metrics


@gin.configurable
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
