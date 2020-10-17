
import tensorflow.compat.v1 as tf


def neg_ll(logits, labels):

    log_softmax_var = tf.nn.log_softmax(logits)
    return -tf.reduce_sum(log_softmax_var * labels, axis=1)


def pairwise_loss(logits, labels, n_pairs=100, logistic=True, margin=1.0,
                  clip_labels=False, binarize_labels=False):
    """Pairwise loss from logits
    TODO: fix bug where we get a NaN (very rare)
    """
    if clip_labels:
        labels = tf.minimum(tf.maximum(0., labels), 1.0)
    if binarize_labels:
        labels = tf.cast(tf.greater(labels, 0.5), 'float32')
    pos_sample = tf.random.categorical(tf.math.log(labels), n_pairs)
    neg_sample = tf.random.categorical(tf.math.log(1. - labels), n_pairs)
    y_pos = tf.gather(logits, pos_sample, batch_dims=-1)
    y_neg = tf.gather(logits, neg_sample, batch_dims=-1)
    if logistic:  # logistic pairwise loss ~ soft relu of y_neg - y_pos
        pairwise_losses = tf.math.log(1 + tf.math.exp(y_neg - y_pos))
    else:  # hinge loss ~ relu of y_neg - y_pos
        pairwise_losses = tf.nn.relu(margin + y_neg - y_pos)

    return tf.reduce_sum(pairwise_losses, axis=1)


def mse(labels, logits):
    """per-sample mean square error
    since we want the mean: sum over
    """
    return tf.square(labels - logits) / 2.0
