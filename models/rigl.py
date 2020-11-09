"""A self-pruning, multi-layer, fully-connected auto-encoder.

TODO: make the rigl repo into a package, until then:
export PYTHONPATH=$PYTHONPATH:/Users/jan/code/rigl/

Based on RIGL (https://github.com/google-research/rigl)
and in particular the MNIST example, rigl/mnist/mnist_train_eval.py
"""
import gin
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.contrib import layers as contrib_layers
from tensorflow.contrib.model_pruning.python import pruning
from tensorflow.contrib.model_pruning.python.layers import layers
from tensorflow.contrib.training.python.training import hparam

from rigl import sparse_optimizers
from rigl import sparse_utils

from losses import mse, neg_ll


@gin.configurable
class MaskedAutoencoder(object):
    """Autoencoder with support for masked layers and various sparsity-aware optimizers
    """
    def __init__(self, n_items, hidden_dim=200, n_hidden=1, keep_prob=1.0, l2_reg=0.01, loss='nll',
                 training_method='rigl'):
        self.n_items = n_items
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden
        self.l2_reg = l2_reg
        self.random_seed = 88

        loss_functions = {"mse": mse,
                          "nll": neg_ll,
                          "cxe": neg_ll,
                          "bce": tf.nn.sigmoid_cross_entropy_with_logits,
                          "bxe": tf.nn.sigmoid_cross_entropy_with_logits}
        loss_fn = loss_functions[loss]

        # placeholders
        self.input_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.n_items])
        self.label_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.n_items])
        self.keep_prob_ph = tf.placeholder_with_default(keep_prob, shape=None)

        # build graph
        self.logits = self.forward()
        self.loss = tf.reduce_mean(
            loss_fn(labels=self.label_ph, logits=self.logits)) + self.reg_term()

        # get handle for weights and masks
        self.weights = pruning.get_weights()
        self.masks = pruning.get_masks()

        # train, init and save ops
        self.train_op, self.init_ops = sparse_optimizer(training_method, self.loss, self.masks)
        self.saver = tf.train.Saver()

    def forward(self, reuse=False):
        """Construct forward graph (up until logits) and loss"""
        regularizer = contrib_layers.l2_regularizer(scale=self.l2_reg)
        h = tf.nn.dropout(self.input_ph, rate=1-self.keep_prob_ph)
        init = contrib_layers.xavier_initializer(seed=self.random_seed)
        for i in range(self.n_hidden):
            h = layers.masked_fully_connected(
                inputs=h,
                num_outputs=self.hidden_dim,
                activation_fn=tf.nn.tanh,
                weights_regularizer=regularizer,
                weights_initializer=init,
                reuse=False,
                scope=f'layer{i + 1}')
        logits = layers.masked_fully_connected(
            inputs=h,
            num_outputs=self.n_items,
            reuse=False,
            activation_fn=None,
            weights_regularizer=regularizer,
            weights_initializer=init,
            scope='logits')

        return logits

    def reg_term(self):

        return tf.losses.get_regularization_loss()

    def get_density(self, sess):
        """From rigl/mnist_train_eval.py
        """
        masks = sess.run(self.masks)

        # Dead input pixels.
        inds = np.sum(masks[0], axis=1) != 0
        masks[0] = masks[0][inds]
        compressed_masks = []
        for i in range(len(masks)):
            w = masks[i]
            # Find neurons that doesn't have any incoming edges.
            do_w = np.sum(w, axis=0) != 0
            if i < (len(masks) - 1):
                # Find neurons that doesn't have any outgoing edges.
                di_wnext = np.sum(masks[i+1], axis=1) != 0
                # Kept neurons should have at least one incoming and one outgoing edges.
                do_w = np.logical_and(do_w, di_wnext)
            compressed_w = w[:, do_w]
            compressed_masks.append(compressed_w)
            if i < (len(masks) - 1):
                # Remove incoming edges from removed neurons.
                masks[i+1] = masks[i+1][do_w]
        nnzs = [np.sum(m != 0) for m in compressed_masks]
        sizes = [np.size(m) for m in compressed_masks]

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
def sparse_optimizer(training_method, loss, masks,
                     momentum=0.9, use_nestorov=True, lr=3e-4, lr_drop_steps=None,
                     mask_init_method='random', end_sparsity=0.9, prune_step=50000,
                     maskupdate_begin_step=0, maskupdate_end_step=50000, maskupdate_frequency=100,
                     grow_init='zeros', drop_fraction=0.3, drop_fraction_anneal='cosine',
                     rigl_acc_scale=0., s_momentum=0.9):
    """Maybe easier to just gin.register the existing ops directly?
    """
    # standard-ish momentum optimizer
    global_step = tf.train.get_or_create_global_step()
    if lr_drop_steps is not None:
        lr_vals = [lr / (3. ** i) for i in range(len(lr_drop_steps) + 1)]
        lr = tf.train.piecewise_constant(global_step, lr_drop_steps, values=lr_vals)
    opt = tf.train.MomentumOptimizer(lr, momentum, use_nesterov=use_nestorov)

    # modify the optimizer to also update the masks
    pruning_obj = None
    if training_method == 'set':
        opt = sparse_optimizers.SparseSETOptimizer(
            opt, begin_step=maskupdate_begin_step,
            end_step=maskupdate_end_step, grow_init=grow_init,
            frequency=maskupdate_frequency, drop_fraction=drop_fraction,
            drop_fraction_anneal=drop_fraction_anneal)
    elif training_method == 'static':
        opt = sparse_optimizers.SparseStaticOptimizer(
            opt, begin_step=maskupdate_begin_step,
            end_step=maskupdate_end_step, grow_init=grow_init,
            frequency=maskupdate_frequency, drop_fraction=drop_fraction,
            drop_fraction_anneal=drop_fraction_anneal)
    elif training_method == 'momentum':
        opt = sparse_optimizers.SparseMomentumOptimizer(
            opt, begin_step=maskupdate_begin_step,
            end_step=maskupdate_end_step, momentum=s_momentum,
            frequency=maskupdate_frequency, drop_fraction=drop_fraction,
            grow_init=grow_init,
            drop_fraction_anneal=drop_fraction_anneal, use_tpu=False)
    elif training_method == 'rigl':
        opt = sparse_optimizers.SparseRigLOptimizer(
            opt, begin_step=maskupdate_begin_step,
            end_step=maskupdate_end_step, grow_init=grow_init,
            frequency=maskupdate_frequency,
            drop_fraction=drop_fraction,
            drop_fraction_anneal=drop_fraction_anneal,
            initial_acc_scale=rigl_acc_scale, use_tpu=False)
    elif training_method == 'prune':
        # no masks, use pruning object (see tf.contrib.model_pruning.python.pruning)
        pruning_spec = hparam.HParams(
            name='model_pruning',
            begin_pruning_step=0, end_pruning_step=prune_step,
            sparsity_function_begin_step=0, sparsity_function_end_step=prune_step,
            initial_sparsity=0.0, target_sparsity=end_sparsity, pruning_frequency=prune_step,
            weight_sparsity_map=[''], block_dims_map=[''], threshold_decay=0.0, nbins=256,
            block_height=1, block_width=1, block_pooling_function='AVG',
            sparsity_function_exponent=3.0, use_tpu=False)
        pruning_obj = pruning.Pruning(pruning_spec, global_step=global_step)
    elif training_method not in ['baseline', 'lth']:
        raise ValueError('training_method must be one of: set, static, momentum, rigl, lth')

    # train op
    train_op = opt.minimize(loss, global_step=global_step)
    if training_method == 'prune':
        with tf.control_dependencies([train_op]):
            train_op = pruning_obj.conditional_mask_update_op()

    # init ops
    init_ops = [tf.global_variables_initializer()]
    if training_method not in ['baseline', 'prune']:
        mask_init_op = sparse_utils.get_mask_init_fn(
            masks, mask_init_method, end_sparsity, custom_sparsity_map={})
        init_ops.append(mask_init_op)

    return train_op, init_ops
