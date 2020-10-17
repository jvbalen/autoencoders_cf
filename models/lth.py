
import os
import numpy as np
import tensorflow as tf
import gin

from models.tf import TFRecommender, TFLogger
from models.rigl import MaskedAutoencoder
from util import get_pruning_threshold


@gin.configurable
class LTHRecommender(TFRecommender):

    def __init__(self, log_dir=None, batch_size=100, n_epochs=10, n_finetune_epochs=10,
                 target_density=0.1, reset_weights=True, load_dir=None):
        """Lottery Ticket Hypothesis recommender

        A TFRecommender that can be trained using "Lottery Ticket" training:
        train to convergence, prune small-magnitude weights, reset all other weights
        to their initial values, and retrain while keeping pruned weights zero.
        """
        self.n_finetune_epochs = n_finetune_epochs
        self.target_density = target_density
        self.reset_weights = reset_weights
        self.load_dir = load_dir
        super().__init__(log_dir, Model=MaskedAutoencoder, batch_size=batch_size, exact_batches=True, n_epochs=n_epochs)

    def train(self, x_train, y_train, x_val, y_val):
        """train LTHRecommender: train, prune, reset to inits, and train more"""
        tf.compat.v1.reset_default_graph()
        self.model = self.Model(n_items=x_train.shape[1], training_method='baseline')
        with tf.compat.v1.Session() as self.sess:
            self.logger = TFLogger(self.log_dir, self.sess)
            self.logger.log_config(gin.operative_config_str())

            # set, save and evaluate inits
            self.sess.run(self.model.init_ops)
            init_dir = os.path.join(self.logger.log_dir, 'init')
            self.model.save(self.sess, log_dir=init_dir)

            # fit, prune and fine-tune
            if self.n_epochs:
                self._fit(x_train, y_train, x_val, y_val, n_epochs=self.n_epochs)
            self.restore_and_prune()
            metrics = self._fit(x_train, y_train, x_val, y_val, n_epochs=self.n_finetune_epochs)
        self.sess = None

        return metrics

    def _fit(self, x_train, y_train, x_val, y_val, n_epochs):
        """Fit some number of epochs"""
        best_ndcg = 0.0
        metrics = self.evaluate(x_val, y_val)
        for epoch in range(n_epochs):
            print(f'Training. Epoch = {epoch + 1}/{n_epochs}')
            self.train_one_epoch(x_train, y_train)
            print('Evaluating...')
            metrics = self.evaluate(x_val, y_val, other_metrics={'epoch': epoch + 1})
            if metrics['ndcg'] > best_ndcg:
                best_ndcg = metrics['ndcg']
                self.model.save(self.sess, log_dir=self.logger.log_dir)

        return metrics

    def restore_and_prune(self):
        """set masks"""
        # restore best weights and compute masks
        load_dir = self.logger.log_dir if self.load_dir is None else self.load_dir
        self.model.restore(self.sess, load_dir)
        weights = self.sess.run(self.model.weights)
        masks = [np.abs(w) >= get_pruning_threshold(w, self.target_density) for w in weights]
        print({v: m.mean() for v, m in zip(self.model.masks, masks)})

        # set masks
        if self.reset_weights:
            init_dir = os.path.join(load_dir, 'init')
            self.model.restore(self.sess, init_dir)
        mask_ops = [tf.compat.v1.assign(ref, val) for ref, val in zip(self.model.masks, masks)]
        self.sess.run(mask_ops)
