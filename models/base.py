from collections import defaultdict

import gin
import numpy as np

from metric import ndcg_binary_at_k_batch, recall_at_k_batch, \
    binary_crossentropy_from_logits, count_finite
from util import Logger


class BaseRecommender(object):

    def __init__(self, log_dir, batch_size=100):
        """Generic recommender."""
        self.logger = Logger(log_dir) if log_dir else None
        self.batch_size = batch_size

    def train(self, x_train, y_train, x_val, y_val):
        """Optionally train, return validation metrics"""
        return self.evaluate(x_val, y_val)

    def predict(self, x, y=None):
        """Predict scores.
        If y is not None, also return a loss.
        """
        raise NotImplementedError()

    def evaluate(self, x_val, y_val, other_metrics=None):
        """Evaluate model on observed and unobserved validation data x_val, y_val
        """
        batch_metrics = defaultdict(list)
        for x, y in self.gen_batches(x_val, y_val):
            y_pred, loss = self.predict(x, y)
            batch_metrics['fin'].extend(count_finite(y_pred))
            batch_metrics['loss'].append(loss)

            # exclude examples from training and validation (if any) and compute rank metrics
            y_pred[x.nonzero()] = np.min(y_pred)
            batch_metrics['ndcg'].extend(ndcg_binary_at_k_batch(y_pred, y, k=100))
            batch_metrics['r100'].extend(recall_at_k_batch(y_pred, y, k=100))
            batch_metrics['bce'].extend(binary_crossentropy_from_logits(y_pred, y))

        metrics = {k: np.mean(v) for k, v in batch_metrics.items()}
        if other_metrics:
            metrics.update(other_metrics)
        self.logger.log_metrics(metrics, config=gin.operative_config_str())

        return metrics

    def gen_batches(self, x, y=None, shuffle=False, print_interval=1):
        """Generate batches from data arrays x and y
        """
        n_examples = x.shape[0]
        n_batches = int(np.ceil(n_examples / self.batch_size))
        inds = list(range(n_examples))
        if shuffle:
            np.random.shuffle(inds)
        for i_batch, start in enumerate(range(0, n_examples, self.batch_size)):
            end = min(start + self.batch_size, n_examples)
            if i_batch % print_interval == 0:
                print('batch {}/{}...'.format(i_batch + 1, n_batches))
            if y is None:
                yield x[inds[start:end]], None
            else:
                yield x[inds[start:end]], y[inds[start:end]]
