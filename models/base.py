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

    def evaluate(self, x_val, y_val):
        """Evaluate model on observed and unobserved validation data x_val, y_val
        """
        fin_list = []
        loss_list = []
        ndcg_list = []
        r100_list = []
        bce_list = []
        for x, y in self.gen_batches(x_val, y_val):
            print(f'Evaluating...')
            y_pred, loss = self.predict(x, y)
            fin_list.append(count_finite(y_pred))

            # exclude examples from training and validation (if any) and compute rank metrics
            y_pred[x.nonzero()] = np.min(y_pred)
            ndcg_list.append(ndcg_binary_at_k_batch(y_pred, y, k=100))
            r100_list.append(recall_at_k_batch(y_pred, y, k=100))
            bce_list.append(binary_crossentropy_from_logits(y_pred, y))
            loss_list.append(loss)

        val_fin = np.concatenate(fin_list).mean()  # mean over n_val
        val_ndcg = np.concatenate(ndcg_list).mean()
        val_r100 = np.concatenate(r100_list).mean()
        val_bce = np.concatenate(bce_list).mean()
        val_loss = np.mean(loss_list)  # mean over batches

        metrics = {'val_ndcg': val_ndcg, 'val_r100': val_r100,
                   'val_bce': val_bce, 'val_loss': val_loss,
                   'val_fin': val_fin}
        self.logger.log_metrics(metrics, config=gin.operative_config_str())

        return metrics

    def gen_batches(self, x, y, shuffle=False, print_interval=1):
        """Generate batches from data arrays x and y
        """
        n_examples = x.shape[0]
        inds = list(range(n_examples))
        if shuffle:
            np.random.shuffle(inds)
        for i_batch, start in enumerate(range(0, n_examples, self.batch_size)):
            end = min(start + self.batch_size, n_examples)
            if i_batch % print_interval == 0:
                print('batch {}/{}...'.format(i_batch + 1, int(n_examples / self.batch_size)))

            yield x[inds[start:end]], y[inds[start:end]]
