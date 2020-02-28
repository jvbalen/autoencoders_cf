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
        """Optionally train recommender"""
        raise NotImplementedError()

    def predict(self, x, y=None):
        """Predict scores.
        If y is not None, also return a loss.
        """
        raise NotImplementedError()

    def evaluate(self, x_val, y_val):
        """Evaluate model on observed and unobserved validation data x_val, y_val
        """
        n_val = x_val.shape[0]
        val_inds = list(range(n_val))

        fin_list = []
        loss_list = []
        ndcg_list = []
        r100_list = []
        bce_list = []
        for i_batch, start in enumerate(range(0, n_val, self.batch_size)):
            print('validation batch {}/{}...'.format(i_batch + 1, int(n_val / self.batch_size)))

            end = min(start + self.batch_size, n_val)
            x = x_val[val_inds[start:end]]
            y = y_val[val_inds[start:end]]

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
                   'isfin': val_fin}
        self.logger.log_metrics(metrics, config=gin.operative_config_str())

        return metrics
