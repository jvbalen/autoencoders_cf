
import gin
import numpy as np
from scipy.sparse import vstack

from util import Logger, prune
from models.base import BaseRecommender


@gin.register
class PopularityRecommender(BaseRecommender):

    def __init__(self, log_dir, batch_size=100, nnz=100):
        """Popularity recommender."""

        self.logger = Logger(log_dir) if log_dir else None
        self.batch_size = batch_size
        self.popularity = None
        self.nnz = nnz

    def train(self, x_train, y_train, x_val, y_val):
        """Optionally train, return validation metrics"""

        self.prior = np.array(x_train.sum(0))
        self.prior = prune(self.prior, row_nnz=self.nnz)

        return self.evaluate(x_val, y_val)

    def predict(self, x, y=None):
        """Predict scores.
        If y is not None, also return a loss.
        """
        y_pred = vstack([self.prior] * x.shape[0])

        return y_pred, None
