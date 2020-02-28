
from warnings import warn

import gin
import numpy as np
from scipy.sparse import issparse

from models.base import BaseRecommender
from util import prune_global, load_weights


@gin.configurable
class SLIMRecommender(BaseRecommender):

    def __init__(self, log_dir, reg=500, density=1.0, batch_size=100):
        """Recommender based on Harald Steck's closed form variant [1]
        of Sparse Linear Methods (SLIM) [2].

        [1] Harald Steck, Embarrassingly shallow auto-encoders. WWW 2019
        https://arxiv.org/pdf/1905.03375.pdf

        [3] Xia Ning and George Karypis, SLIM: Sparse Linear Methods for
        Top-N Recommender Systems. ICDM 2011
        http://glaros.dtc.umn.edu/gkhome/node/774
        """
        self.reg = reg
        self.density = density
        self.weights = None
        super().__init__(log_dir, batch_size=batch_size)

    def train(self, x_train, y_train, x_val, y_val):
        """Train and evaluate"""
        if y_train is not None:
            warn("SLIM is unsupervised, y_train will be ignored")

        print('Computing Gramm matrix...')
        gramm = gramm_matrix(x_train).toarray()
        print('Computing weights...')
        self.weights = closed_form_slim(gramm, l2_reg=self.reg)
        if self.density < 1.0:
            self.weights = prune_global(self.weights, self.density)
        if self.logger is not None:
            self.logger.log_config(gin.operative_config_str())
            self.logger.save_weights(self.weights)

        metrics = self.evaluate(x_val, y_val)

        return metrics

    def predict(self, x, y=None):
        """Predict scores"""
        y_pred = x @ self.weights

        return y_pred, np.nan


def closed_form_slim(gramm, l2_reg=500):

    if issparse(gramm):
        gramm = gramm.toarray()
    diag_indices = np.diag_indices(gramm.shape[0])
    gramm[diag_indices] += l2_reg
    inv_gramm = np.linalg.inv(gramm)
    weights = inv_gramm / (-np.diag(inv_gramm))
    weights[diag_indices] = 0.0

    return weights


def gramm_matrix(x):

    return x.T.dot(x)
