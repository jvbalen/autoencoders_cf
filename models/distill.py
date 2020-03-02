
import gin
from scipy.special import expit

from models.tf import TFRecommender
from util import load_weights


@gin.configurable
class DistilledRecommender(TFRecommender):

    def __init__(self, log_dir=None,
                 dense_weights_path=None, sparse_weights_path=None,
                 n_layers=1, batch_size=100, n_epochs=10):
        """Build a sparse, TF-based wide auto-encoder model with given initial sparse weights,
        and train it to predict the predictions of a linear model with dense weights.
        """
        self.teacher_weights, self.teacher_biases = load_weights(dense_weights_path)
        super().__init__(log_dir=log_dir, weights_path=sparse_weights_path,
                         n_layers=n_layers, batch_size=batch_size, n_epochs=10)

    def prepare_batch(self, x, y=None):
        """Ignore y and return new targets y = x @ weights + biases"""

        x, _ = super().prepare_batch(x, y=None)
        y = x @ self.teacher_weights
        if self.teacher_biases is not None:
            y = y + self.teacher_biases
        if self.model.loss in ['bce', 'nll']:
            y = expit(y)

        return x, y
