
import gin
from models.tf import TFRecommender


@gin.configurable
class DistilledRecommender(TFRecommender):

    def __init__(self, log_dir=None, teacher=None, batch_size=100, n_epochs=10):
        """Build a sparse, TF-based wide auto-encoder model with given initial sparse weights,
        and train it to predict the predictions of a *trained* teacher model.
        """
        self.teacher = teacher
        super().__init__(log_dir=log_dir, batch_size=batch_size, n_epochs=n_epochs)

    def prepare_batch(self, x, y=None):
        """Convert a batch of x and y to a sess.run-compatible format,
        but in this case, use teacher.predict(x) instead of y
        """
        y, _ = self.teacher.predict(x)
        x, _ = super().prepare_batch(x, y=None)

        return x, y
