import gin
import numpy as np
from scipy.sparse import issparse
from scipy.special import expit
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.naive_bayes import BernoulliNB, ComplementNB
from sklearn.svm import LinearSVC

from metric import ndcg_binary_at_k_batch, recall_at_k_batch
from util import prune, load_weights, Logger

# register some external classes to be able to refer to them via gin
gin.external_configurable(BernoulliNB)
gin.external_configurable(ComplementNB)
gin.external_configurable(LinearSVC)
gin.external_configurable(LogisticRegression)
gin.external_configurable(Ridge)


@gin.configurable
class SKLRecommender(object):

    def __init__(self, log_dir, Model=LogisticRegression, ovr=True, batch_size=100):
        """Recommender based on a sklearn classification or regression model.

        If ovr=True, wrap the Model in a OneVsRestClassifier. This is required for most
        sklearn classifiers.
        Added so we can test (a.o.) the ComplementNB classifier as described in:
        https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf
        """
        self.model = OneVsRestClassifier(Model()) if ovr else Model()
        self.logger = Logger(log_dir) if log_dir else None
        self.batch_size = batch_size

    def train(self, model, x_train, y_train, x_val, y_val):
        """Train and evaluate a sklearn model."""
        model.fit(x_train, y_train.toarray() > 0.0)
        if self.logger is not None:
            self.logger.log_config(gin.operative_config_str())
            self.logger.log_coefs(*coefs_from_model(model))

        metrics = self.evaluate(x_val, y_val)
        if self.logger is not None:
            self.logger.log_results(metrics, config=gin.operative_config_str())

        return metrics

    def evaluate(self, x_val, y_val):
        """Evaluate model on observed and unobserved validation data x_val, y_val"""
        n_val = x_val.shape[0]
        val_inds = list(range(n_val))

        loss_list = []
        ndcg_list = []
        r100_list = []
        for i_batch, start in enumerate(range(0, n_val, self.batch_size)):
            print('validation batch {}/{}...'.format(i_batch + 1, int(n_val / self.batch_size)))

            end = min(start + self.batch_size, n_val)
            x = x_val[val_inds[start:end]]
            y = y_val[val_inds[start:end]]

            if issparse(x):
                x = x.toarray()
            x = x.astype('float32')

            try:
                y_pred = self.model.predict_proba(x)
            except AttributeError:
                try:
                    y_pred = self.model.decision_function(x)
                except AttributeError:
                    y_pred = self.model.predict(x)
            ae_loss = np.sum(np.array(x - y_pred)**2) / x.shape[0]
            # exclude examples from training and validation (if any)
            y_pred[x.nonzero()] = -np.inf
            ndcg_list.append(ndcg_binary_at_k_batch(y_pred, y, k=100))
            r100_list.append(recall_at_k_batch(y_pred, y, k=100))
            loss_list.append(ae_loss)

        val_ndcg = np.concatenate(ndcg_list).mean()  # mean over n_val
        val_r100 = np.concatenate(r100_list).mean()  # mean over n_val
        val_loss = np.mean(loss_list)  # mean over batches
        metrics = {'val_ndcg': val_ndcg, 'val_r100': val_r100, 'val_loss': val_loss}

        if self.logger is not None:
            self.logger.log_metrics(metrics)

        return metrics


@gin.configurable
class SparsePretrainedLR(object):

    def __init__(self, coefs, intercepts=None, target_density=None, row_nnz=None):
        """Sparse pretrained logistic regression (LR).

        Given a dense 2d array of LR coeficients, optional 1d intercepts and
        pruning parameters, predict class probabilities.
        """
        if intercepts is None:
            intercepts = np.zeros((1, coefs.shape[1]))
        else:
            intercepts = intercepts.reshape(1, -1)
        if target_density is not None or row_nnz is not None:
            coefs = prune(coefs, target_density=target_density, row_nnz=row_nnz)
        self.coefs = coefs
        self.intercepts = intercepts

    def fit(self, *args):
        pass

    def predict_logits(self, x):
        return np.asarray(x @ self.coefs + self.intercepts)

    def predict_proba(self, x):
        return expit(self.predict_logits(x))


@gin.configurable
class SparsePretrainedLRFromFile(SparsePretrainedLR):

    def __init__(self, path=None, target_density=None, row_nnz=None):
        """Helper class to make SparsePretrainedLR a bit more gin-friendly"""
        coefs, intercepts = load_weights(path)
        super().__init__(coefs, intercepts,
                         target_density=target_density, row_nnz=row_nnz)


def coefs_from_model(model):
    # if model is a pipeline, get final estimator
    try:
        model = model._final_estimator
    except AttributeError:
        pass
    # if model has no coefs, assume it's a multiclass wrapper
    try:
        return model.coef_, model._intercept
    except AttributeError:
        return coefs_from_multiclass_wrapper(model)


def coefs_from_multiclass_wrapper(model, n_features=None):
    try:
        n_classes = len(model.estimators_)
    except AttributeError:
        return None, None
    if n_features is None:
        _, n_features = model.estimators_[0].coef_.shape

    coefs = np.zeros((n_features, n_classes))
    intercepts = np.zeros((n_classes,))
    for i, estimator in enumerate(model.estimators_):
        try:
            coefs[:, i] = estimator.coef_.flatten()
            intercepts[i] = estimator.intercept_.flatten()
        except AttributeError:
            pass

    return coefs, intercepts
