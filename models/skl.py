import gin
import numpy as np
from scipy.sparse import issparse
from scipy.special import expit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.naive_bayes import BernoulliNB, ComplementNB
from sklearn.svm import LinearSVC

from metric import ndcg_binary_at_k_batch, recall_at_k_batch
from util import prune

# register some external classes to be able to refer to them via gin
gin.external_configurable(BernoulliNB)
gin.external_configurable(ComplementNB)
gin.external_configurable(LinearSVC)
gin.external_configurable(LogisticRegression)
gin.external_configurable(Ridge)


@gin.configurable
def build_model(Model=LogisticRegression, ovr=True, tfidf=False, norm=None):
    """Return an initialized sklearn classification or regression model with optional
    TFIDF and normalization preprocessing as described in e.g.:
    https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf

    If ovr=True, wrap the Model in a OneVsRestClassifier. This is required for most
    sklearn classifiers.
    """
    model = OneVsRestClassifier(Model()) if ovr else Model()

    if tfidf:
        Model = Pipeline([('tfidf', TfidfTransformer(norm=norm)),
                          ('mod', model)])
    elif norm:
        Model = Pipeline([('norm', Normalizer(norm=norm)),
                          ('mod', model)])

    return Model


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

    def predict_proba(self, x):
        return expit(x @ self.coefs + self.intercepts)


@gin.configurable
class SparsePretrainedLRFromFile(SparsePretrainedLR):

    def __init__(self, path=None, target_density=None, row_nnz=None):
        """Helper class to make SparsePretrainedLR a bit more gin-friendly"""
        data = np.load(path, allow_pickle=True)
        super().__init__(data['coefs'], data['intercepts'],
                         target_density=target_density, row_nnz=row_nnz)


@gin.configurable
def evaluate(model, x_val, y_val, batch_size=100, metric_logger=None):
    """Evaluate model on observed and unobserved validation data x_val, y_val"""
    n_val = x_val.shape[0]
    val_inds = list(range(n_val))

    loss_list = []
    ndcg_list = []
    r100_list = []
    for i_batch, start in enumerate(range(0, n_val, batch_size)):
        print('validation batch {}/{}...'.format(i_batch + 1, int(n_val / batch_size)))

        end = min(start + batch_size, n_val)
        x = x_val[val_inds[start:end]]
        y = y_val[val_inds[start:end]]

        if issparse(x):
            x = x.toarray()
        x = x.astype('float32')

        try:
            y_pred = model.predict_proba(x)
        except AttributeError:
            try:
                y_pred = model.decision_function(x)
            except AttributeError:
                y_pred = model.predict(x)
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

    if metric_logger is not None:
        metric_logger.log_metrics(metrics)

    return metrics


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
