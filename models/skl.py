import gin
import numpy as np
from scipy.sparse import issparse
from scipy.special import expit
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.naive_bayes import BernoulliNB, ComplementNB
from sklearn.svm import LinearSVC

from models.base import BaseRecommender
from util import load_weights

# register some external classes to be able to refer to them via gin
gin.external_configurable(BernoulliNB)
gin.external_configurable(ComplementNB)
gin.external_configurable(LinearSVC)
gin.external_configurable(LogisticRegression)
gin.external_configurable(Ridge)


@gin.configurable
class SKLRecommender(BaseRecommender):

    def __init__(self, log_dir=None, Model=LogisticRegression, ovr=True, batch_size=100):
        """Recommender based on a sklearn classification or regression model.

        If ovr=True, wrap the Model in a OneVsRestClassifier. This is required for most
        sklearn classifiers.
        Added so we can test (a.o.) the ComplementNB classifier as described in:
        https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf
        """
        self.model = OneVsRestClassifier(Model()) if ovr else Model()
        super().__init__(log_dir, batch_size=batch_size)

    def train(self, x_train, y_train, x_val, y_val):
        """Train and evaluate a sklearn model"""
        self.model.fit(x_train, y_train.toarray() > 0.0)
        if self.logger is not None:
            self.logger.log_config(gin.operative_config_str())
            self.logger.save_weights(*coefs_from_model(self.model))
        metrics = self.evaluate(x_val, y_val)

        return metrics

    def predict(self, x, y=None):
        """Predict scores
        TODO: fails on OVR classifiers - they don't expose predict_logits of members?
        """
        if issparse(x):
            x = x.toarray()
        x = x.astype('float32')
        try:
            y_pred = self.model.predict_logits(x)
        except AttributeError:
            try:
                y_pred = self.model.predict_proba(x)
            except AttributeError:
                y_pred = self.model.decision_function(x)

        return y_pred, np.nan


@gin.configurable
class LogisticRegressionFromFile(object):

    def __init__(self, path=None):
        """Sparse pretrained logistic regression (LR).

        Given a dense 2d array of LR coeficients and optional 1d intercepts,
        predict class probabilities.
        """
        coefs, intercepts = load_weights(path)
        if intercepts is None:
            print('LogisticRegressionFromFile: no intercepts loaded...')
            intercepts = np.zeros((1, coefs.shape[1]))
        else:
            intercepts = intercepts.reshape(1, -1)
        self.intercepts = intercepts
        self.coefs = coefs

    def fit(self, *args):
        pass

    def predict_logits(self, x):
        return np.asarray(x @ self.coefs + self.intercepts)

    def predict_proba(self, x):
        return expit(self.predict_logits(x))


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
