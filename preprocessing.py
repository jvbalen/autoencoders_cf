
import gin
from scipy.sparse import issparse
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfTransformer


@gin.configurable
def preprocess(x_train, y_train, x_val, y_val, x_test=None, y_test=None, tfidf=False, norm=None,
               process_labels=False, user_cap=None, item_cap=None):

    if user_cap or item_cap:
        x_train = x_train[:user_cap, :item_cap]
        x_val = x_val[:user_cap, :item_cap]
        y_val = y_val[:user_cap, :item_cap]
        x_test = x_test[:user_cap, :item_cap]
        y_test = y_test[:user_cap, :item_cap]
    x_train, y_train, x_val, y_val, x_test, y_test = \
        map(to_csr_if_sparse, [x_train, y_train, x_val, y_val, x_test, y_test])

    if tfidf:
        trans = TfidfTransformer(norm=norm)
    elif norm:
        trans = Normalizer(norm=norm)
    else:
        return x_train, y_train, x_val, y_val, x_test, y_test

    x_train = trans.fit_transform(x_train)
    x_val = trans.transform(x_val)
    x_test = trans.transform(x_test) if x_test is not None else None
    if process_labels:
        y_train = trans.transform(y_train)
        y_val = trans.transform(y_val)
        y_test = trans.transform(y_test) if y_test is not None else None

    return x_train, y_train, x_val, y_val, x_test, y_test


def to_csr_if_sparse(x):

    return x if x is None or not issparse(x) else x.tocsr()
