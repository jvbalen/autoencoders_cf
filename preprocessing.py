
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfTransformer


def preprocess(x_train, y_train, x_val, y_val, tfidf=False, norm=None,
               process_labels=False):

    if tfidf:
        trans = TfidfTransformer(norm=norm)
    elif norm:
        trans = Normalizer(norm=norm)
    else:
        return x_train, y_train, x_val, y_val

    x_train = trans.fit_transform(x_train)
    x_val - trans.transform(x_val)
    if process_labels:
        y_train - trans.transform(y_train)
        y_val - trans.transform(y_val)

    return x_train, y_train, x_val, y_val
