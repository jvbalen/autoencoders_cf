import numpy as np
from scipy.sparse import issparse


def ndcg_binary_at_k_batch(x_pred, x_true, k=100):
    '''
    Normalized Discounted Cumulative Gain@k for binary relevance
    ASSUMPTIONS: all the 0's in x_true indicate 0 relevance
    '''
    x_pred = x_pred.toarray() if issparse(x_pred) else np.array(x_pred)

    batch_users = x_pred.shape[0]
    idx_topk_part = np.argpartition(-x_pred, k, axis=1)
    topk_part = x_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)

    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]

    tp = 1. / np.log2(np.arange(2, k + 2))

    dcg = (x_true[np.arange(batch_users)[:, np.newaxis],
                  idx_topk].toarray() * tp).sum(axis=1)
    idcg = np.array([(tp[:min(n, k)]).sum()
                     for n in x_true.getnnz(axis=1)])
    return dcg / idcg


def recall_at_k_batch(x_pred, x_true, k=100):
    x_pred = x_pred.toarray() if issparse(x_pred) else np.array(x_pred)

    batch_users = x_pred.shape[0]
    idx = np.argpartition(-x_pred, k, axis=1)
    x_pred_binary = np.zeros_like(x_pred, dtype=bool)
    x_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    x_true_binary = (x_true > 0).toarray()
    tmp = (np.logical_and(x_true_binary, x_pred_binary).sum(axis=1)).astype(
        np.float32)
    recall = tmp / np.minimum(k, x_true_binary.sum(axis=1))
    return recall


def binary_crossentropy_from_logits(x_pred, x_true):
    x_pred = x_pred.toarray() if issparse(x_pred) else np.array(x_pred)
    x_true = x_true.toarray() if issparse(x_true) else np.array(x_true)

    # bce = x_pred - x_pred * x_true + np.log(1 + np.exp(-x_pred))
    bce = np.maximum(x_pred, 0) - x_pred * x_true + np.log(1 + np.exp(-np.abs(x_pred)))   # more stable

    return bce


def count_finite(x_pred, x_true=None):
    # TODO: be more clever about sparse x_pred
    x_pred = x_pred.toarray() if issparse(x_pred) else np.array(x_pred)

    return np.mean(np.isfinite(x_pred), axis=1)


def count_nonzero(x_pred, x_true=None):
    # TODO: be more clever about sparse x_pred
    x_pred = x_pred.toarray() if issparse(x_pred) else np.array(x_pred)

    return np.mean(x_pred != 0, axis=1)


def mean_item_rank(y_pred, y_all, k=100):

    item_counts = np.array(y_all.sum(axis=0)).flatten()
    item_ranks = np.argsort(-item_counts)

    y_pred = y_pred.toarray() if issparse(y_pred) else np.array(y_pred)
    y_pred_topk = np.argpartition(-y_pred, k, axis=1)[:, :k]

    return np.median(item_ranks[y_pred_topk], axis=1)
