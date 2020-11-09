
import numpy as np
from scipy.sparse import csr_matrix

from metric import ndcg_binary_at_k_batch, recall_at_k_batch, mean_item_rank


def test_ndcg():
    """Test if perfect ranking return a score of 1.0,
    opposite ranking returns a score of 0,
    and a few variations
    """
    # perfect ranking, k = 1
    y_true = csr_matrix([[1, 0, 0, 0],
                         [1, 1, 0, 0],
                         [1, 1, 1, 0]])
    y_pred = np.array([[.4, .3, .2, .1],
                       [.4, .3, .2, .1],
                       [.4, .3, .2, .1]])
    expected = np.array([1., 1., 1.])
    result = ndcg_binary_at_k_batch(y_pred, y_true, k=1)
    assert np.all(result == expected)

    # perfect ranking, k = 3
    result = ndcg_binary_at_k_batch(y_pred, y_true, k=3)
    assert np.all(result == expected)

    # terrible ranking, k = 1
    y_pred = np.array([[.1, .2, .3, .4],
                       [.1, .2, .3, .4],
                       [.1, .2, .3, .4]])
    expected = np.array([0, 0, 0])
    result = ndcg_binary_at_k_batch(y_pred, y_true, k=1)
    assert np.all(result == expected)

    # terrible ranking, k = 3
    dcg = np.array([0, 1 / np.log2(4), 1 / np.log2(3) + 1 / np.log2(4)])
    max_dcg = np.array([1, 1 + 1 / np.log2(3), 1 + 1 / np.log2(3) + 1 / np.log2(4)])
    expected = dcg / max_dcg
    result = ndcg_binary_at_k_batch(y_pred, y_true, k=3)
    assert np.all(result == expected)


def test_recall():
    """Test if perfect ranking return a score of 1.0,
    opposite ranking returns a score of 0,
    and a few variations

    NOTE: in our implemenation of recall_at_k_batch,
    the number of relevant items in the recall denominator
    is *capped at k* (as a design decision),
    therefore the first expected result is all ones.
    """
    # perfect ranking, k = 1
    y_true = csr_matrix([[1, 0, 0, 0],
                        [1, 1, 0, 0],
                        [1, 1, 1, 0]])
    y_pred = np.array([[.4, .3, .2, .1],
                       [.4, .3, .2, .1],
                       [.4, .3, .2, .1]])
    expected = np.array([1., 1., 1])
    result = recall_at_k_batch(y_pred, y_true, k=1)
    assert np.all(result == expected)

    # perfect ranking, k = 3
    expected = np.array([1., 1., 1.])
    result = recall_at_k_batch(y_pred, y_true, k=3)
    assert np.all(result == expected)

    # terrible ranking, k = 1
    y_pred = np.array([[.1, .2, .3, .4],
                       [.1, .2, .3, .4],
                       [.1, .2, .3, .4]])
    expected = np.array([0, 0, 0])
    result = recall_at_k_batch(y_pred, y_true, k=1)
    assert np.all(result == expected)

    # terrible ranking, k = 3
    expected = np.array([0, 1./2, 2./3])
    result = recall_at_k_batch(y_pred, y_true, k=3)
    assert np.all(result == expected)


def test_mean_item_rank():

    y_all = csr_matrix([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [1, 1, 0, 0],
                        [1, 0, 1, 0]])
    y_pred = np.array([[.4, .3, .2, .1],
                       [.1, .2, .3, .4]])

    # k = 1
    expected = np.array([0., 3.])
    result = mean_item_rank(y_pred, y_all, k=1)
    assert np.all(result == expected)

    # k = 3
    expected = np.array([1., 2.])
    result = mean_item_rank(y_pred, y_all, k=3)
    assert np.all(result == expected)
