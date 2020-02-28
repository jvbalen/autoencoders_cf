import numpy as np
from scipy.sparse import coo_matrix
from util import prune, prune_global, prune_rows


def test_prune():

    x = np.array([[1, 2, 3, 4],
                  [10, 20, 30, 40],
                  [4, 3, 2, 1]])
    exp = np.array([[0, 0, 0, 4],
                    [10, 20, 30, 40],
                    [4, 0, 0, 0]])
    res = prune_global(x, target_density=0.5)
    assert np.all(res.toarray() == exp)

    exp = np.array([[0, 0, 3, 4],
                    [0, 0, 30, 40],
                    [4, 3, 0, 0]])
    res = prune_rows(x, target_nnz=2)
    assert np.all(res.toarray() == exp)

    exp = np.array([[0, 0, 3, 4],
                    [10, 20, 30, 40],
                    [4, 3, 0, 0]])
    res = prune(x, target_density=0.5, row_nnz=2)
    assert np.all(res.toarray() == exp)


def test_prune_sparse():

    x = coo_matrix([[0, 0, 3, 4],
                    [0, 20, 30, 40],
                    [4, 3, 0, 0]])
    exp = np.array([[0, 0, 0, 0],
                    [0, 20, 30, 40],
                    [0, 0, 0, 0]])
    res = prune_global(x, target_density=0.25)
    assert np.all(res.toarray() == exp)

    exp = np.array([[0, 0, 0, 4],
                    [0, 0, 0, 40],
                    [4, 0, 0, 0]])
    res = prune_rows(x, target_nnz=1)
    assert np.all(res.toarray() == exp)

    exp = np.array([[0, 0, 0, 4],
                    [0, 20, 30, 40],
                    [4, 0, 0, 0]])
    res = prune(x, target_density=0.25, row_nnz=1)
    assert np.all(res.toarray() == exp)


def test_prune_very_sparse():
    """Don't crash when matrix is already sparse
    """
    x = coo_matrix([[0, 0, 0, 4],
                    [0, 0, 0, 40],
                    [0, 0, 0, 0]])
    exp = np.array([[0, 0, 0, 4],
                    [0, 0, 0, 40],
                    [0, 0, 0, 0]])
    res = prune(x, target_density=0.25, row_nnz=1)
    assert np.all(res.toarray() == exp)


def test_prune_none():

    x = np.array([[1, 2, 3, 4],
                  [10, 20, 30, 40],
                  [4, 3, 2, 1]])
    exp = np.array([[0, 0, 0, 4],
                    [10, 20, 30, 40],
                    [4, 0, 0, 0]])
    res = prune(x, target_density=0.5, row_nnz=None)
    assert np.all(res.toarray() == exp)

    exp = np.array([[0, 0, 3, 4],
                    [0, 0, 30, 40],
                    [4, 3, 0, 0]])
    res = prune(x, target_density=None, row_nnz=2)
    assert np.all(res.toarray() == exp)


if __name__ == '__main__':

    test_prune()
    test_prune_sparse()
    test_prune_very_sparse()
    test_prune_none()
