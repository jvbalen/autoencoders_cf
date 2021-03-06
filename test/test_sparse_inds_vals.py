
import numpy as np
from scipy.sparse import csr_matrix
from models.pairwise import sparse_inds_vals


def test_inds_vals():

    a = [
        [5, 0, 4, 0, 3, 0, 2, 0, 1, 0, 0, 0],
        [0, 5, 0, 4, 0, 3, 0, 2, 0, 1, 0, 0],
        [0, 5, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0],
    ]
    inds_expected = [
        [0, 2, 4],
        [1, 3, 5],
        [1, 3, 0],
        [7, 9, 0],
    ]
    vals_expected = [
        [5, 4, 3],
        [5, 4, 3],
        [5, 4, 0],
        [2, 1, 0],
    ]
    a_sp = csr_matrix(np.array(a))
    inds, vals = sparse_inds_vals(a_sp, row_nnz=3)

    assert np.allclose(inds_expected, inds)
    assert np.allclose(vals_expected, vals)
