import numpy as np
from scipy.sparse import random

from models.slim import gramm_matrix, closed_form_slim, woodbury_slim


def test_woodbury_slim():

    n_users, n_items = 1000, 100
    np.random.seed(1234)

    x = random(n_users, n_items, density=0.01).tocsr()
    gramm = gramm_matrix(x)

    steck = closed_form_slim(gramm)
    woodbury = woodbury_slim(x, batch_size=100, density=1.0)

    assert np.allclose(steck, woodbury.toarray(), rtol=1e-3)
