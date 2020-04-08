from scipy.sparse import csr_matrix

from models.slim import get_neighbors


def test_get_two_neighbors():

    a = [[3, 2, 0, 0],  # n: [0, 1]
         [2, 3, 0, 0],  # n: [0, 1]
         [3, 3, 1, 0]]  # n: [0, 1]
    len_a = len(a)
    a = csr_matrix(a)
    for i in range(len_a):
        ans = [0, 1]
        res = get_neighbors(a, i, max_neighbors=2)
        assert set(res) == set(ans)


def test_get_neighbors_empty():

    a = [[0, 0, 0, 0],
         [1, 0, 0, 0],
         [0, 1, 0, 0],
         [1, 1, 0, 0]]
    b = [[],
         [0],
         [1],
         [0, 1]]
    a = csr_matrix(a)
    for i, ans in enumerate(b):
        res = get_neighbors(a, i, max_neighbors=2)
        assert set(res) == set(ans)

    # asking for more neighbors shouldn't return more if they're all zeros
    for i, ans in enumerate(b):
        res = get_neighbors(a, i, max_neighbors=3)
        assert set(res) == set(ans)
