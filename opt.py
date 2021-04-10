"""Least squares CG implementation from https://github.com/benfred/implicit,
reference implementation for testing (changes to) its Cython version in
extensions.pyx
"""
import numpy as np
from scipy.sparse import csr_matrix


def nonzeros(m, row):
    """ returns the non zeroes of a row in csr_matrix """
    for index in range(m.indptr[row], m.indptr[row + 1]):
        yield m.indices[index], m.data[index]


def least_squares_cg(Cui, X, Y, regularization, num_threads=0, cg_steps=3):
    users, factors = X.shape
    YtY = Y.T.dot(Y) + regularization * np.eye(factors, dtype=Y.dtype)

    for u in range(users):
        # start from previous iteration
        x = X[u]

        # calculate residual error r = (YtCuPu - (YtCuY.dot(Xu)
        r = -YtY.dot(x)
        for i, confidence in nonzeros(Cui, u):
            if confidence > 0:
                r += (confidence - (confidence - 1) * Y[i].dot(x)) * Y[i]
            else:
                confidence *= -1
                r += -(confidence - 1) * Y[i].dot(x) * Y[i]

        p = r.copy()
        rsold = r.dot(r)
        if rsold < 1e-20:
            continue

        for it in range(cg_steps):
            # calculate Ap = YtCuYp - without actually calculating YtCuY
            Ap = YtY.dot(p)
            for i, confidence in nonzeros(Cui, u):
                if confidence < 0:
                    confidence *= -1

                Ap += (confidence - 1) * Y[i].dot(p) * Y[i]

            # standard CG update
            alpha = rsold / p.dot(Ap)
            x += alpha * p
            r -= alpha * Ap
            rsnew = r.dot(r)
            if rsnew < 1e-20:
                break
            p = r + (rsnew / rsold) * p
            rsold = rsnew

        X[u] = x


def rank_least_squares_cg(Cui, X, Y, mu, sigma, regularization,
                          min_weight=0, max_weight=60, min_target=0, max_target=10,
                          num_threads=0, cg_steps=3, cauchy=False, hinge_loss=False, seed=42,
                          no_rank=False):
    """This is a reference implementation in python of the cython extension with the same name,
    so that we can test for consistency.
    """
    if not isinstance(Cui, csr_matrix):
        raise ValueError('Cui must be a sparse matrix in CSR format')
    users, factors = X.shape
    n_probe = 1000

    # it is convenient for the absolute scale of returned weights (as measured by their median)
    # to be independent of the data. Therefore we estimate the median 'new' weights first from a
    # few random users (below) and compute a simple correcting factor `weight_scale`
    # that we can use later to scale them to the median of the weights given by Cui
    weight_scales = list()
    np.random.seed(seed)
    for u in np.random.choice(users, n_probe):
        x = X[u]
        for i, confidence in nonzeros(Cui, u):
            pred = Y[i].dot(x)
            if no_rank:
                weight, target = 1.0, 1.0
            else:
                weight, _ = compute_wy(pred, mu[u], sigma[u], cauchy=cauchy, hinge_loss=hinge_loss)
            weight_scales.append((confidence + 1) / weight)
    weight_scale = np.median(weight_scales)

    YtY = Y.T.dot(Y) + regularization * np.eye(factors, dtype=Y.dtype)
    for u in range(users):
        # start from previous iteration
        x = X[u]

        # calculate residual error r = (YtCuPu - (YtCuY.dot(Xu)
        r = -YtY.dot(x)
        for i, confidence in nonzeros(Cui, u):
            pred = Y[i].dot(x)

            # NEW: calculate weight, target from pred, mu, sigma
            if no_rank:
                weight, target = 1.0, 1.0
            else:
                weight, target = compute_wy(pred, mu[u], sigma[u], cauchy=cauchy, hinge_loss=hinge_loss)
            if confidence > 0:
                confidence = min(max_weight - 1, max(min_weight - 1, weight_scale * weight - 1))
                target = min(max_target, max(min_target, target))
                r += (confidence * target - (confidence - 1) * Y[i].dot(x)) * Y[i]
            else:
                print('confidence < 0')
                confidence *= -1
                r += -(confidence - 1) * Y[i].dot(x) * Y[i]

        p = r.copy()
        rsold = r.dot(r)
        if rsold < 1e-20:
            continue

        for _ in range(cg_steps):
            # calculate Ap = YtCuYp - without actually calculating YtCuY
            Ap = YtY.dot(p)
            for i, confidence in nonzeros(Cui, u):
                if confidence < 0:
                    confidence *= -1

                Ap += (confidence - 1) * Y[i].dot(p) * Y[i]

            # standard CG update
            alpha = rsold / p.dot(Ap)
            x += alpha * p
            r -= alpha * Ap
            rsnew = r.dot(r)
            if rsnew < 1e-20:
                break
            p = r + (rsnew / rsold) * p
            rsold = rsnew

        X[u] = x


def compute_wy(pred, mu, sigma, cauchy=False, hinge_loss=False):
    """Given the predicted scores for a user's "positives" (items they interacted with),
    and the mean and standard deviation of the user's scores for a sample of "negatives",
    compute the weights and targets that should be used during the
    next least squares step of an alternating least squares recommender,
    such that its square loss approximates a (hinge or discordance-based) ranking loss.
    """
    z = (pred - mu) / sigma

    # compute PDF, CDF
    if cauchy:
        pdf = 1 / (1 + z ** 2) / np.pi / sigma
        cdf = np.arctan(z) / np.pi + 0.5
    else:
        exp_min_z = np.exp(-z)
        pdf = exp_min_z / (1 + exp_min_z) ** 2 / sigma
        cdf = 1 / (1 + exp_min_z)

    # from PDF, CDF, compute expected hinge / discordance loss
    if hinge_loss:
        exp_min_rev_z = np.exp(-(-pred - mu) / sigma)
        loss = np.log1p(np.exp(-np.abs(exp_min_rev_z))) + np.maximum(exp_min_rev_z, 0)  # ccdf(-pred)
        grad = -cdf
    else:
        loss = 1. - cdf
        grad = -pdf
    weight = (grad ** 2) / (4 * loss)
    target = pred - 2 * loss / grad

    return weight, target
