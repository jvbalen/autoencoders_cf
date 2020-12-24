
import cython
import numpy as np
from scipy.sparse import csr_matrix
from libc.math cimport exp, log1p, atan, pi

from cython cimport floating, integral
from cython.parallel import parallel, prange

cimport scipy.linalg.cython_blas as cython_blas

# requires scipy v0.16
cimport scipy.linalg.cython_lapack as cython_lapack
from libc.stdlib cimport free, malloc
from libc.string cimport memcpy, memset


# lapack/blas wrappers for cython fused types
cdef inline void axpy(int * n, floating * da, floating * dx, int * incx, floating * dy,
                      int * incy) nogil:
    if floating is double:
        cython_blas.daxpy(n, da, dx, incx, dy, incy)
    else:
        cython_blas.saxpy(n, da, dx, incx, dy, incy)

cdef inline void symv(char *uplo, int *n, floating *alpha, floating *a, int *lda, floating *x,
                      int *incx, floating *beta, floating *y, int *incy) nogil:
    if floating is double:
        cython_blas.dsymv(uplo, n, alpha, a, lda, x, incx, beta, y, incy)
    else:
        cython_blas.ssymv(uplo, n, alpha, a, lda, x, incx, beta, y, incy)

cdef inline floating dot(int *n, floating *sx, int *incx, floating *sy, int *incy) nogil:
    if floating is double:
        return cython_blas.ddot(n, sx, incx, sy, incy)
    else:
        return cython_blas.sdot(n, sx, incx, sy, incy)

cdef inline void scal(int *n, floating *sa, floating *sx, int *incx) nogil:
    if floating is double:
        cython_blas.dscal(n, sa, sx, incx)
    else:
        cython_blas.sscal(n, sa, sx, incx)


cdef inline (float, float) compute_wy(float pred, float mu, float sigma, int cauchy=0, int hinge_loss=0) nogil:

    cdef float z, pdf, cdf, loss, grad
    z = (pred - mu) / sigma
    if cauchy == 1:
        pdf = 1 / (1 + z ** 2) / pi / sigma
        cdf = atan(z) / pi + 0.5
    elif z < -46:  # exp(-z) > 1e20 -> consider inf
        pdf = 0.
        cdf = 0.
    else:
        exp_min_z = exp(-z)
        pdf = exp_min_z / (1 + exp_min_z) ** 2 / sigma
        cdf = 1 / (1 + exp_min_z)
    if hinge_loss == 1:
        exp_min_rev_z = exp(-(-pred - mu) / sigma)
        loss = log1p(exp_min_rev_z)  # ccdf(-pred)
        grad = -cdf
    else:
        loss = 1. - cdf
        grad = -pdf
    if loss == 0.:
        weight = 0.
    else:
        weight = (grad ** 2) / (4. * loss)
    target = pred - 2. * loss / grad

    return weight, target


def least_squares_cg(Cui, X, Y, regularization, num_threads=0, cg_steps=3):
    return _least_squares_cg(Cui.indptr, Cui.indices, Cui.data.astype('float32'),
                             X, Y, regularization, num_threads, cg_steps)


@cython.cdivision(True)
@cython.boundscheck(False)
def _least_squares_cg(integral[:] indptr, integral[:] indices, float[:] data,
                      floating[:, :] X, floating[:, :] Y, float regularization,
                      int num_threads=0, int cg_steps=3):
    dtype = np.float64 if floating is double else np.float32

    cdef integral users = X.shape[0], u, i, index
    cdef int one = 1, it, N = X.shape[1]
    cdef floating confidence, temp, alpha, rsnew, rsold
    cdef floating zero = 0.

    cdef floating[:, :] YtY = np.dot(np.transpose(Y), Y) + regularization * np.eye(N, dtype=dtype)

    cdef floating * x
    cdef floating * p
    cdef floating * r
    cdef floating * Ap

    with nogil, parallel(num_threads=num_threads):
        # allocate temp memory for each thread
        Ap = <floating *> malloc(sizeof(floating) * N)
        p = <floating *> malloc(sizeof(floating) * N)
        r = <floating *> malloc(sizeof(floating) * N)
        try:
            for u in prange(users, schedule='guided'):
                # start from previous iteration
                x = &X[u, 0]

                # if we have no items for this user, skip and set to zero
                if indptr[u] == indptr[u+1]:
                    memset(x, 0, sizeof(floating) * N)
                    continue

                # calculate residual r = (YtCuPu - (YtCuY.dot(Xu)
                temp = -1.0
                symv("U", &N, &temp, &YtY[0, 0], &N, x, &one, &zero, r, &one)

                for index in range(indptr[u], indptr[u + 1]):
                    i = indices[index]
                    confidence = data[index]

                    if confidence > 0:
                        temp = confidence
                    else:
                        temp = 0
                        confidence = -1 * confidence

                    temp = temp - (confidence - 1) * dot(&N, &Y[i, 0], &one, x, &one)
                    axpy(&N, &temp, &Y[i, 0], &one, r, &one)

                memcpy(p, r, sizeof(floating) * N)
                rsold = dot(&N, r, &one, r, &one)

                if rsold < 1e-20:
                    continue

                for it in range(cg_steps):
                    # calculate Ap = YtCuYp - without actually calculating YtCuY
                    temp = 1.0
                    symv("U", &N, &temp, &YtY[0, 0], &N, p, &one, &zero, Ap, &one)

                    for index in range(indptr[u], indptr[u + 1]):
                        i = indices[index]
                        confidence = data[index]

                        if confidence < 0:
                            confidence = -1 * confidence

                        temp = (confidence - 1) * dot(&N, &Y[i, 0], &one, p, &one)
                        axpy(&N, &temp, &Y[i, 0], &one, Ap, &one)

                    # alpha = rsold / p.dot(Ap);
                    alpha = rsold / dot(&N, p, &one, Ap, &one)

                    # x += alpha * p
                    axpy(&N, &alpha, p, &one, x, &one)

                    # r -= alpha * Ap
                    temp = alpha * -1
                    axpy(&N, &temp, Ap, &one, r, &one)

                    rsnew = dot(&N, r, &one, r, &one)
                    if rsnew < 1e-20:
                        break

                    # p = r + (rsnew/rsold) * p
                    temp = rsnew / rsold
                    scal(&N, &temp, p, &one)
                    temp = 1.0
                    axpy(&N, &temp, r, &one, p, &one)

                    rsold = rsnew
        finally:
            free(p)
            free(r)
            free(Ap)

def rank_least_squares_cg(Cui, X, Y, mu, sigma, regularization,
                          min_weight=0, max_weight=60, min_target=0, max_target=10,
                          num_threads=0, cg_steps=3, cauchy=False, hinge_loss=False, seed=42):
    if not isinstance(Cui, csr_matrix):
        raise ValueError('Cui must be a sparse matrix in CSR format')
    return _rank_least_squares_cg(Cui.indptr, Cui.indices, Cui.data.astype('float32'),
                                  X, Y, mu.astype('float32'), sigma.astype('float32'), regularization,
                                  min_weight, max_weight, min_target, max_target,
                                  num_threads, cg_steps, int(cauchy), int(hinge_loss), seed)

@cython.cdivision(True)
@cython.boundscheck(False)
def _rank_least_squares_cg(integral[:] indptr, integral[:] indices, float[:] data,
                           floating[:, :] X, floating[:, :] Y, float[:] mu, float[:] sigma, float regularization,
                           float min_weight=0, float max_weight=60, float min_target=0, float max_target=10,
                           int num_threads=0, int cg_steps=3, int cauchy=0, int hinge_loss=0, int seed=42):
    dtype = np.float64 if floating is double else np.float32

    cdef integral users = X.shape[0], items = Y.shape[1], u, i, index
    cdef int one = 1, it, N = X.shape[1], n_probe = 1000
    cdef floating confidence, temp, alpha, rsnew, rsold, pred, weight, target, weight_scale
    cdef floating zero = 0.

    cdef floating[:, :] YtY = np.dot(np.transpose(Y), Y) + regularization * np.eye(N, dtype=dtype)

    cdef floating * x
    cdef floating * p
    cdef floating * r
    cdef floating * Ap

    weight_scales = list()
    np.random.seed(seed)
    for u in np.random.choice(users, n_probe):
        x = &X[u, 0]
        weights = list()
        for index in range(indptr[u], indptr[u + 1]):
            i = indices[index]
            confidence = data[index]
            pred = dot(&N, &Y[i, 0], &one, x, &one)
            weight, _ = compute_wy(pred, mu[u], sigma[u], cauchy=cauchy, hinge_loss=hinge_loss)
            weight_scales.append((confidence + 1) / weight)
    weight_scale = np.median(weight_scales)
    print(weight_scale)

    with nogil, parallel(num_threads=num_threads):
        # allocate temp memory for each thread
        Ap = <floating *> malloc(sizeof(floating) * N)
        p = <floating *> malloc(sizeof(floating) * N)
        r = <floating *> malloc(sizeof(floating) * N)
        try:
            for u in prange(users, schedule='guided'):
                # start from previous iteration
                x = &X[u, 0]

                # if we have no items for this user, skip and set to zero
                if indptr[u] == indptr[u+1]:
                    memset(x, 0, sizeof(floating) * N)
                    continue

                # calculate residual r = (YtCuPu - (YtCuY.dot(Xu)
                temp = -1.0
                symv("U", &N, &temp, &YtY[0, 0], &N, x, &one, &zero, r, &one)

                for index in range(indptr[u], indptr[u + 1]):
                    i = indices[index]
                    confidence = data[index]

                    # NEW: calculate weight, target from pred, mu, sigma
                    pred = dot(&N, &Y[i, 0], &one, x, &one)
                    weight, target = compute_wy(pred, mu[u], sigma[u], cauchy=cauchy, hinge_loss=hinge_loss)
                    if confidence > 0:
                        confidence = min(max_weight - 1, max(min_weight - 1, weight_scale * weight - 1))
                        target = min(max_target, max(min_target, target))
                        temp = confidence
                    else:
                        confidence = -1 * confidence
                        temp = 0

                    temp = temp * target - (confidence - 1) * dot(&N, &Y[i, 0], &one, x, &one)
                    axpy(&N, &temp, &Y[i, 0], &one, r, &one)

                memcpy(p, r, sizeof(floating) * N)
                rsold = dot(&N, r, &one, r, &one)

                if rsold < 1e-20:
                    continue

                for it in range(cg_steps):
                    # calculate Ap = YtCuYp - without actually calculating YtCuY
                    temp = 1.0
                    symv("U", &N, &temp, &YtY[0, 0], &N, p, &one, &zero, Ap, &one)

                    for index in range(indptr[u], indptr[u + 1]):
                        i = indices[index]
                        confidence = data[index]

                        if confidence < 0:
                            confidence = -1 * confidence

                        temp = (confidence - 1) * dot(&N, &Y[i, 0], &one, p, &one)
                        axpy(&N, &temp, &Y[i, 0], &one, Ap, &one)

                    # alpha = rsold / p.dot(Ap);
                    alpha = rsold / dot(&N, p, &one, Ap, &one)

                    # x += alpha * p
                    axpy(&N, &alpha, p, &one, x, &one)

                    # r -= alpha * Ap
                    temp = alpha * -1
                    axpy(&N, &temp, Ap, &one, r, &one)

                    rsnew = dot(&N, r, &one, r, &one)
                    if rsnew < 1e-20:
                        break

                    # p = r + (rsnew/rsold) * p
                    temp = rsnew / rsold
                    scal(&N, &temp, p, &one)
                    temp = 1.0
                    axpy(&N, &temp, r, &one, p, &one)

                    rsold = rsnew

        finally:
            free(p)
            free(r)
            free(Ap)