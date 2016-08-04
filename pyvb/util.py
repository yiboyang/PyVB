import numpy as np
from scipy.linalg import eigh, det, inv
from scipy.spatial.distance import cdist
from scipy.special import gammaln

from .moments import E_lndetW_Wishart


def logsum(A, axis=None):
    """Computes the sum of A assuming A is in the log domain.

    Returns log(sum(exp(A), axis)) while minimizing the possibility of
    over/underflow.
    """
    Amax = A.max(axis)
    if axis and A.ndim > 1:
        shape = list(A.shape)
        shape[axis] = 1
        Amax.shape = shape
    Asum = np.log(np.sum(np.exp(A - Amax), axis))
    Asum += Amax.reshape(Asum.shape)
    if axis:
        # Look out for underflow.
        Asum[np.isnan(Asum)] = - np.Inf
    return Asum


def normalize(A, axis=None):
    A += np.finfo(float).eps
    Asum = A.sum(axis)
    if axis and A.ndim > 1:
        # Make sure we don't divide by zero.
        Asum[Asum == 0] = 1
        shape = list(A.shape)
        shape[axis] = 1
        Asum.shape = shape
    return A / Asum


# def _sym_quad_form_old(x,mu,A):
#    """
#    calculate x.T * inv(A) * x
#    """
#    A_chol = cholesky(A,lower=True)
#    A_sol = solve(A_chol, (x-mu).T, lower=True).T
#    q = np.sum(A_sol ** 2, axis=1)
#    return q

def _sym_quad_form(x, mu, A):
    """
    calculate x.T * inv(A) * x
    """
    q = (cdist(x, mu[np.newaxis], "mahalanobis", VI=inv(A)) ** 2).reshape(-1)
    return q


def log_like_Gauss(obs, mu, cv):
    """
    Log probability for Gaussian with full covariance matrices.
    lnP = -0.5 * (ln2pi + lndet(cv) + (obs-mu)cv(obs-mu))
    """
    nobs, ndim = obs.shape
    nmix = len(mu)
    lnf = np.empty((nobs, nmix))
    for k in range(nmix):
        dln2pi = ndim * np.log(2.0 * np.pi)
        lndetV = np.log(det(cv[k]))
        q = _sym_quad_form(obs, mu[k], cv[k])
        lnf[:, k] = -0.5 * (dln2pi + lndetV + q)
    return lnf


def log_like_Gauss2(obs, nu, V, beta, m):
    """
    Log probability for Gaussian with full covariance matrices.
    Here mean vectors and covarience matrices are probability variable with
    respect to Gauss-Wishart distribution.
    """
    nobs, ndim = obs.shape
    nmix = len(m)
    lnf = np.empty((nobs, nmix))
    for k in range(nmix):
        dln2pi = ndim * np.log(2.0 * np.pi)
        lndetV = - E_lndetW_Wishart(nu[k], V[k])
        cv = V[k] / nu[k]
        q = _sym_quad_form(obs, m[k], cv) + ndim / beta[k]
        lnf[:, k] = -0.5 * (dln2pi + lndetV + q)

    return lnf


def cnormalize(X):
    """
    Z transformation
    """
    return (X - np.mean(X, 0)) / np.std(X, 0)


def correct_k(k, m):
    """
    Poisson prior for P(Model)
    input
        k [int] : number of clusters
        m [int] : poisson parameter
    output
        log-likelihood
    """
    return k * np.log(m) - m - 2.0 * gammaln(k + 1)


def num_param_Gauss(d):
    """
    count number of parameters for Gaussian d-dimension.
    input
        d [int] : dimension of data
    """
    return 0.5 * d * (d + 3.0)


