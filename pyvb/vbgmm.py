﻿from scipy.cluster import vq

from .emgmm import EMGMM
from .moments import *
from .util import log_like_Gauss2


class VBGMM(EMGMM):
    """
    Gaussian Mixture Model with Variational Bayesian (VB) Learning.
    This class is mostly based on the EMGMM class.

    Attributes
      _K [int] : number of hidden states, K
      _u0 [ndarray, shape (K) : prior parameter for mixing coefficients
      _nu0 [float] : dof prior parameter for precision matrix
      _V0 [ndarray, shape (dim,dim)] : scale matrix prior parameter
        for precision matrix
      _beta0 [float] : dof prior parameter for mean vector
      _m0 [ndarray, shape (dim)] : prior for mean vector
      _u [ndarray, shape (K) : posterior parameters for mixing coefficients
      _nu [ndarray, shape (K)] : dof posterior parameter for precision matrix
      _V [ndarray, shape (K,dim,dim)] : scale matrix posterior parameter
        for precision matrices
      _beta [ndarray, shape (K)] : dof posterior parameter for mean vector
      _m [ndarray, shape (K,dim)] : posterior parameter for mean vector
      pi [ndarray, shape (_K)] : expected mixing coefficients
      mu [ndarray, shape (_K, dim)] : expected mean vectors
      cv [ndarray, shape (_K, dim, dim)] : expected covariance matrix
    Methods
      getExpectations : calculate expectation of parameters i.e. pi,mu and cv
      showModel : show model parameters
      score : score the model with respect to variational free energy

      # inheritted from EMGMM
      eval_hidden_states : get the propability of hidden states
      fit : fit model parameters
      decode : return most probable hidden states
      plot1d : plot most probable hidden states along one axis of data
      plot2d : plot most probable hidden states along two axes of data
      makeTransMat : make transition matrix by regarding the data as time series
    """

    def __init__(self, K=10, u0=0.5, m0=0.0, beta0=1, nu0=1, V0=10.0):
        # maximum number of the hidden clusters
        self.K = K
        # hyper parameter for Dirichlet prior mixing coefficients
        self._u0 = np.ones(K) * u0
        # hyper parameters for prior precision matrix
        self._nu0 = nu0
        self._V0 = V0
        # hyperparameters for prior mean vector
        self._beta0 = beta0
        self._m0 = m0

    def _init_prior(self, obs, adjust_prior=True):
        """
        Initialize prior parameters
        """
        N, D = obs.shape

        # assure nu0 >= D + 1
        if self._nu0 < D + 1:
            self._nu0 += D + 1

        if adjust_prior:
            # adjust prior with observed data
            self._m0 = obs.mean(0)
            self._V0 = np.atleast_2d(np.cov(obs.T)) * self._V0
        else:
            # use simple prior
            self._m0 = np.zeros(D)
            self._V0 = np.identity(D) * self._V0

    def _init_posterior(self, obs):
        """
        Initialize posterior parameters
        """
        K = self.K
        N, D = obs.shape
        avr_N = float(N) / float(K)
        # parameters of posterior mixing coefficients
        self._u = np.ones(K) * (self._u0 + avr_N)
        # parameters of posterior precision matrices
        self._nu = np.ones(K) * (self._nu0 + avr_N)
        self._V = np.tile(np.array(self._V0), (K, 1, 1))
        # parameters of posterior mean vectors
        self._beta = np.ones(K) * (self._beta0 + avr_N)
        self._m, temp = vq.kmeans2(obs, K)  # initialize by K-Means

    def getExpectations(self):
        """
        Calculate expectations of parameters over posterior distribution
        """
        # <pi_k>_Q(pi_k)
        self.pi = E_pi_Dirichlet(self._u)

        # <mu_k>_Q(mu_k,W_k)
        self.mu = np.array(self._m)

        # inv(<W_k>_Q(W_k))
        self.cv = self._V / self._nu[:, np.newaxis, np.newaxis]

        return self.pi, self.mu, self.cv

    def showModel(self, show_mu=False, show_cv=False, min_pi=0.01):
        """
        Obtain model parameters for relavent clusters
        """

        # first take expectations over posterior
        _ = self.getExpectations()

        # then return posterior parameters
        return EMGMM.showModel(self, show_mu, show_cv, min_pi)

    def score(self, obs):
        """
        score the model
        input
          obs [ndarray, shape(N,D)] : observed data
        output
          F [float] : variational free energy of the model
        """
        z, lnP = self.eval_hidden_states(obs)
        F = -lnP + self._KL_div()
        return F

    def _log_like_f(self, obs):
        """
        mean log-likelihood function of of complete data over posterior
            of parameters, <lnP(X,Z|theta)>_Q(theta)
        input
          obs [ndarray, shape (N,D)] : observed data
        output
          lnf [ndarray, shape (N, K)] : log-likelihood
            where lnf[n,k] = <lnP(X_n,Z_n=k|theta_k)>_Q(theta_k)
        """

        lnf = E_lnpi_Dirichlet(self._u)[np.newaxis, :] \
              + log_like_Gauss2(obs, self._nu, self._V, self._beta, self._m)
        return lnf

    def _KL_div(self):
        """
        Calculate KL-divergence of parameter distribution KL[Q(theta)||P(theta)]
        output
          KL [float] : KL-div
        """
        K = self.K

        # first calculate KL-div of mixing coefficients
        KL = KL_Dirichlet(self._u, self._u0)

        # then calculate KL-div of mean vectors and precision matrices
        for k in range(K):
            KL += KL_GaussWishart(self._nu[k], self._V[k], self._beta[k], \
                                  self._m[k], self._nu0, self._V0, self._beta0, self._m0)

        return KL

    def _E_step(self, obs):
        """
        VB-E step
        Calculate variational posterior distribution of hidden states Q(Z)
        output
          L [float] : lower-bound of marginal log-likelihood
        """

        # calculate Q(Z)
        lnP = EMGMM._E_step(self, obs)

        # calculate lower-bound
        KL = self._KL_div()
        L = lnP - KL

        return L

    def _update_parameters(self, min_cv=None):
        """
        Update parameters of variational posterior distribution by precomputed
            sufficient statistics
        """

        K = self.K
        # parameter for mixing coefficients
        self._u = self._u0 + self._N

        # parameters for mean vectors and precision matrices
        # scalar parameters of Gauss-Wishart
        self._nu = self._nu0 + self._N
        self._beta = self._beta0 + self._N
        # vector or matrix parameters of Gauss-Wishart
        for k in range(K):
            self._m[k] = (self._beta0 * self._m0 \
                          + self._N[k] * self._xbar[k]) / self._beta[k]
            dx = self._xbar[k] - self._m0
            self._V[k] = self._V0 + self._C[k] \
                         + (self._N[k] * self._beta0 / self._beta[k]) * np.outer(dx, dx)
