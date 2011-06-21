#!/usr/bin/python

import numpy as np
from numpy.random import random,randn,dirichlet
from scikits.learn.mixture import logsum,lmvnpdf,sample_gaussian

try:
  import _hmmf
  #import _hmmc
  ext_imported = True
except ImportError:
  print "cython_ext was not imported"
  ext_imported = False

class _BaseHMM():
  """
  Base HMM class.
  Any HMM class should be an inheritent of it.
  No instance of this class can be made.
  """
  def __init__(self,N):
    self.n_states = N # number of hidden states
    self.lnpi = np.log(np.tile(1.0/N,N)) # log initial probability
    self.lnA = np.log(dirichlet([1.0]*N,N)) # log transition probability
    
  def log_like_f(obs):
    """
    Calculate log-likelihood of emissions
    """
    pass

  def _initialize_HMM(self,obs):
    """
    Do some initializations
    """
    pass
  
  def _allocate_temp(self,obs):
    """
    Allocate tempolary space for running forward-backward algorithm
    see 
    """
    T = len(obs) 
    lnalpha = np.zeros((T,self.n_states)) #  log forward variable
    lnbeta = np.zeros((T,self.n_states)) # log backward variable
    lneta = np.zeros((T-1,self.n_states,self.n_states)) 
    return lnalpha, lnbeta, lneta
  
  def _forward(self,lnf,lnalpha,use_ext="F"):
    """
    Use forward algorith to calculate forward variables and loglikelihood
    input
      lnf [ndarray, shape (n x nmix)] : loglikelihood of emissions
      lnalpha [ndarray, shape (n x nmix)] : log forward variable
      use_ext [("F","C",None)] : flag to use extension
    output
      lnalpha [ndarray, shape (n x nmix)] : log forward variable
      lnP [float] : lnP(X|theta)
    """
    T = len(lnf) 
    lnalpha *= 0.0 
    if use_ext and ext_imported:
      if use_ext in ("c","C"):
        _hmmc._forward_C(T,self.n_states,self.lnpi,self.lnA,lnf,lnalpha)
      elif use_ext in ("f","F"):
        lnalpha = _hmmf.forward_f(self.lnpi,self.lnA,lnf)
      else :
        raise ValueError, "ext_use must be either 'C' or 'F'"
    else:
      lnalpha[0,:] = self.lnpi + lnf[0,:]
      for t in xrange(1,T):
        lnalpha[t,:] = logsum(lnalpha[t-1,:] + self.lnA.T,1) \
            + lnf[t,:]
    return lnalpha,logsum(lnalpha[-1,:])

  def _backward(self,lnf,lnbeta,use_ext="F"):
    """
    Use backward algorith to calculate backward variables and loglikelihood
    input
      lnf [ndarray, shape (n x nmix)] : loglikelihood of emissions
      lnbeta [ndarray, shape (n x nmix)] : log backward variable
      use_ext [("F","C",None)] : flag to use extension
    output
      lnbeta [ndarray, shape (n x nmix)] : log backward variable
      lnP [float] : lnP(X|theta)
    """
    T = len(lnf) 
    lnbeta *= 0.0
    if ext_imported:
      if use_ext in ("c","C"):
        _hmmc._backward_C(T,self.n_states,self.lnpi,self.lnA,lnf,lnbeta)
      elif use_ext in ("f","F"):
        lnbeta = _hmmf.backward_f(self.lnpi,self.lnA,lnf)
      else :
        raise ValueError, "ext_use must be either 'C' or 'F'"
    else:
      lnbeta[T-1,:] = 0.0 
      for t in xrange(T-2,-1,-1):
        lnbeta[t,:] = logsum(self.lnA + lnf[t+1,:] + lnbeta[t+1,:],1)
    return lnbeta,logsum(lnbeta[0,:] + lnf[0,:] + self.lnpi)

  def eval(self,obs,use_ext="F"):
    """
    Performe one Estep.
    Then obtain variational free energy and posterior over hidden states
    """
    lnf = self.log_like_f(obs)
    lnalpha, lnbeta, lneta = self._allocate_temp(obs)
    lneta, lngamma, lnP = self._Estep(lnf,lnalpha,lnbeta,lneta,use_ext)
    return -lnP,np.exp(lngamma)

  def decode(self,obs,use_ext="F"):
    """
    Get the most probable cluster id
    """
    posterior = self.eval(obs,use_ext)[1]
    return posterior.argmax(1)

  def fit(self,obs,niter=1000,eps=1.0e-4,ifreq=10,init=True,use_ext="F"):
    """
    Fit the model parameters iteratively via EM algorithm
    """

    # performe initialization if needed
    if init:
      self._initialize_HMM(obs)
      old_F = 1.0e20

    # allocate temporary for Forward-Backward
    lnalpha, lnbeta, lneta = self._allocate_temp(obs)
    
    # main loop
    for i in xrange(niter):
      # E step
      lnf = self.log_like_f(obs)
      lneta, lngamma, lnP = self._Estep(lnf,lnalpha,lnbeta,lneta,use_ext)
      # check for convergence 
      F = -lnP
      dF = F - old_F
      if(abs(dF) < eps):
        print "%8dth iter, Free Energy = %12.6e, dF = %12.6e" %(i,F,dF)
        print "%12.6e < %12.6e Converged" %(dF, eps)
        break
      if i % ifreq == 0 and dF < 0.0:
        print "%6dth iter, F = %15.8e  df = %15.8e"%(i,F,dF)
      elif dF > 0.0:
        print "%6dth iter, F = %15.8e  df = %15.8e warning"%(i,F,dF)
      old_F = F
      # M step
      self._Mstep(obs,lneta,lngamma,use_ext)
    
    return self

  def fit_multi(self,obss,niter=1000,eps=1.0e-4,ifreq=10,init=True,use_ext="F"):
    """
    Performe EM step for multiple iid time-series
    """
    nobss = len(obss) # number of trajectories
    nobs = [len(obs) for obs in obss] # numbers of observations in all trajs
    i_max_obs = np.argmax(nobs)
    obs_flatten = np.vstack(obss) # flattened observations (sum(nobs)xdim)
    nmix = self.n_states

    # get posistion id for each traj
    # i.e. obss[i] = obs[pos_ids[i][0]:pos_ids[i][1]]
    pos_ids = [] 
    j = 0
    for i in xrange(nobss):
      pos_ids.append((j,j+nobs[i]))
      j += nobs[i]

    if init:
      self._initialize_HMM(obs_flatten)
      old_F = 1.0e20   

    # allocate space for forward-backward
    lneta = []
    lngamma = []
    for nn in xrange(nobss):
      lneta.append(np.zeros((len(obss[nn])-1,nmix,nmix)))
      lngamma.append(np.zeros((len(obss[nn]),nmix)))
    lnalpha, lnbeta, lneta_temp = self._allocate_temp(obss[i_max_obs])

    for i in xrange(niter):
      lnP = 0.0
      lnf = self.log_like_f(obs_flatten)
      for nn in xrange(nobss):
        Ti,Tf = pos_ids[nn]
        e, g, p = self._Estep(lnf[Ti:Tf],lnalpha[:nobs[nn]],\
            lnbeta[:nobs[nn]],lneta_temp[:nobs[nn]-1],use_ext)
        lneta[nn] = e[:]
        lngamma[nn] = g[:]
        lnP += p

      F = -lnP 
      dF = F - old_F
      if(abs(dF) < eps):
        print "%8dth iter, Free Energy = %12.6e, dF = %12.6e" %(i,F,dF)
        print "%12.6e < %12.6e Converged" %(dF, eps)
        break
      if i % ifreq == 0 and dF < 0.0:
        print "%6dth iter, F = %15.8e  df = %15.8e"%(i,F,dF)
      elif dF > 0.0:
        print "%6dth iter, F = %15.8e  df = %15.8e warning"%(i,F,dF)
      old_F = F
      self._Mstep(obs_flatten,lneta,lngamma,use_ext,multi=True)

    return self

  def _Estep(self,lnf,lnalpha,lnbeta,lneta,use_ext="F"):
    T = len(lnf)

    # Forward-Backward algorithm
    lnalpha, lnP_f = self._forward(lnf,lnalpha,use_ext)
    lnbeta, lnP_b = self._backward(lnf,lnbeta,use_ext)

    # check if forward and backward were done correctly
    dlnP = lnP_f-lnP_b
    if abs(dlnP) > 1.0e-6:
      print "warning forward and backward are not equivalent"

    # compute lneta for updating transition matrix
    if ext_imported and use_ext:
      if use_ext in ("c","C"):
        _hmmc._compute_lnEta_C(T,self.n_states,lnalpha,self.lnA, \
            lnbeta,lnf,lnP_f,lneta)
      elif use_ext in ("f","F"):
        lneta = _hmmf.compute_lneta_f(lnalpha,self.lnA,lnbeta,lnf,lnP_f)
      else :
        raise ValueError, "ext_use must be either 'C' or 'F'"
    else:
      for i in xrange(self.n_states):
        for j in xrange(self.n_states):
          for t in xrange(T-1):
            lneta[t,i,j] = lnalpha[t,i] + self.lnA[i,j,] + \
                lnf[t+1,j] + lnbeta[t+1,j]
      lneta -= lnP_f

    # compute lngamma for posterior on hidden states
    lngamma = lnalpha + lnbeta - lnP_f
    
    return lneta,lngamma,lnP_f

  def _Mstep(self,obs,lneta,lngamma,use_ext="F",multi=False):
    self._calcSufficientStatistic(obs,lneta,lngamma,multi)
    self._updatePosteriorParameters(obs,lneta,lngamma,multi)

  def _calcSufficientStatistic(self,obs,lneta,lngamma,multi=False):
    pass

  def _updatePosteriorParameters(self,obs,lneta,lngamma,multi=False):
    if multi :
      # for multiple trajectories
      lg = np.vstack(lngamma)
      le = np.vstack(lneta)
      lngamma_sum = logsum(lg,0)
      #self.lnpi = lngamma_sum - logsum(lg)
      self.lnpi = logsum(np.array([lg_temp[0] for lg_temp in lngamma]),0)
      self.lnA = logsum(le,0) - logsum(np.vstack(\
          [lg_temp[:-1] for lg_temp in lngamma]),0)[:,np.newaxis]
      
    else:
      lngamma_sum = logsum(lngamma,0)
      #self.lnpi = lngamma_sum - logsum(lngamma_sum)
      # update initial probability
      self.lnpi = lngamma[0] 
      # update transition matrix
      self.lnA = logsum(lneta,0) \
          - logsum(lngamma[:-1],0)[:,np.newaxis]
    return lngamma_sum

class MultinomialHMM(_BaseHMM):
  def __init__(self,N,M):
    _BaseHMM.__init__(self,N)
    self.m_states = M
    self.lnB = np.log(dirichlet([1.0]*M,N))
  
  def log_like_f(self,obs):
    return self.lnB[:,obs].T

  def simulate(self,T):
    pi_cdf = np.exp(self.lnpi).cumsum()
    A_cdf = np.exp(self.lnA).cumsum(1)
    B_cdf = np.exp(self.lnB).cumsum(1)
    z = np.zeros(T,dtype=np.int)
    o = np.zeros(T,dtype=np.int)
    r = random((T,2))
    z[0] = (pi_cdf > r[0,0]).argmax()
    o[0] = (B_cdf[z[0]] > r[0,1]).argmax()
    for t in xrange(1,T):
      z[t] = (A_cdf[z[t-1]] > r[t,0]).argmax()
      o[t] = (B_cdf[z[t]] > r[t,1]).argmax()
    return z,o

  def _updatePosteriorParameters(self,obs,lneta,lngamma,multi=False):
    logsum = _BaseHMM._updatePosteriorParameters(self,obs,lneta,lngamma)
    for j in xrange(self.m_states):
      self.lnB[:,j] = logsum(lngamma[obs==j,:],0) - lngamma_sum

class GaussianHMM(_BaseHMM):
  def __init__(self,N):
    _BaseHMM.__init__(self,N)
      
  def _initialize_HMM(self,obs,params="mc"):
    _BaseHMM._initialize_HMM(self,obs)
    T,D = obs.shape
    if "m" in params:
      self.m = randn(self.n_states,D)
    if "c" in params:
      self.cv = np.tile(np.identity(D),(self.n_states,1,1))
    
  def log_like_f(self,obs):
    return lmvnpdf(obs,self.m,self.cv,"full")
    
  def simulate(self,T):
    N,D = self.m.shape
    pi_cdf = np.exp(self.lnpi).cumsum()
    A_cdf = np.exp(self.lnA).cumsum(1)
    z = np.zeros(T,dtype=np.int)
    o = np.zeros((T,D))
    r = random(T)
    z[0] = (pi_cdf > r[0]).argmax()
    o[0] = sample_gaussian(self.m[z[0]],self.cv[z[0]],"full")
    for t in xrange(1,T):
      z[t] = (A_cdf[z[t-1]] > r[t]).argmax()
      o[t] = sample_gaussian(self.m[z[t]],self.cv[z[t]],"full")
    return z,o
    
  def _updatePosteriorParameters(self,obs,lneta,lngamma,multi=False):
    logsum = _BaseHMM._updatePosteriorParameters(self,obs,lneta,lngamma,multi)
    if multi:
      posteriors = np.exp(np.vstack(lngamma))
    else:
      posteriors = np.exp(lngamma)
    for k in xrange(self.n_states):
        post = posteriors[:, k]
        norm = 1.0 / post.sum()
        self.m[k] = np.dot(post,obs) * norm
        avg_cv = np.dot(post * obs.T, obs) * norm
        self.cv[k] = avg_cv - np.outer(self.m[k], self.m[k])

test_model = GaussianHMM(3)
test_model.m = np.array([[0.0,0.0],[1.0,3.0],[-3.0,0.0]])
test_model.cv = np.tile(np.identity(2),(3,1,1))
test_model.lnA = np.log([[0.9,0.05,0.05],[0.1,0.7,0.2],[0.1,0.4,0.5]])
    
if __name__ == "__main__":
  from sys import argv
  from scipy.linalg import eig
  ifreq = 10
  model = GaussianHMM(int(argv[1]))
  os = []
  zs = []
  for i in range(int(argv[2])):
    z,o = test_model.simulate(50)
    os.append(o)
    zs.append(z)
  o2 = np.vstack(os)
  if "-mult" in argv :
    model.fit_multi(os,ifreq=ifreq)
  else:
    model.fit(o2,ifreq=ifreq)
  print model.m
  print model.cv
  print np.exp(model.lnpi)
  A = np.exp(model.lnA)
  e_val,e_vec = eig(A.T)
  print e_val.real
  print e_vec
