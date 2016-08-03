import numpy as np
from pyvb.sampling import testData

def test_EMGMM(N=5,**keywards):
    from pyvb.emgmm import EMGMM
    X = testData(10000)
    model = EMGMM(N)
    model.fit(X)
    model.showModel()
    print(model.score(X))

def test_VBGMM(N=5,**keywards):
    from pyvb.vbgmm import VBGMM
    X = testData(10000)
    model = VBGMM(N)
    model.fit(X)
    model.plot2d(X)

def test_DPGMM(N=5,**keywards):
    from pyvb.dpgmm import DPGMM
    X = testData(10000)
    model = DPGMM(N)
    model.fit(X)
    model.plot2d(X)



def test_moments(**keywards):
    from pyvb.moments import E_lnpi_Dirichlet,KL_Dirichlet,E_lndetW_Wishart,\
        KL_GaussWishart
    alpha1 = np.array([0.3,0.7])
    alpha2 = np.array([0.5,0.5])
    nu1 = 10
    nu2 = 15
    m1 = np.array([1.0,2.0])
    m2 = m1 * 2.0
    V1 = np.identity(2)
    V2 = np.identity(2) * 3.0
    beta1 = 3
    beta2 = 4
    print(E_lnpi_Dirichlet(alpha1))
    print(KL_Dirichlet(alpha1,alpha2))
    print(E_lndetW_Wishart(nu1,V1))
    print(KL_GaussWishart(nu1,V1,beta1,m1,nu2,V2,beta2,m2))

tests = {"EMGMM":test_EMGMM,"VBGMM":test_VBGMM,"DPGMM":test_DPGMM,\
    "moments":test_moments}

if __name__ == "__main__":
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-m","--mode",dest="mode",default="VBGMM",\
      help="Specify test mode from EMGMM, VBGMM, DPGMM, and moments")
    parser.add_option("-n","--nstate",dest="n",type="int",default=5,\
      help="Number of hidden states")
    parser.add_option("-t","--Total",dest="t",type="int",default=10,\
      help="Number of independent sequences, only needed for hmm test")
    parser.add_option("-r","--rseed",dest="rseed",type="int",\
      help="Seed of the random number generator")
      
    options, args = parser.parse_args()
    np.random.seed(options.rseed)
    tests[options.mode.upper()](N=options.n,T=options.t)    
