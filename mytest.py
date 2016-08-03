import numpy as np
from pyvb.sampling import testData
from pyvb.emgmm import EMGMM
from pyvb.vbgmm import VBGMM

X = testData(100)
model = VBGMM(K=5)
print(X)
model.fit(X)
model.showModel()
print(model.score(X))
model.plot2d(X)
