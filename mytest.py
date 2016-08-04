import numpy as np
from pyvb.sample_vars import testData
from pyvb.emgmm import EMGMM
from pyvb.vbgmm import VBGMM

X = testData(100)
#X = testData(10)
model = VBGMM(K=5)
print(X)
model.fit(X)
model.showModel()
print(model.score(X))
model.plot2d(X)
