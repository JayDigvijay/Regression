from Normal_Equations import test_size, W, X1test, X2test, Ytest
from math import sqrt
import numpy as np

Ypred = np.zeros((test_size,), dtype = 'int64' )
Ys = 0
Yss = 0
ys = 0
yss = 0
Yys = 0


for i in range(test_size):
	Ypred[i] = (W[0] + W[1]*X1test[i] + W[2]*X2test[i])
	Yss += Ypred[i]*Ypred[i]
	ys += Ytest[i]
	yss += Ytest[i]*Ytest[i]
	Ys += Ypred[i]
	Yys += Ytest[i]*Ypred[i]
den = sqrt((test_size*Yss - (Ys*Ys))*(test_size*yss - (ys*ys)))
num = test_size*Yys - Ys*ys
Rsq = num/den
print(Rsq)