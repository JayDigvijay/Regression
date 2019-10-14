import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


dataset=pd.read_csv('D:\shrey\Documents\StudyMaterial\\fods_project\Data.csv')
X1 = dataset.iloc[:,1].values
X2 = dataset.iloc[:,2].values
Y = dataset.iloc[:,3].values

indices = list(range(X1.shape[0]));
num_inst = int(0.7*X1.shape[0]);
np.random.shuffle(indices)
train_ind = indices[:num_inst]
test_ind = indices[num_inst:]

X1train, X1test = X1[train_ind], X1[test_ind]
X2train, X2test = X2[train_ind], X2[test_ind]
Ytrain, Ytest = Y[train_ind], Y[test_ind]

def func(w):
    sum = 0
    for(i:range(num_inst)):
        sum =sum + (w[0]+w[1]*X1train[i]+w[2]*X2train[i])
    return sum


#fig = plt.figure()
#ax = plt.axes(projection="3d")
#ax.scatter3D(X1,X2,Y);
#plt.show()

