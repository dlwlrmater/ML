from sklearn import linear_model
import numpy as np
import pandas as pd

x_train = np.arange(18).reshape(6,3)
print(x_train)
y_train = np.array([1,0,1,0,1,0])
b = np.mat([[2,9,10],[3,22,3],[6,233,1313]])
c = np.array([0,1,1])

clf = linear_model.LogisticRegression()
clf.fit(x_train, y_train)
rr = clf.predict(b,c)
print(rr)