import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.linear_model import LogisticRegressionCV


Path = r'/Users/ternencekk/Downloads/machine-learning-ex-withoutanswer/ex2/ex2data2.txt'
data = pd.read_csv(Path, header=None, names=['Test1','Test2','Accepted'])
# print(data2.head())

def sigmoid(z):
    return 1 / (1 + np.exp(- z))


def cost(theta, X, y):
    first = (-y) * np.log(sigmoid(X @ theta))
    second = (1 - y)*np.log(1 - sigmoid(X @ theta))
    return np.mean(first - second)



def costReg(theta, X, y, l=1):
    # 不惩罚第一项
    _theta = theta[1:]
    reg = (l / (2 * len(X))) * (_theta @ _theta)  # _theta@_theta == inner product

    return cost(theta, X, y) + reg

# add a ones column - this makes the matrix multiplication work out easier
if 'Ones' not in data.columns:
    data.insert(0, 'Ones', 1)

# set X (training data) and y (target variable)
X = np.mat(data.iloc[:, :-1])  # Convert the frame to its Numpy-array representation.
y = np.mat(data.iloc[:, -1])  # Return is NOT a Numpy-matrix, rather, a Numpy-array.
print(X)

theta = np.zeros((X.shape[1],1))
print(theta.shape)

print(cost(theta, X, y))


def feature_mapping(x1, x2, power):
    data = {}
    for i in np.arange(power + 1):
        for p in np.arange(i + 1):
            data["f{}{}".format(i - p, p)] = np.power(x1, i - p) * np.power(x2, p)

#     data = {"f{}{}".format(i - p, p): np.power(x1, i - p) * np.power(x2, p)
#                 for i in np.arange(power + 1)
#                 for p in np.arange(i + 1)
#             }
    return pd.DataFrame(data)

plt.scatter(data.Test1,data.Test2,c='black',marker='+',label='y=1',s=50)
plt.scatter(data.Test1,data.Test2,c='yellow',marker='o',label='y=0',s=50)
plt.xlabel('Microchip Test1')
plt.ylabel('Microchip Test2')
plt.legend()


x = np.linspace(-1, 1.5, 250)
xx, yy = np.meshgrid(x, x)

final_theta = np.array([8.47457627e-03,1.87880932e-02,7.77711864e-05,5.03446395e-02,1.15013308e-02,3.76648474e-02,1.83559872e-02,7.32393391e-03
,8.19244468e-03,2.34764889e-02,3.93486234e-02,2.23923907e-03
,1.28600503e-02,3.09593720e-03,3.93028171e-02,1.99707467e-02
,4.32983232e-03,3.38643902e-03,5.83822078e-03,4.47629067e-03
,3.10079849e-02,3.10312442e-02,1.09740238e-03,6.31570797e-03
,4.08503006e-04,7.26504316e-03,1.37646175e-03,3.87936363e-02])


z = np.mat(feature_mapping(xx.ravel(), yy.ravel(), 6))
z = z @ final_theta
z = z.reshape(xx.shape)

# plot_data()
plt.contour(xx, yy, z, 0)
plt.ylim(-.8, 1.2)
plt.show()


