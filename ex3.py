    # https://blog.csdn.net/qq_26402041/article/details/109194221

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.io import loadmat
from sklearn.metrics import classification_report
from scipy.optimize import minimize
np.set_printoptions(threshold = 1e6)

# 读取ex3data1.mat
# data = loadmat(r'/Users/ternencekk/Downloads/machine-learning-ex-withoutanswer/ex3/ex3data1.mat')
data = loadmat(r'C:\Users\dell\OneDrive\machine-learning-ex-withoutanswer\ex3\ex3data1.mat')
# print(data)

# 随机选取100行 == 随机选择了100个数字
sample_idx = np.random.choice(np.arange(data['X'].shape[0]),100)
# print(sample_idx)
# print(data['X'])
# print(data['X'].shape[0])
# print(np.arange(data['X'].shape[0]))
sample_image = data['X'][sample_idx,:]
# print(sample_image)

# 画图  100个数字
fig,ax_array = plt.subplots(nrows=10,ncols=10,sharey=True,sharex=True,figsize=(12,12))
for r in range(10):
    for c in range(10):
        # ax_array[r,c]在图上的位置
        # np.array(sample_image[10*r+c].reshape((20,20)))  400像素的数字
        ax_array[r,c].matshow(np.array(sample_image[10*r+c].reshape((20,20))).T,cmap=matplotlib.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
plt.show()

def sigmoid(z):
    return 1/(1+np.exp(-z))

def costFunction(theta,X,y,lam):
    m = X.shape[0]
    # print(X.shape)
    # print(theta.shape)
    z = np.mat(sigmoid(X@theta))
    # print('z',z.shape)
    theta1 = theta.copy()
    theta1[0] = 0
    J = np.sum(np.multiply(-y,np.log(z)) - np.multiply((1-y),np.log(1-z)))/m + lam/2/m*np.sum(np.power(theta1,2))
    return J

def gradient(theta,X,y,lam):
    m = X.shape[0]
    z = sigmoid(X @ theta)
    # print('X',X.shape)
    # print('z',z.shape)
    # print('theta',theta.shape)
    # print('y',y.shape)
    grad = (X.T@(z-y))/m + lam/m*theta
    # print('grad',grad.shape)
    return grad

X = data['X']
print('X.shape',X.shape)
y = data['y']
theta = np.array([[-2],[-1],[1],[2]])

# First row
Fr = np.ones((5, 1))
# Second row
Sr = np.array([i for i in range(1, 16)]).reshape(3,5).T/10
# 方法一 np.c_
X_t = np.c_[Fr,Sr]
# 方法二 np.insert() 在Sr在每行(axis = 1)[0]加入 1
X_t = np.insert(Sr,0,values=1,axis=1)
# print(X_t)
y_t = np.array([1,0,1,0,1]).reshape(5,1)
J = costFunction(theta,X_t,y_t,3)
print(J)

# a = np.array([i for i in range(1,11)])
# print(a)
# b = 3
# print(np.disp(a==b))

# lam学习速率
def oneVsAll(X,y,num_labels,lam):
    # row=400
    row = X.shape[1]
    all_theta = np.zeros((num_labels,row+1))   # (10,401)
    # 加入theta0
    X = np.insert(X,0,values=1,axis=1)   # (5000,401)
    for i in range(1,num_labels+1):
        theta = np.mat(np.zeros(row+1)).T
        y_i = np.array([1 if label ==i else 0 for label in y])

        # print('zz',theta.shape)
        # 根据学习速率直接达到最优解
        fmin = minimize(fun=costFunction,x0=theta,args=(X,y_i,lam),method='TNC',jac=gradient)
        # print(fmin)
        # fmin.x是minimize的结果
        all_theta[i-1,:]=fmin.x
    return all_theta

zz = oneVsAll(X,y,10,0.01)
# print(zz)
# print(zz.shape)

def predictOneVsAll(all_theta,X):
    X = np.insert(X,0,values=1,axis=1)
    h = sigmoid(X@all_theta.T)
    # 返回每行的最大值
    # a = np.amax(h,axis=1)
    # 横着比较，返回列号
    a = np.argmax(h,axis=1)
    return a+1

x = predictOneVsAll(zz,X)
# print(x.ravel())
# print(y.ravel())
# print(x.shape)
# print(y)

asss = np.mean(y.ravel()==x.ravel())
print('{}%'.format(asss*100))
print(classification_report(y,x))
