# https://blog.csdn.net/qq_26402041/article/details/109301645

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder
from scipy.optimize import minimize
from sklearn.metrics import classification_report#这个包是评价报告
import time

start = time.time()
# data = loadmat(r'/Users/ternencekk/Downloads/machine-learning-ex-withoutanswer/ex4/ex4data1.mat')
data = loadmat(r'C:\Users\dell\OneDrive\machine-learning-ex\machine-learning-ex\ex4\ex4data1.mat')

# print(data)

X = data['X']
y = data['y']
# print(X.shape)
# print(y.shape)
# print(y)

# weight = loadmat(r'/Users/ternencekk/Downloads/machine-learning-ex-withoutanswer/ex4/ex4weights.mat')
weight = loadmat(r'C:\Users\dell\OneDrive\machine-learning-ex\machine-learning-ex\ex4\ex4weights.mat')
theta1,theta2 = weight['Theta1'],weight['Theta2']
# print(theta1.shape)
# print(theta2.shape)

sample_idx = np.random.choice(np.arange(X.shape[0]),100)
sample_images = X[sample_idx,:]
fig,ax_array = plt.subplots(nrows=10,ncols=10,sharex=True,sharey=True,figsize=(12,12))
for r in range(10):
    for c in range(10):
        ax_array[r,c].matshow(np.array(sample_images[10*r+c].reshape((20,20))).T,cmap = matplotlib.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
# plt.show()

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoidGradient(z):
    return np.multiply(sigmoid(z),(1-sigmoid(z)))

def forward_propagate(X,theta1,theta2):
    m = X.shape[0]
    a1 = np.insert(X,0,values=np.ones(m),axis=1)
    z2 = a1@theta1.T
    a2 = np.insert(sigmoid(z2),0,values=np.ones(m),axis=1)
    z3 = a2@theta2.T
    h = sigmoid(z3)
    return a1,z2,a2,z3,h

def cost(theta1,theta2,input_size,hidden_size,num_lables,X,y,learning_rate):
# def cost(theta1, theta2, X, y,num_lables):
    m = X.shape[0]
    y1 = y.copy()
    X = np.mat(X)
    y = np.mat(y)
    a1,z2,a2,z3,h = forward_propagate(X,theta1,theta2)
    J = 0
    # print('y.shape',y.shape)
    # print('h.shape',h.shape)

    # 自己代码
    # ylabel = np.zeros((m,num_lables))
    # for i in range(m):
    #     ylabel[i,y1[i]-1] = 1
    # print(ylabel[0])
    # J = np.sum(np.multiply(-ylabel,np.log(h)))-np.sum(np.multiply((1-ylabel),np.log(1-h)))+
    # print((np.log(h)).shape)
    # J = J/m
    # print('m',m)


    for i in range(m):
        first_term = np.multiply(-y[i,:],np.log(h[i,:]))
        second_term = np.multiply((1-y[i,:]),np.log(1-h[i,:]))
        J += np.sum(first_term-second_term)
    J = J/m

    J = J + learning_rate*(np.sum(np.power(theta1,2))+np.sum(np.power(theta2,2)))/m/2
    return J


# 通过OneHotEncoder把[10]变成[0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
# OneHotEncoder CSDN : https://blog.csdn.net/weixin_40807247/article/details/82812206
encoder = OneHotEncoder(sparse=False)
y_onehot = encoder.fit_transform(y)
# print(y[0])
# print(y_onehot[0])
# print('y_onehot',y_onehot)
# print('y_onehot.shape',y_onehot.shape)
input_size = 400
hidden_size = 25
num_labels = 10
learning_rate = 1

resultJ = cost(theta1,theta2,input_size,hidden_size,num_labels,X,y_onehot,learning_rate)
print('J',resultJ)


# 测试归一化cost
# a = cost(theta1,theta2,input_size,hidden_size,num_lables,X,y_onehot,learning_rate)
# 测试sigmoidGrandient函数
# a = sigmoidGradient(0)

# 随机初始化
param = (np.random.random(size=hidden_size * (input_size+1) + num_labels * (hidden_size + 1)) - 0.5) * 0.24
param = param.reshape(len(param),1)


def backprop(param,input_size,hidden_size,num_labels,X,y,learning_rate):
    m = X.shape[0]
    X = np.mat(X)
    y = np.mat(y)

    theta1 = np.mat(np.reshape(param[:hidden_size*(input_size+1)],(hidden_size,(input_size+1))))
    theta2 = np.mat(np.reshape(param[hidden_size*(input_size+1):],(num_labels,(hidden_size+1))))

    a1,z2,a2,z3,h = forward_propagate(X,theta1,theta2)
    # print('a1.shape',a1.shape)
    # print('a2.shape',a2.shape)
    # print('theta1.shape',theta1.shape)
    # print('theta2.shape',theta2.shape)

    # 反向传播中每个theta的 △D 变化量
    delta1 = np.zeros(theta1.shape)
    delta2 = np.zeros(theta2.shape)

    J = cost(theta1,theta2,input_size,hidden_size,num_labels,X,y,learning_rate)

    # 结果层→hiddenlayer

    for i in range(m):

        δ3 = h[i,:]-y[i,:]  # (1,10)
        z2t = np.insert(z2[i,:],0,values=1,axis=1)  # (1,26)
        δ2 = np.multiply(δ3*theta2,sigmoidGradient(z2t))  # (1,26)
        # (10,1) * (1,26)
        delta2 = delta2 + δ3.T * a2[i,:]
        # (25,1) * (1,401)
        delta1 = delta1 + δ2[:,1:].T * a1[i,:]
    delta1 = delta1/m
    delta2 = delta2/m
    delta2[:, 1:] = delta2[:, 1:] + (learning_rate * theta2[:, 1:]) / m
    delta1[:,1:] = delta1[:,1:] + (learning_rate*theta1[:,1:])/m

    # print('delta1.shape',delta1.shape)
    # print('delta2.shape',delta2.shape)
    grad = np.concatenate((np.ravel(delta1),np.ravel(delta2)))

    return J,grad




a = backprop(param, input_size, hidden_size, num_labels, X, y_onehot, learning_rate)
print(a)

print('param.shape',param.shape)


fmin = minimize(fun=backprop, x0=param, args=(input_size, hidden_size, num_labels, X, y_onehot, learning_rate), method='TNC', jac=True, options={'maxiter':350})
print(fmin)


# 测试bp模型结果
X1 = np.mat(X)
thetafinal1 = np.mat(np.reshape(fmin.x[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
thetafinal2 = np.mat(np.reshape(fmin.x[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
a1, z2, a2, z3, h = forward_propagate(X, thetafinal1, thetafinal2 )
y_pred = np.array(np.argmax(h, axis=1) + 1)
print(y_pred)
print(classification_report(y, y_pred))
end = time.time()
print(end-start)

