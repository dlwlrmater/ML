import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

#读取数据并显示
Path = r'/Users/ternencekk/Downloads/machine-learning-ex-withoutanswer/ex2/ex2data2.txt'
data2 = pd.read_csv(Path, header=None, names=['Test1','Test2','Accepted'])
print(data2.head())

positive = data2[data2['Accepted'].isin([1])]
negative = data2[data2['Accepted'].isin([0])]
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(positive['Test1'], positive['Test2'], s=50,  c='b', marker='o', label='Accepted')
ax.scatter(negative['Test1'], negative['Test2'], s=50,  c='r', marker='x', label='Not Accepted')
ax.legend()#在图形中加入颜色不同的备 注
ax.set_xlabel('Test1 Score')
ax.set_ylabel('Test2 Score')
plt.show()

# 处理数据
degree = 5
x1 = data2['Test1']
x2 = data2['Test2']
data2.insert(3, 'Ones', 1)

for i in range(1, degree):
    for j in range(0, i):
        data2['F' + str(i) + str(j)] = np.power(x1, i-j) * np.power(x2, j)

data2.drop('Test1', axis=1, inplace=True)
data2.drop('Test2', axis=1, inplace=True)
print(data2.head())

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#定义代价函数
def costReg(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    reg = (learningRate / (2 * len(X)) * np.sum(np.power(theta[:,1:theta.shape[1]],2)))
    return  np.sum(first - second) / len(X) + reg

#定义梯度下降法
def gradient(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    # print(X)
    parameters = int(theta.ravel().shape[1])  # 计算需求解的参数个数
    grad = np.zeros(parameters)#创建一个空矩阵用来放每一步计算的值

    error = sigmoid(X * theta.T) - y
    # print('error\n',error.shape)
    for i in range(parameters):
        term = np.multiply(error, X[:, i])
        # print('X[:,i]\n',X[:,i].shape)
        print('term\n',term)
        print('------------------------')
        if (i == 0):
            grad[i] = np.sum(term) / len(X)
        else:
            grad[i] = (np.sum(term) / len(X)) + ((learningRate / len(X)) * theta[:, i])

    return grad

cols = data2.shape[1]
X2 = data2.iloc[:,1:cols]
y2 = data2.iloc[:,0:1]


X2 = np.array(X2.values)
y2 = np.array(y2.values)
theta2 = np.zeros(11)

learningRate = 0.001
costReg(theta2, X2, y2, learningRate)
aaa = gradient(theta2, X2, y2, learningRate)
print(aaa)
print(aaa.shape)