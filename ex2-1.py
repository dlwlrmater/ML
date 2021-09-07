import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

Path = r'C:\Users\steve\Downloads\machine-learning-ex2\ex2\ex2data1.txt'
data = pd.read_csv(Path, header=None, names=['exam1','exam2','Admitted'])
# print(data.head())

positive = data[data['Admitted'].isin([1])]
# print(positive.shape)
negative = data[data['Admitted'].isin([0])]
# print(negative.shape)
fig, ax = plt.subplots(figsize=(12,8))  #定义图的大小
ax.scatter(positive['exam1'], positive['exam2'], s=50,  c='b', marker='o', label='Admitted')    #具体作用参见scatter函数参数设置
ax.scatter(negative['exam1'], negative['exam2'], s=50,  c='r', marker='x', label='Not Admitted')
ax.legend()  #在图形中加入颜色不同的备注
ax.set_xlabel('exam1 Score')
ax.set_ylabel('exam2 Score')
plt.show()

# 引入Sigmod函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#画图展示Sigmod函数
nums = np.arange(-10, 10, step=1)
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(nums, sigmoid(nums), 'r')
ax.set_xlabel(' X ')
ax.set_ylabel(' sigmoid(x) ')
plt.show()

# 定义代价函数
def cost(theta, X, y):
    theta = np.mat(theta)
    X = np.mat(X)
    y = np.mat(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    return np.sum(first - second) / (len(X))

# 数据处理
data.insert(0, 'Ones', 1)  #引入X0=1
# print(data.head())
cols = data.shape[1]
# print(cols)
X = data.iloc[:,0:cols-1]  #X是所有行，去掉最后一列
y = data.iloc[:,cols-1:cols]  #y是所有行的最后一列
X = np.array(X.values)  #将X的值转化为矩阵形式，方便计算
y = np.array(y.values)
theta = np.zeros(3)  #初始化theta为0

# print(X.shape, theta.shape, y.shape)
# print(cost(theta, X, y))

#第一种方法
#定义梯度下降法
def gradient(theta, X, y):
    theta = np.mat(theta)
    X = np.mat(X)
    y = np.mat(y)

    parameters = int(theta.ravel().shape[1])  # 计算需求解的参数个数
    grad = np.zeros(parameters)

    error = sigmoid(X * theta.T) - y
    for i in range(parameters):
        term = np.multiply(error, X[:,i])
        grad[i] = np.sum(term) / len(X)
    return grad

# print(gradient(theta, X, y))  #这里只是输出一次迭代的效果，查看梯度下降函数是否运行正确

#使用Scipy的优化函数来优化参数，计算代价，自动迭代，给出结果
result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))

print(result)
# print(cost(result[0], X, y))
