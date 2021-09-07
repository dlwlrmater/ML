import types
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# path =r'/Users/ternencekk/Downloads/machine-learning-ex-withoutanswer/ex1/ex1data1.txt'
path = r'C:\Users\dell\OneDrive\machine-learning-ex-withoutanswer\ex1\ex1data1.txt'
data = pd.read_csv(path,header=None,names=['population','profit'])
# plt.scatter(data.population,data.profit)
plt.show()
# 加入x0 为0的一列
data.insert(0,"ones",1)
# print(data.head())
# 查看data的size
cols = data.shape[1]
# print(data.shape)
# 分开 x & y
x = data.iloc[:,:cols-1]
# y保证其为dataframe
y = data.iloc[:,cols-1:cols]
# print(x.head())
# 把x&y从dataframe变成np.array 方便阶乘运算
# x.shape  (97,2)
x = np.array(x.values)
y = np.array(y.values)
# matlab zeros(2,1)
theta = np.zeros((1,2))
def compute(x,y,theta):
    m = x.shape[0]
    J = np.sum(np.power(x.dot(theta.T)-y,2))/2/m
    return J
s = compute(x,y,theta)
# print(s)


def gradientdescent(x,y,theta,iter,lr):
    m = x.shape[0]
    aa = theta
    costgrad = np.zeros((iter,1))
    for i in range(iter):
        # for i in range(x.shape[1]):
        r1 = np.sum((x.dot(theta.T)-y)*x[:,0:1])
        r2 = np.sum((x.dot(theta.T)-y)*x[:,1:2])
        # print(r1.shape)

        aa[0][0] = aa[0][0] - r1 * lr /m
        aa[0][1] = aa[0][1] - r2 * lr / m

        costgrad[i,0] = compute(x, y, aa)
    theta = aa
    J = compute(x,y,theta)
    # theta为每次迭代之后的梯度下降theta
    # costgrad为每次迭代后的cost
    # J为最终的cost
    return theta,J,costgrad

iters = 1000
a,J,cost = gradientdescent(x,y,theta,iters,lr=0.01)
print(a,J)

x =np.linspace(data.population.min(),data.population.max(),100)
y = a[0,0]+a[0,1]*x
fig,ax = plt.subplots(figsize=(12,8))
ax.plot(x,y,'r',label='Prediction')
ax.scatter(data.population,data.profit,label='Training Data')
ax.legend()
ax.set_xlabel('Population')
ax.set_ylabel('profit')
ax.set_title('Predicted Profit vs Population Size')
plt.show()

fig,ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters),cost,'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs Training Epoch')
plt.show()