import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn import linear_model


# Path = r'C:\Users\steve\Downloads\machine-learning-ex2\ex2\ex2data2.txt'
Path = r'/Users/ternencekk/Downloads/machine-learning-ex-withoutanswer/ex2/ex2data2.txt'
data2 = pd.read_csv(Path, header=None, names=['Test1','Test2','Accepted'])
# print(data2.head())

# 筛选出y=0 y=1的结果行
neg = data2[data2.Accepted == 0]
pos = data2[data2.Accepted == 1]


# 图形表达
plt.scatter(pos.Test1,pos.Test2,c='black',marker='+',label='y=1',s=50)
plt.scatter(neg.Test1,neg.Test2,c='yellow',marker='o',label='y=0',s=50)
plt.xlabel('Microchip Test1')
plt.ylabel('Microchip Test2')
plt.legend()
plt.show()


def mapFeature(X1,X2):
    degree = 7
    out = pd.DataFrame(np.ones(len(X1)),columns=['Ones'])
    for i in range(1,degree):
        for j in range(i+1):
            out['F'+str(i)+str(j)] = np.power(X1,i-j)*np.power(X2,j)
    return out

X1 = data2['Test1']
X2 = data2['Test2']
y = data2.iloc[:,-1:]
X = mapFeature(X1,X2)


def sigmoid(z):
    return 1/(1+np.exp(-z))

def costFunctionReg(theta,X,y,lam=1):
    z = sigmoid(X@theta)
    J = sum(np.multiply(-y,np.log(z)) - np.multiply((y-1),np.log(1 - z)))/len(z)+lam/2/len(z)*np.sum(np.power(theta,2))
    return J

def gradient(theta,X,y,lam):
    grad = np.zeros(len(theta))
    error = sigmoid(X@theta)-y
    parameters = int(theta.ravel().shape[0])  # 计算需求解的参数个数
    for i in range(parameters):
        # print(np.multiply((z-y),X[:,i]))
        grad[i] = np.sum(error*X[:,i])/len(X)    +   lam/len(X)*theta[i]
    return grad


initial_theta = np.zeros((X.shape[1],1))
# initial_theta = initial_theta.values
X = X.values
y = y.values
J = costFunctionReg(initial_theta,X,y,1)
grad = gradient(initial_theta,X,y,1)
# print(J)
# print(grad)
# print(grad.shape)

# zzzz = np.zeros(3)
# print(zzzz.shape)
# print(b.shape)

print('theta\n',initial_theta.shape)
print('Xshape\n',X.shape)
print('yshape\n',y.shape)

# ValueError: tnc: invalid return value from minimized function.
result= opt.fmin_tnc(func=costFunctionReg,x0=initial_theta,fprime=gradient,args=(X,y,1))
print(result)

opt.f
# model = linear_model.LogisticRegression(penalty='l2',C=1.0)
# model.fit(X,y.ravel())
# rr = model.score(X,y)
# print(rr)
#
# x = np.linspace(-1, 1.5, 250)
# xx, yy = np.meshgrid(x, x)
#
# z = mapFeature(xx.ravel(), yy.ravel())
# z = z @ b
# z = z.values
# z = z.reshape(xx.shape)
#
# print(xx.ravel())
# print(xx.shape)
# # print(z)
# # print(xx)
# # print(yy)
#
#
# plt.contour(xx, yy, z, 100)
# # plt.ylim(-.8, 1.2)
# plt.show()

r = np.linspace(-1, 1.5, 250)
r1, r2 = np.meshgrid(r, r)

df = X@grad
df = df.reshape(r1.shape)
print(df)
print(df.shape)

def plot_decision_boundary(pred_func):
    # 设定最大最小值，附加一点点边缘填充

    print(X.shape)
    x_min, x_max = X[:, 0:-1].min() - .5, X[:, 0:-1].max() + .5
    y_min, y_max = X[:, -1].min() - .5, X[:, -1].max() + .5
    h = 0.01

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # 用预测函数预测一下
    # np.c_ 矩阵横向相加
    # np.c_ [1,2] [5,6]   result  [1,2,5,6]
    #       [3,4] [7,8]           [3,4,7,8]
    # .ravel多维变一维
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 然后画出图
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(df1[:, 0], df1[:, -1], c=y, cmap=plt.cm.Spectral)




# 咱们先来瞄一眼逻辑斯特回归对于它的分类效果
# clf = linear_model.LogisticRegressionCV()
# clf.fit(X, y)
#
# # 画一下决策边界
# plot_decision_boundary(lambda X: clf.predict(X))
# plt.title("Logistic Regression")
# plt.show()
