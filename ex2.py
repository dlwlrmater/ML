import pandas as pd
import numpy as np
import types
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.linear_model import LogisticRegressionCV

# Path = r'/Users/ternencekk/Downloads/machine-learning-ex-withoutanswer/ex2/ex2data1.txt'
Path = r'C:\Users\dell\OneDrive\machine-learning-ex-withoutanswer\ex2\ex2data1.txt'
df = pd.read_csv(Path,header=None,names=['scor1','scor2','judge'])
print(df)
df1 = df.copy()
pos = df[df.judge == 1]
neg = df[df.judge == 0]

fig,ax = plt.subplots(figsize=(12,8))
# s=size  c=color b=blue  o
ax.scatter(pos.scor1,pos.scor2,s=100,c='black',marker='+',label='Admitted')
ax.scatter(neg.scor1,neg.scor2,s=100,c='y',marker='o',label='Not admitted')
plt.legend()
# plt.show()

# 假如a0  位置 name values
df.insert(0,'ones',1)
cols = df.shape[1]
x = df.iloc[:,:cols-1]
y = df.iloc[:,cols-1:cols]
x = np.mat(x.values)
y = np.mat(y.values)
# print('x1\n',x)
# print(x.shape)

theta = np.mat(np.zeros((3,1)))


def sigmoid(X,theta):
    # 不知道为何R3*100 和 R100*1出来不是R3*1
    z = np.dot(X,theta).T
    sig = 1/(1+np.exp(-z))
    return sig

# print(sigmoid(x,theta).shape)

def computecost(theta,X,y):    # 计算J
    A = sigmoid(X,theta)
    inner1 = np.multiply(-y,np.log(A))
    inner2 = np.multiply(y-1,np.log(1-A))
    inner = inner1+inner2
    return sum(inner)/y.shape[0]

def gradientdescent(theta,X,y):   # 更新grad矩阵
    grad = np.zeros(len(theta))
    error = sigmoid(X,theta)-y
    for i in range(len(theta)):
        grad[i] = np.sum(error.T*x[:,i])/len(X)
    return grad

# print('theta\n',theta.shape)
# print(theta.shape)
# print('xshape\n',x.shape)
# print('yshape\n',y.shape)

# https://blog.csdn.net/weixin_30797199/article/details/95163586  有关fmin_tnc和minimize的区别

result = opt.fmin_tnc(func=computecost,x0=theta,fprime=gradientdescent,args=(x,y))
# print('result\n',result)

grad = result[0]
def predict(X,theta):
    re = sigmoid(X,theta)
    return [1 if i >=0.5 else 0 for i in re]

# 计算预测值和实际值的匹配率
y_pre = np.array(predict(x,grad))
# print('y_pre 1\n',y_pre)
y_pre = y_pre.reshape(len(y_pre),1)
# print('y_pre 2\n',y_pre)
acc = np.mean(y_pre==y)
# print(acc)


# 把dataframe变成np.array
x = df1.iloc[:,:2].values
# print(x)
# print('=========')
y = df1.iloc[:,-1].values


# 由于使用了sklearn库 logisticRegression自动得到desicion boundary
def plot_decision_boundary(pred_func):
    # print(x)
    # 设定最大最小值，附加一点点边缘填充
    x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
    # print(x_min,x_max)
    y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
    # print(y_min,y_max)
    h = 0.01

    # 根据(x_min,x_max)和(y_min,y_max)做矩阵 通过矩阵画网格
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # 用预测函数预测一下
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    # print(xx.ravel())
    # print(xx.ravel())
    # print(Z)
    # print('-----------')
    Z = Z.reshape(xx.shape)
    # print(Z)

    # 然后画出图
    # xx x轴 , yy y轴 Z等高线函数
    plt.contour(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Spectral)



# 咱们先来瞄一眼逻辑斯特回归对于它的分类效果
# LogisticRegressionCV使用了交叉验证来选择正则化系数C。而LogisticRegression需要自己每次指定一个正则化系数
clf = LogisticRegressionCV()
# print(x)
rr = clf.fit(x, y)

# 画一下决策边界
plot_decision_boundary(lambda x: clf.predict(x))
plt.title("Logistic Regression")
plt.show()

