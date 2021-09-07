import numpy as np
import matplotlib.pyplot as plt



x = np.array([0,1,2,3])
y = np.array([0,1,2,3])

X,Y = np.meshgrid(x,x)
a = np.meshgrid(x,x)
print(a)

# print(X)
# print(Y)


def f(x, y):
    return 2*x+y

plt.contour(X,Y,f(X,Y),levels=20)
plt.show()

plt.plot(X, Y,color='red',  # 全部点设置为红色
         marker='.',  # 点的形状为圆点
         linestyle='')  # 线型为空，也即点与点之间不用线连接

plt.grid(True)



# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
#
# # 计算x,y坐标对应的高度值
# def f(x, y):
#     return (1 - x / 2 + x ** 3 + y ** 5) * np.exp(-x ** 2 - y ** 2)
#
#
# # 生成x,y的数据
# n = 512
# x = np.linspace(-10, 10, n)
# y = np.linspace(-10, 10, n)
#
# # 把x,y数据生成mesh网格状的数据，因为等高线的显示是在网格的基础上添加上高度值
# X, Y = np.meshgrid(x, y)
#
# # 填充等高线
# plt.contourf(X, Y, f(X, Y),levels=10)
# # 显示图表
# plt.show()