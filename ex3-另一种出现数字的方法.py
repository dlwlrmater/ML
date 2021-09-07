# https://blog.csdn.net/weixin_41509677/article/details/105192457

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat


# data= loadmat(r"C:\Users\steve\Downloads\machine-learning-ex3\ex3\ex3data1.mat") #mat格式转换为dict字典
data = loadmat(r'/Users/ternencekk/Downloads/machine-learning-ex-withoutanswer/ex3/ex3data1.mat')
# data

raw_X = data['X']
# print(raw_X[1])
# print(len(raw_X[1]))
def image(X):
    pick_one = np.random.randint(5000)  # 返回一个随机整型数
    image = X[pick_one]

    fig, ax = plt.subplots()
    ax.imshow(image.reshape(20, 20).T)
    plt.show()

image(raw_X)
