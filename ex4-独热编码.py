from sklearn import preprocessing
import numpy as np
enc = preprocessing.OneHotEncoder()
df = np.mat([[0, 0, 5], [1, 1, 0], [0, 2, 1], [1, 0, 2],[1,0,4]])
print(df)
enc.fit(df)    # fit来学习编码
a = enc.transform([[0, 1, 5]]).toarray()    # 进行编码
print(a)