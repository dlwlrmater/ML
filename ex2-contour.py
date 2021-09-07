import numpy as np
import matplotlib.pyplot as plt

x = np.array([[0,1,2,3],[1,2,3,4],[2,3,4,5],[3,4,5,6]])
print(x)
plt.contourf(x)
plt.show()
