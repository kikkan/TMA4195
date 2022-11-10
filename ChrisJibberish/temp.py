import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, hstack


offset = np.array([-1, 0, 1])

d1 = diags([1, 2, 1], offset, (4, 4))
d2 = diags([2, 1, 2], offset, (4, 4))
d3 = np.hstack((d1.toarray(), d2.toarray()))
# print(d3)

# %% sparse stack
d3s = hstack([d1, d2])
d3sArr = d3s.toarray()
print(d3sArr)


# %% stacked and sparse stacked matrix mult
n1 = d3s.shape[1]
print(n1)
v = np.array([1 for _ in range(n1)])
m = d3s@v
print(m)
