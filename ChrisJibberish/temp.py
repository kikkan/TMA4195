import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, hstack
# A = coo_matrix([[1, 2], [3, 4]])
# B = coo_matrix([[5, 6]])
# S = vstack([A, B]).toarray()
# print(S)

offset = np.array([-1, 0, 1])

d1 = diags([1, 2, 1], offset, (4, 4))
d2 = diags([2, 1, 2], offset, (4, 4))
d3 = np.hstack((d1.toarray(), d2.toarray()))
# print(d3)

# %% sparse stack
d3s = hstack([d1, d2])
d3sArr = d3s.toarray()
print(d3sArr)
