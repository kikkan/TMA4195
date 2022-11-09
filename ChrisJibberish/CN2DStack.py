import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags

# %% Functions


# %% 2D
L = 1  # Length of cleft
D = 0.2  # Diffusion coefficient
k1 = 0.1  # Forward reaction coefficient
km1 = 0.01  # Backward reaction coefficient
n = 100  # m=n, x and y same discretization
dx = L/n
dt = dx**2
alpha = D * dt/(dx)**2  # D * dt/(dx)**2

K = (n+1)**2  # Need N*M nodes
k = [alpha * np.ones(K-(n+1)), alpha * np.ones(K-1), (1-4*alpha) * np.ones(K),
     alpha * np.ones(K-1), alpha * np.ones(K-(n+1))]  # Constructing the coefficient matrix A
offset = [-n-1, -1, 0, 1, n+1]
A = diags(k, offset).toarray()
