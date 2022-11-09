import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, hstack

# %% Functions


# %% 2D
# Units
L = 1  # Length of cleft
D = 0.2  # Diffusion coefficient
k1 = 0.1  # Forward reaction coefficient (4e6)
km1 = 0.01  # Backward reaction coefficient (5)
rBar = 152  # receptors on membrane (1000/(1e-6)^2 * 2*pi*(220e-9))
nBar = 5000  # Neurotransmitters per vesicle pop

# Discretization
n = 100  # m=n, x and y same discretization
dx = L/n
dt = dx**2
alpha = D * dt/(dx)**2  # D * dt/(dx)**2

# Sparse matrices
K = (n+1)**2  # Need N*M nodes
k = [alpha * np.ones(K-(n+1)), alpha * np.ones(K-1), (1-4*alpha) * np.ones(K),
     alpha * np.ones(K-1), alpha * np.ones(K-(n+1))]  # Constructing the coefficient matrix A
offset = [-n-1, -1, 0, 1, n+1]
A = diags(k, offset).toarray()

offsetA = [-1, 0, 1]  # Offset for tridiag matrix
Avals = []
# TODO offset for block matrix lower and upper. simply diags on -n and n respectively.

# TODO implement upper and lower boundary. What to do on sides of cleft?
