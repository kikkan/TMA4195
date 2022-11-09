import scipy.sparse as sp
import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt
#%% Hello friends!
def construct_coefficient_matrix(n, alpha):
    diagsAu = [alpha*np.ones(n), (1-3*alpha)*np.ones(n+1), alpha*np.ones(n)]
    offset = [-1, 0, 1]
    Au = diags(diagsAu, offset).tolil()
    Au[0,0] = 1-2*alpha
    Au[-1,-1] = 1-2*alpha
    
    diagsAm = alpha*np.ones(n+1)
    Am = diags(diagsAm)
    
    diagsAd = [alpha*np.ones(n), (1-4*alpha)*np.ones(n+1), alpha*np.ones(n)]
    Ad = diags(diagsAd, offset).tolil()
    Ad[0,0] = 1-3*alpha
    Ad[-1,-1] = 1-3*alpha
    
    A0 = sp.csr_matrix(np.zeros((n+1, n+1)))
    
    r1 = sp.hstack((Au, Am, sp.hstack([A0]*(n+1-2))))
    r2 = sp.hstack((Am, Ad, Am, sp.hstack([A0]*(n+1-3))))
    rm2 = sp.hstack((sp.hstack([A0]*(n+1-3)), Am, Ad, Am))
    rm1 = sp.hstack((sp.hstack([A0]*(n+1-2)), Am, Au))
    rows = 0
    for j in range(2, n-1):
        left = sp.hstack([A0]*(j-1))
        right = sp.hstack([A0]*(n+1-j-2))
        row = sp.hstack((left, Am, Ad, Am, right))
        if j == 2:
            rows = row
            continue
        rows = sp.vstack((rows, row))
    
    return sp.vstack((r1, r2, rows, rm2, rm1))
#%%
def plot_heatmap(n, C):
    p = C.reshape((n+1, n+1))
    plt.imshow(p, cmap="hot")#, interpolation="nearest")
    plt.show()
def plot_3D(n, C):
    x = np.linspace(0, 1, n+1)
    y = np.linspace(0, 1, n+1)
    x, y = np.meshgrid(x, y)
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.axes.set_zlim3d(0, 1)
    z = C.reshape((n+1, n+1))
    ax.plot_surface(x, y, z, cmap="viridis")
    plt.show()    
#%%
L = 1
D = 0.1
k1 = 0.1
km1 = 0.01
n = 100 # m=n, x and y same discretization
K = (n+1)**2
dx = L/n
dt = dx**2
alpha = D * dt/(dx)**2 #D * dt/(dx)**2
A = construct_coefficient_matrix(n=n, alpha=alpha)

cFR = np.zeros(K) #Free receptors
cFR[int((n+1)**2/2):] = .5 
cBR = np.zeros(K) #Bound receptors
cNT = np.zeros(K) #Neurotransmitters
cNT[int((n+1)/2)] = 5000.
plot_3D(n, cNT)
# plot_heatmap(n, cNT)
timesteps = int(0.5/dt)
for i in range(timesteps):
    cNT0 = np.copy(cNT) #need to make temp variables since they are updated in one after another, but should use the original value
    cFR0 = np.copy(cFR)
    cBR0 = np.copy(cBR)
    cNT = A@cNT0 - k1 * cNT0*cFR0 + km1 * cBR0
    cFR = cFR0   - k1 * cNT0*cFR0 + km1 * cBR0
    cBR = cBR0   + k1 * cNT0*cFR0 - km1 * cBR0
    if i%100 == 0:
        print(i/timesteps, sum(cNT)+sum(cFR)+2*sum(cBR)) #sum of n, r, 2*b should be constant
        plot_3D(n, cNT)
