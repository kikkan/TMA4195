import scipy.sparse as sp
import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt
#%%3D
def construct_firstandlast_layer(n, alpha):
    #first layer
    diagsA1u = [alpha*np.ones(n), (1-4*alpha)*np.ones(n+1), alpha*np.ones(n)] #upper/north row of first layer, equal to lower/south row
    offset = [-1, 0, 1]
    A1u = diags(diagsA1u, offset).tolil() 
    A1u[0,0] = 1-3*alpha #corner
    A1u[-1,-1] = 1-3*alpha #corner
    
    diagsAa = alpha*np.ones(n+1) #diagonal alpha matrix
    Aa = diags(diagsAa)
    
    diagsA1d = [alpha*np.ones(n), (1-5*alpha)*np.ones(n+1), alpha*np.ones(n)] #inner rows of first layer
    A1d = diags(diagsA1d, offset).tolil()
    A1d[0,0] = 1-4*alpha #side
    A1d[-1,-1] = 1-4*alpha #side
    
    A0 = sp.csr_matrix(np.zeros((n+1, n+1)))                #0 matrix
    #assembling first layer
    r1 = sp.hstack((A1u, Aa, sp.hstack([A0]*(n+1-2))))      #first row A1u Aa 0 0 0
    r2 = sp.hstack((Aa, A1d, Aa, sp.hstack([A0]*(n+1-3))))  #second row
    rm2 = sp.hstack((sp.hstack([A0]*(n+1-3)), Aa, A1d, Aa)) #second to last row
    rm1 = sp.hstack((sp.hstack([A0]*(n+1-2)), Aa, A1u))     #last row
    rows = 0                                            
    for j in range(2, n-1):                                 #assembling intermediate rows
        left = sp.hstack([A0]*(j-1))
        right = sp.hstack([A0]*(n+1-j-2))
        row = sp.hstack((left, Aa, A1d, Aa, right))
        if j == 2:
            rows = row
            continue
        rows = sp.vstack((rows, row))
    return sp.vstack((r1, r2, rows, rm2, rm1))

def construct_intermediate_layer(n, alpha):
    #intermediate layers
    diagsA1u = [alpha*np.ones(n), (1-5*alpha)*np.ones(n+1), alpha*np.ones(n)] #upper/north row, equal to lower/south row
    offset = [-1, 0, 1]
    A1u = diags(diagsA1u, offset).tolil() 
    A1u[0,0] = 1-4*alpha #corner
    A1u[-1,-1] = 1-4*alpha #corner
    
    diagsAa = alpha*np.ones(n+1) #diagonal alpha matrix
    Aa = diags(diagsAa)
    
    diagsA1d = [alpha*np.ones(n), (1-6*alpha)*np.ones(n+1), alpha*np.ones(n)] #inner rows of intermediate layer
    A1d = diags(diagsA1d, offset).tolil()
    A1d[0,0] = 1-5*alpha #side
    A1d[-1,-1] = 1-5*alpha #side
    
    A0 = sp.csr_matrix(np.zeros((n+1, n+1)))                #0 matrix
    #assembling intermediate layer
    r1 = sp.hstack((A1u, Aa, sp.hstack([A0]*(n+1-2))))      #first row of matrices
    r2 = sp.hstack((Aa, A1d, Aa, sp.hstack([A0]*(n+1-3))))  #second row of matrices
    rm2 = sp.hstack((sp.hstack([A0]*(n+1-3)), Aa, A1d, Aa)) #second to last row of matrices
    rm1 = sp.hstack((sp.hstack([A0]*(n+1-2)), Aa, A1u))     #last row of matrices
    rows = 0                                            
    for j in range(2, n-1):                                 #assembling intermediate rows of matrices
        left = sp.hstack([A0]*(j-1))
        right = sp.hstack([A0]*(n+1-j-2))
        row = sp.hstack((left, Aa, A1d, Aa, right))
        if j == 2:
            rows = row
            continue
        rows = sp.vstack((rows, row))
    return sp.vstack((r1, r2, rows, rm2, rm1))

def construct_3D_coefficient_matrix(n, alpha):
    Afirst = construct_firstandlast_layer(n, alpha)
    Aintermediate = construct_intermediate_layer(n, alpha)
    diagsAa = alpha*np.ones((n+1)**2) #diagonal alpha matrix
    Aa = diags(diagsAa)
    A0 = sp.csr_matrix(np.zeros(((n+1)**2, (n+1)**2)))
    #assembling
    l1 = sp.hstack((Afirst, Aa, sp.hstack([A0]*(n+1-2))))                 #first layer
    l2 = sp.hstack((Aa, Aintermediate, Aa, sp.hstack([A0]*(n+1-3))))             #second layer
    lm2= sp.hstack((sp.hstack([A0]*(n+1-3)), Aa, Aintermediate, Aa))             #second to last layer
    lm1= sp.hstack((sp.hstack([A0]*(n+1-2)), Aa, Afirst))                 #last layer
    layers = 0
    for k in range(2, n-1):
        left = sp.hstack([A0]*(k-1))
        right = sp.hstack([A0]*(n+1-k-2))
        layer = sp.hstack((left, Aa, Aintermediate, Aa, right))
        if k == 2:
            layers = layer
            continue
        layers = sp.vstack((layers, layer))
    return sp.vstack((l1, l2, layers, lm2, lm1))

def plot_3D(n, C):
    x = np.linspace(0, 1, n+1)
    y = np.linspace(0, 1, n+1)
    x, y = np.meshgrid(x, y)
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    # ax.axes.set_zlim3d(0, 1)
    z = C.reshape((n+1, n+1))
    ax.plot_surface(x, y, z, cmap="viridis")
    plt.show()
#%%3D
L = 1.
D = 0.1
k1 = 0.1
km1 = 0.01
n = 100 # m=n, x and y same discretization
K = (n+1)**3
dx = L/n
dt = dx**2
alpha = D * dt/(dx)**2 #D * dt/(dx)**2. D*dt*(1/(dx)**2+1/(dy)**2+1/(dz)**2) should be less than 1/2
A = construct_3D_coefficient_matrix(n=n, alpha=alpha)

cFR = np.zeros(K) #Free receptors
cFR[100*(n+1)**2:] = 150./10201 #putting free receptors only in final layer
cBR = np.zeros(K) #Bound receptors
cNT = np.zeros(K) #Neurotransmitters
cNT[int((n+1)**2/2)] = 5000.
plot_3D(n, cFR[100*(n+1)**2:])
timesteps = int(0.5/dt)
for i in range(timesteps):
    cNT0 = np.copy(cNT) #need to make temp variables since they are updated in one after another, but should use the original value
    cFR0 = np.copy(cFR)
    cBR0 = np.copy(cBR)
    cNT = A@cNT0 - k1 * cNT0*cFR0 + km1 * cBR0
    cFR = cFR0   - k1 * cNT0*cFR0 + km1 * cBR0
    cBR = cBR0   + k1 * cNT0*cFR0 - km1 * cBR0
    if i%100 == 0:
        print(i/timesteps, sum(cFR[100*(n+1)**2:])) #plotting concentration of free receptors in final layer
        plot_3D(n, cFR[100*(n+1)**2:])
        # print(i/timesteps, sum(cNT[50*(n+1)**2:51*(n+1)**2])) #plotting concentration of neurotransmitters in starting layer
        # plot_3D(n, cNT[50*(n+1)**2:51*(n+1)**2])
