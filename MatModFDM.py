import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
#%%
def plot_step(x, cNT, cFR, cBR):
    plt.figure()
    # plt.ylim(0, 1)
    plt.xlabel(r"$x$")
    plt.ylabel("Concentration")
    plt.plot(x, cNT, label="Neurotransmitters")
    plt.plot(x, cFR, drawstyle="steps-pre", label="Free Receptors")
    plt.plot(x, cBR, label="Bound Receptors")
    plt.legend()
#%% 1D
L = 1
D = 0.4
k1 = 0.1
km1 = 0.01
n = 100
dx = L/n
dt = dx**2
alpha = D * dt/(dx)**2 #D * dt/(dx)**2
print("Alpha:", alpha)

k = [alpha * np.ones(n), (1-2*alpha) * np.ones(n+1), alpha * np.ones(n)] #Constructing the coefficient matrix A
offset = [-1, 0, 1]
A = diags(k, offset).toarray()
A[0, 0] = 1-alpha
A[-1, -1] = 1-alpha

x = np.linspace(0, 1, n+1)
cFR = np.zeros(n+1) #Free receptors
cFR[x>=0.5] = .5
cBR = np.zeros(n+1) #Bound receptors
cNT = np.zeros(n+1) #Neurotransmitters
cNT[0] = 100.

plot_step(x, cNT, cFR, cBR)
timesteps = int(0.2/dt)
for i in range(timesteps):
    cNT0 = np.copy(cNT) #need to make temp variables since they are updated in one after another, but should use the original value
    cFR0 = np.copy(cFR)
    cBR0 = np.copy(cBR)
    cNT = A@cNT0 - k1 * cNT0*cFR0 + km1 * cBR0
    cFR = cFR0   - k1 * cNT0*cFR0 + km1 * cBR0
    cBR = cBR0   + k1 * cNT0*cFR0 - km1 * cBR0
    if i%10 == 0:
        print(i/timesteps)
        plot_step(x, cNT, cFR, cBR)
        
#%% 1D Experiment: A + B <-> C
def plot_step(x, A, B, C):
    plt.figure()
    # plt.ylim(0, 1)
    plt.xlabel(r"$x$")
    plt.ylabel("Concentration")
    plt.plot(x, A, label="A")
    plt.plot(x, B, label="B")
    plt.plot(x, C, label="C")
    plt.legend()
#%% 1D Experiment: A + B <-> C
L = 1
D = 0.4
k1 = 0.1
km1 = 0.01
n = 100
dx = L/n
dt = dx**2
alpha = D * dt/(dx)**2 #D * dt/(dx)**2
print("Alpha:", alpha)

k = [alpha * np.ones(n), (1-2*alpha) * np.ones(n+1), alpha * np.ones(n)] #Constructing the coefficient matrix A
offset = [-1, 0, 1]
A = diags(k, offset).toarray()
A[0, 0] = 1-alpha
A[-1, -1] = 1-alpha

x = np.linspace(0, 1, n+1)
cA = np.zeros(n+1) #Concentration of A
cB = np.zeros(n+1) #Concentration of B
cC = np.zeros(n+1) #Concentration of C
cA[0] = 100.
cB[-1] = 100.

plot_step(x, cA, cB, cC)
print(sum(cA)+sum(cB)+sum(cC))
timesteps = int(0.2/dt)
for i in range(timesteps):
    cA = A@cA - k1 * cA*cB + km1 * cC
    cB = A@cB - k1 * cA*cB + km1 * cC
    cC = cC   + k1 * cA*cB - km1 * cC
    if i%10 == 0:
        print(i/timesteps)
        print(sum(cA)+sum(cB)+2*sum(cC))
        plot_step(x, cA, cB, cC)
#%% 2D plot functions
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

    
#%% 2D
L = 1
D = 0.2
k1 = 0.1
km1 = 0.01
n = 100 # m=n, x and y same discretization
dx = L/n
dt = dx**2
alpha = D * dt/(dx)**2 #D * dt/(dx)**2

K = (n+1)**2 # Need N*M nodes
k = [alpha * np.ones(K-(n+1)), alpha * np.ones(K-1), (1-4*alpha) * np.ones(K),
     alpha * np.ones(K-1), alpha * np.ones(K-(n+1))] #Constructing the coefficient matrix A
offset = [-n-1, -1, 0, 1, n+1]
A = diags(k, offset).toarray()
#Fixing Neumann Boundary Conditions
A[0, 0] = 1-2*alpha                     #"First and last" corner
A[-1, -1] = 1-2*alpha
for i in range(1, n):                   #"North" and "South" boundary conditions
    A[i, i] = 1-3*alpha
    A[-i-1, -i-1] = 1-3*alpha
for i in range(1, n):                   #"No wrap around" boundary conditions
    A[(n+1)*i, (n+1)*i] = 1-3*alpha     #West
    A[(n+1)*i, (n+1)*i-1] = 0
    A[(n+1)*i+n, (n+1)*i+n] = 1-3*alpha #East
    A[(n+1)*i+n, (n+1)*i+n+1] = 0
A[n, n] = 1-2*alpha                     #Two remaining corners
A[n, n+1] = 0
A[(n+1)*n, (n+1)*n] = 1-2*alpha
A[(n+1)*n, (n+1)*n-1] = 0

cFR = np.zeros(K) #Free receptors
cFR[int((n+1)**2/2):] = .5 
cBR = np.zeros(K) #Bound receptors
cNT = np.zeros(K) #Neurotransmitters
cNT[int((n+1)/2)] = 5000.
plot_3D(n, cNT)
# plot_heatmap(n, cNT)
timesteps = int(0.2/dt)
for i in range(timesteps):
    cNT0 = np.copy(cNT) #need to make temp variables since they are updated in one after another, but should use the original value
    cFR0 = np.copy(cFR)
    cBR0 = np.copy(cBR)
    cNT = A@cNT0 - k1 * cNT0*cFR0 + km1 * cBR0
    cFR = cFR0   - k1 * cNT0*cFR0 + km1 * cBR0
    cBR = cBR0   + k1 * cNT0*cFR0 - km1 * cBR0
    if i%100 == 0:
        print(i/timesteps)
        plot_3D(n, cNT)
        # plot_3D(n, cBR)
        # print(i, sum(cBR))
        # plot_heatmap(n, cNT)
