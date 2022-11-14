import scipy.sparse as sp
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import inv, spsolve
import matplotlib.pyplot as plt
import sys
#%%
def construct_firstandlast_layer(Nx, Ny, Nz, a, b, c):
    #first layer
    diagsA1u = [a*np.ones(Nx), (1-2*a-b-c)*np.ones(Nx+1), a*np.ones(Nx)] #upper/north row of first layer, equal to lower/south row
    offset = [-1, 0, 1]
    A1u = diags(diagsA1u, offset).tolil() 
    A1u[0,0] = 1-a-b-c #corner
    A1u[-1,-1] = 1-a-b-c #corner
    
    diagsAb = b*np.ones(Nx+1) #diagonal alpha y matrix
    Ab = diags(diagsAb)
    
    diagsA1d = [a*np.ones(Nx), (1-2*a-2*b-c)*np.ones(Nx+1), a*np.ones(Nx)] #inner rows of first layer
    A1d = diags(diagsA1d, offset).tolil()
    A1d[0,0] = 1-a-2*b-c #side
    A1d[-1,-1] = 1-a-2*b-c #side
    
    A0 = sp.csr_matrix(np.zeros((Nx+1, Nx+1)))                #0 matrix
    #assembling first layer
    r1 = sp.hstack((A1u, Ab, sp.hstack([A0]*(Ny+1-2))))      #first row A1u Aa 0 0 0
    r2 = sp.hstack((Ab, A1d, Ab, sp.hstack([A0]*(Ny+1-3))))  #second row
    rm2 = sp.hstack((sp.hstack([A0]*(Ny+1-3)), Ab, A1d, Ab)) #second to last row
    rm1 = sp.hstack((sp.hstack([A0]*(Ny+1-2)), Ab, A1u))     #last row
    rows = 0                                            
    for j in range(2, Ny-1):                                 #assembling intermediate rows
        left = sp.hstack([A0]*(j-1))
        right = sp.hstack([A0]*(Ny+1-j-2))
        row = sp.hstack((left, Ab, A1d, Ab, right))
        if j == 2:
            rows = row
            continue
        rows = sp.vstack((rows, row))
    return sp.vstack((r1, r2, rows, rm2, rm1))

def construct_intermediate_layer(Nx, Ny, Nz, a, b, c):
    #intermediate layers
    diagsA1u = [a*np.ones(Nx), (1-2*a-b-2*c)*np.ones(Nx+1), a*np.ones(Nx)] #upper/north row, equal to lower/south row
    offset = [-1, 0, 1]
    A1u = diags(diagsA1u, offset).tolil() 
    A1u[0,0] = 1-a-b-2*c #corner
    A1u[-1,-1] = 1-a-b-2*c #corner
    
    diagsAb = b*np.ones(Nx+1) #diagonal alpha matrix
    Ab = diags(diagsAb)
    
    diagsA1d = [a*np.ones(Nx), (1-2*a-2*b-2*c)*np.ones(Nx+1), a*np.ones(Nx)] #inner rows of intermediate layer
    A1d = diags(diagsA1d, offset).tolil()
    A1d[0,0] = 1-a-2*b-2*c #side
    A1d[-1,-1] = 1-a-2*b-2*c #side
    
    A0 = sp.csr_matrix(np.zeros((Nx+1, Nx+1)))                #0 matrix
    #assembling intermediate layer
    r1 = sp.hstack((A1u, Ab, sp.hstack([A0]*(Ny+1-2))))      #first row of matrices
    r2 = sp.hstack((Ab, A1d, Ab, sp.hstack([A0]*(Ny+1-3))))  #second row of matrices
    rm2 = sp.hstack((sp.hstack([A0]*(Ny+1-3)), Ab, A1d, Ab)) #second to last row of matrices
    rm1 = sp.hstack((sp.hstack([A0]*(Ny+1-2)), Ab, A1u))     #last row of matrices
    rows = 0                                            
    for j in range(2, Ny-1):                                 #assembling intermediate rows of matrices
        left = sp.hstack([A0]*(j-1))
        right = sp.hstack([A0]*(Ny+1-j-2))
        row = sp.hstack((left, Ab, A1d, Ab, right))
        if j == 2:
            rows = row
            continue
        rows = sp.vstack((rows, row))
    return sp.vstack((r1, r2, rows, rm2, rm1))

def construct_3D_coefficient_matrix(Nx, Ny, Nz, ax, ay, az):
    Afirst = construct_firstandlast_layer(Nx, Ny, Nz, ax, ay, az)
    Aintermediate = construct_intermediate_layer(Nx, Ny, Nz, ax, ay, az)
    diagsAc = az*np.ones((Nx+1)*(Ny+1)) #diagonal alpha z matrix
    Ac = diags(diagsAc)
    A0 = sp.csr_matrix(((Nx+1)*(Ny+1), (Nx+1)*(Ny+1)))
    # diagsA0 = np.zeros((Nx+1)*(Ny+1)) #diagonal alpha y matrix
    # A0 = diags(diagsA0)
    #assembling
    l1 = sp.hstack((Afirst, Ac, sp.hstack([A0]*(Nz+1-2))))                 #first layer
    l2 = sp.hstack((Ac, Aintermediate, Ac, sp.hstack([A0]*(Nz+1-3))))      #second layer
    lm2= sp.hstack((sp.hstack([A0]*(Nz+1-3)), Ac, Aintermediate, Ac))      #second to last layer
    lm1= sp.hstack((sp.hstack([A0]*(Nz+1-2)), Ac, Afirst))                 #last layer
    layers = 0
    for k in range(2, Nz-1):
        left = sp.hstack([A0]*(k-1))
        right = sp.hstack([A0]*(Nz+1-k-2))
        layer = sp.hstack((left, Ac, Aintermediate, Ac, right))
        if k == 2:
            layers = layer
            continue
        layers = sp.vstack((layers, layer))
    return sp.vstack((l1, l2, layers, lm2, lm1))

def EFD3D(A, n0, r0, b0, nx, ny, nz, dt, ts):
    n = np.zeros((ts, (nx+1)*(ny+1)*(nz+1)))
    r = np.zeros((ts, (nx+1)*(ny+1)*(nz+1)))
    b = np.zeros((ts, (nx+1)*(ny+1)*(nz+1)))
    n[0,:], r[0,:], b[0,:] = n0, r0, b0
    dtf = np.zeros(nx+1)
    for t in range(ts-1):
        if t % 100 == 0: print(t/(ts-1))
        dtf = dt*f(n[t, :], r[t, :], b[t, :])
        n[t+1,:] = A@n[t,:] + dtf
        r[t+1,:] = r[t,:]   + dtf
        b[t+1,:] = b[t,:]   - dtf
        if np.min(n[t+1,:]) < 0:
            print("Help! The concentration of neurotransmitters is negative at some index!")
            sys.exit()
        if np.min(n[t+1,:]) < 0:
            print("Help! The concentration of free receptors is negative at some index!")
            sys.exit()
        if np.min(n[t+1,:]) < 0:
            print("Help! The concentration of bound receptors is negative at some index!")
            sys.exit()
    return n, r, b

def CN3D(A, B, n0, r0, b0, nx, ny, nz, dt, ts):
    """Performs CN in 2D using matrix multiplication.
    Args:
        A (csr_matrix): Block matrix A (sparse)
        B (csr_matrix): Block matrix A (sparse)
        n0 (ndarray): Initital distribution of neurotransm. concentration
        r0 (ndarray): Initial distr. of receptor concentration
        b0 (ndarray): Init. distr. of bounded receptor concentration
        ts (int, optional): Timesteps. Defaults to 1000.
    Returns:
        (n, r, b): Concentrations of n, r, b for all timesteps
    """
    # TODO Maybe make A,B inside this function
    n = np.zeros((ts, (nx+1)*(ny+1)*(nz+1)))
    r = np.zeros((ts, (nx+1)*(ny+1)*(nz+1)))
    b = np.zeros((ts, (nx+1)*(ny+1)*(nz+1)))
    n[0, :], r[0, :], b[0, :] = n0, r0, b0
    # Precompute
    A = sp.csc_matrix(A)
    Ainv = inv(A)
    AinvB = Ainv@B
    Ainvdtf = np.zeros(nx+1)
    for t in range(ts-1):
        if t % 100 == 0: print(t/(ts-1))
        # dtf = dt*f(n[t, :], r[t, :], b[t, :])
        # n[t+1, :] = spsolve(A, B@n[t, :] + dtf)
        # r[t+1, :] = r[t, :] + dtf
        # b[t+1, :] = b[t, :] - dtf
        
        Ainvdtf = Ainv@(dt*f(n[t, :], r[t, :], b[t, :]))
        n[t+1, :] = AinvB@n[t, :] + Ainvdtf
        r[t+1, :] = r[t, :] + Ainvdtf
        b[t+1, :] = b[t, :] - Ainvdtf
        
        if np.min(n[t+1,:]) < 0: print("Help! The concentration of neurotransmitters is negative at some index!")
        if np.min(n[t+1,:]) < 0: print("Help! The concentration of free receptors is negative at some index!")
        if np.min(n[t+1,:]) < 0: print("Help! The concentration of bound receptors is negative at some index!")
    return n, r, b

def f(n, r, b):
    return -k1*n*r + km1*b

def plot_3D(Nx, Ny, C):
    x = np.linspace(0, 1, Nx+1)
    y = np.linspace(0, 1, Ny+1)
    x, y = np.meshgrid(x, y)
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    # ax.axes.set_zlim3d(0, 1)
    z = C.reshape((Nx+1, Ny+1))
    ax.plot_surface(x, y, z, cmap="viridis")
    plt.show()
    
def plot_lineconcentration(x, y, Lz, Nx, Ny, Nz, C):
    c_list = np.zeros(Nz)
    for i in range(Nz):
        c_list[i] = C[(y*(Nx+1)+x)+i*(Nx+1)*(Ny+1)]
    plt.figure()
    plt.plot(np.linspace(0, Lz, Nz), c_list)
    plt.show()
#%% Initialization
Lx, Ly, Lz = 390.*1e-9, 390.*1e-9, 15.*1e-9 #Length in x, y, z direction
Dx, Dy, Dz = 8*1e-7, 8*1e-7, 8*1e-7         #Diffusion constants in every direction
Nx, Ny, Nz = 300, 300, 10                   #N+1 is the amount of grid points in every direction
dx, dy, dz = Lx/Nx, Ly/Ny, Lz/Nz            #Distance between gridpoints in each direction
dt = dx**2 * 1e5   #Multiply dt to be as big as possible while keeping sx+sy+sz        #Stepsize in time
ax = Dx * dt/(dx)**2                #alpha x direction
ay = Dy * dt/(dy)**2                #alpha y direction
az = Dz * dt/(dz)**2                #alpha z direction
if ax+ay+az >= 0.5: 
    print("alpha_x + alpha_y + alpha_z =", ax+ay+az , "Hey, stop, this will crash")
    sys.exit()
    
k1 = 4*1e6
km1 = 5

K = (Nx+1)*(Ny+1)*(Nz+1)            #Amount of grid points
cFR0 = 1e3*4*0.22**2/((Nx+1)*(Ny+1))
Uv = np.zeros(K)                   
n0 = Uv.copy()
r0 = Uv.copy() + cFR0
b0 = Uv.copy()
n0[int((Nx+1)*(Ny+1)/2)] = 5000
#%% EFD3D constructing A matrix
A = construct_3D_coefficient_matrix(Nx=Nx, Ny=Ny, Nz=Nz, ax=ax, ay=ay, az=az)
#%% EFD3D Run
timesteps = 1000
n, r, b = EFD3D(A, n0, r0, b0, Nx, Ny, Nz, dt, ts=timesteps)
#%% Plotting EFD Lineconcentration
for t in range(0, timesteps, int(timesteps/5)):
    # Plotting the concentration of neurotransmitters on the line x = Nx/2, y = Ny/2, z = 0 to z = Nz
    print(np.sum(n[t,:]))
    print(np.sum(n[t,:])+np.sum(b[t,:]),"\n")
    
    plot_lineconcentration(x=int(Nx/2), y=int(Ny/2), Lz=Lz, Nx=Nx, Ny=Ny, Nz=Nz, C=n[t,:])
#%% Plotting EFD layer concentration
for t in range(0, timesteps, int(timesteps/10)):
    plot_3D(Nx, Ny, n[t,:(Nx+1)*(Ny+1)])  #Plotting concentration of neurtransmitters in the first layer
    # plot_3D(Nx, Ny, r[t,(Nx+1)*(Ny+1)*Nz:]) #Plotting concentration of free receptors in the final layer
    plot_lineconcentration(x=int(Nx/2), y=int(Ny/2), Lz=Lz, Nx=Nx, Ny=Ny, Nz=Nz, C=n[t,:])
    #^Plotting the concentration of neurotransmitters on the line x = Nx/2, y = Ny/2, z = 0 to z = Nz
    print("NT:                      {}\n"
          "NT and bound:            {}\n"
          "Bound and free:          {}\n"
          "NT and free and 2xbound: {}\n"
          "".format(np.sum(n[t, :]),
                    np.sum(n[t, :]) + np.sum(b[t, :]),
                    np.sum(b[t, :]) + np.sum(r[t, :]),
                    np.sum(n[t, :]) + np.sum(r[t, :]) + 2*np.sum(b[t, :])))
#%% If we have too many timesteps, it's not feasible to save all the values all the time, the following is just an adhoc, easy way to
# plot for large amount of timesteps
# To run this code snippet, you first need to run the blocks Initialize and EFD3D constructing A matrix, this then runs as you expect
#
# Or you could uncomment the following values to see a nice plot have a nice plot :)
# Lx = Ly = Lz = 1.
# Dx = Dy = Dz = 0.1
# Nx = Ny = Nz = 100
# dx = dy = dz = Lx/Nx
# dt = dx**2
# ax = ay = az = Dx * dt/(dx)**2
# if ax+ay+az >= 0.5: 
#     print("alpha_x + alpha_y + alpha_z =", ax+ay+az , "Hey, stop, this will crash")
#     sys.exit()
# k1 = 0.1
# km1 = 0.01

# K = (Nx+1)*(Ny+1)*(Nz+1)            #Amount of grid points
# cFR0 = 1e3*4*0.22**2/((Nx+1)*(Ny+1))
# Uv = np.zeros(K)                   
# n0 = Uv.copy()
# r0 = Uv.copy() + cFR0
# b0 = Uv.copy()
# n0[int((Nx+1)*(Ny+1)/2)] = 5000

n, r, b = n0, r0, b0
dtf = np.zeros(Nx+1)
timesteps = 10000
for t in range(timesteps-1):
    if t % 100 == 0: print(t/(timesteps-1))
    dtf = dt*f(n, r, b)
    n = A@n + dtf
    r = r   + dtf
    b = b   - dtf
    if np.min(n) < 0:
        print("Help! The concentration of neurotransmitters is negative at some index!")
        sys.exit()
    if np.min(r) < 0:
        print("Help! The concentration of free receptors is negative at some index!")
        sys.exit()
    if np.min(b) < 0:
        print("Help! The concentration of bound receptors is negative at some index!")
        sys.exit()
    if t % 100 == 0:
        # plot_3D(Nx, Ny, n[:(Nx+1)*(Ny+1)])  #Plotting concentration of neurtransmitters in the first layer
        plot_3D(Nx, Ny, r[(Nx+1)*(Ny+1)*Nz:]) #Plotting concentration of free receptors in the final layer
        # plot_lineconcentration(x=int(Nx/2), y=int(Ny/2), Lz=Lz, Nx=Nx, Ny=Ny, Nz=Nz, C=n[t,:])
        #^Plotting the concentration of neurotransmitters on the line x = Nx/2, y = Ny/2, z = 0 to z = Nz

#%% CN 3D This doesnt work, or at least it takes ages to invert the A matrix
# # %% CN 3D Constructing matrices
# #The A matrix in CN is the same as the A in FD, only with 1+sigma on the diagonal, instead of 1-alpha 
# #sigma in CN is equal to half the alpha used in FD, so one can input ax=-1/2*ax and get the correct matrix
# A = construct_3D_coefficient_matrix(Nx=Nx, Ny=Ny, Nz=Nz, sx=-1/2*ax, sy=-1/2*ay, sz=-1/2*az)
# #The B matrix in CN is the same as the A in FD, with 
# B = construct_3D_coefficient_matrix(Nx=Nx, Ny=Ny, Nz=Nz, sx=ax, sy=-sy, sz=-sz)
# #%% CN 3D 
# timesteps = 100
# n, r, b = CN3D(A, B, n0, r0, b0, Nx, Ny, Nz, dt, ts=timesteps)
# #%% CN 3D plotting the concentration of neurotransmitters on the line x = Nx/2, y = Ny/2, z = 0 to z = Nz
# for t in range(0, timesteps, int(timesteps/100)):
#     print(np.sum(n[t,:]))
#     print(np.sum(n[t,:])+np.sum(b[t,:]),"\n")
    
#     plot_lineconcentration(x=int(Nx/2), y=int(Ny/2), Lz=Lz, Nx=Nx, Ny=Ny, Nz=Nz, C=n[t,:])


