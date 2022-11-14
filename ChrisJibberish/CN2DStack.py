import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve, inv
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from matplotlib import cm
from tqdm import tqdm
import numpy as np
from alive_progress import alive_bar


# %% TODO
"""
It is now working somewhat whith some artifacts prolly due to inaccuracy of the 
CN and way to big steps etc. TODO:
- Make function that computes max timestep.
    Found by considering mult of uniformly distr NT times R sometin?!   
- Make finer grid and test.
- Using progress bar somehow fcks up the animation plots on my computer, so that
    I can only plot one concentration at each run. Left it there cuz it's cool.
"""


# %% Options
np.set_printoptions(linewidth=160)

# %% Functions


def makeA2D(nx: int, ny: int, sigma: float):
    """Make block matrix A and b. (First make A = dia(A_d), then add A_u and A_l
       to A. Furthermore, correct each diagonal block boundaries and corners in 
       interior of A. Lastly fix two points.) (Made under assumption that rowsum
       of A,B must be 1.)

    Args:
        nx (int): # points in x
        ny (int): # points in y
        sigma (float): Coefficient (dt*D/(2dxdy))

    Returns:
        A, B: Block matrix A and B
    """

    # A (nxny,nxny) (Using A_d for the entire block first, then correct)
    offset = [-nx, -1, 0, 1, nx]  # Offset includes A_+-1
    valsA = [-sigma, -sigma, 1+4*sigma, -sigma, -sigma]
    valsB = [sigma, sigma, 1-4*sigma, sigma, sigma]
    A_dia = diags(valsA, offset, (nx*ny, nx*ny))  # guess this is right shape
    B_dia = diags(valsB, offset, (nx*ny, nx*ny))  # guess this is right shape

    # A_u (nx,nx) (Corrects upper left and lover right block of A)
    offsetU = [-1, 0, 1]
    valsUA = [-sigma, 1+3*sigma, -sigma]
    valsUB = [sigma, 1-3*sigma, sigma]
    Au_dia = diags(valsUA, offsetU, (nx, nx))
    Bu_dia = diags(valsUB, offsetU, (nx, nx))

    # Convert to modifiable matrix type and modify
    A = csr_matrix(A_dia)
    Au = csr_matrix(Au_dia)
    B = csr_matrix(B_dia)
    Bu = csr_matrix(Bu_dia)

    # Correct corners
    Au[0, 0] = 1+2*sigma
    Au[nx-1, nx-1] = 1+2*sigma
    Bu[0, 0] = 1-2*sigma
    Bu[nx-1, nx-1] = 1-2*sigma

    # Fix left/right and corner of each block except A_u and A_l
    for i in range(nx, nx*ny-ny, nx):
        A[i, i-1] = 0  # Left of grid
        A[i, i] = 1+3*sigma  # First diag (boundary)
        A[i+nx-1, i+nx-1] = 1+3*sigma  # Last diag (boundary)
        A[i+nx-1, i+nx] = 0  # Right of grid
        # same for B
        B[i, i-1] = 0  # Left of grid
        B[i, i] = 1-3*sigma  # First diag
        B[i+nx-1, i+nx-1] = 1-3*sigma  # Last diag
        B[i+nx-1, i+nx] = 0  # Right of grid

    A[:nx, :ny] = Au  # add A_u block
    A[(nx**2-nx):, (nx**2-nx):] = Au  # Add A_l block
    B[:nx, :ny] = Bu  # add A_u block
    B[(nx**2-nx):, (nx**2-nx):] = Bu  # Add A_l block

    # Fix point to the right/left of top right/top left point
    A[nx-1, nx] = 0  # Fix one fucker
    A[nx**2-nx, nx**2-nx-1] = 0  # Fix another fucker
    B[nx-1, nx] = 0  # Fix one fucker
    B[nx**2-nx, nx**2-nx-1] = 0  # Fix another fucker

    return A, B


def CN2D(A: csr_matrix, B: csr_matrix, n0: np.ndarray, r0: np.ndarray, b0: np.ndarray, ts=1000):
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

    # n = r = b = np.zeros((ts, nx*ny))  # matrices for all timesteps
    n = np.zeros((ts, nx*ny))
    r = np.zeros((ts, nx*ny))
    b = np.zeros((ts, nx*ny))
    n[0, :], r[0, :], b[0, :] = n0, r0, b0
    temp = np.sum(n[0, :])
    # Precompute
    Ainv = inv(A)
    AinvB = Ainv@B
    temp = np.zeros(nx)
    dtf = np.zeros(nx)
    with alive_bar(ts-1) as bar:
        for t in (range(ts-1)):
            # n[t+1, :] = AinvB@n[t, :]  # only diffusion
            # n[t+1, :] = spsolve(A, B@n[t, :])  # Only diffusion
            #     n[t+1, :] = spsolve(A, B@n[t, :] + dt*f(n[t, :], r[t, :], b[t, :]))
            #     r[t+1, :] = dt*f(n[t, :], r[t, :], b[t, :])
            #     b[t+1, :] = -dt*f(n[t, :], r[t, :], b[t, :])
            # print(t)
            dtf = Ainv@(dt*f(n[t, :], r[t, :], b[t, :]))
            n[t+1, :] = AinvB@n[t, :] + dtf
            r[t+1, :] = r[t, :] + dtf
            b[t+1, :] = b[t, :] - dtf
            bar()
    return n, r, b, temp


# %% 2D plot functions
def plot_heatmap(n, C):
    p = C.reshape((n+1, n+1))
    plt.imshow(p, cmap="hot")  # , interpolation="nearest")
    plt.show()


def plot_3D(nx, ny, C):
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    x, y = np.meshgrid(x, y)
    fig = plt.figure()
    ax = fig.gca(projection="3d")
#     ax.axes.set_zlim3d(0, 1)
    z = C.reshape((nx, ny))
    ax.plot_surface(x, y, z, cmap="viridis")
    plt.show()


# %% Units
height = 15e-9  # Height of cleft (15e-9)
radius = 220e-9  # Radius of cleft/membrane (220e-9)
D = 8e-7  # Diffusion coefficient (8e-7)
k1 = 4e6  # Forward reaction coefficient (4e6)
km1 = 5  # Backward reaction coefficient (5)
# r0 = 152  # receptors on membrane (1000/(1e-6)^2 * 2*pi*(220e-9))
r0 = 193.6  # Square area
n0 = 5000  # Neurotransmitters per vesicle pop

# Units in micro
# height = 15e-3  # Height of cleft (15e-9)
# radius = 220e-3  # Radius of cleft/membrane (220e-9)
# D = 8e-7*1e12  # Diffusion coefficient (8e-7)
# k1 = 4e6  # Forward reaction coefficient (4e6)
# km1 = 5  # Backward reaction coefficient (5)
# # r0 = 152  # receptors on membrane (1000/(1e-6)^2 * 2*pi*(220e-9))
# r0 = 192  # Square area
# n0 = 5000  # Neurotransmitters per vesicle pop

# %% Discretization
nx = 51  # Discretization in x
ny = 51  # Dicretization in y
dx = 2*radius/nx
dy = 2*radius/ny
dt = dx**2*1e6  # (dx*dy?)  # TODO scale
# dx = 0.1
# dy = 0.1
# dt = 1e-8

# %% scaling
# D = 0.01
kappa = D*dt  # TODO scale?
# sigma = 1
sigma = kappa / (2*dx*dy)

# %% make grid and init state
ma = 4*radius**2  # membrane area
cFR0 = 1e3*4*0.22**2/(nx*ny)  # r0/ma  # concentration of free receptors at t0 (1e-3)
Uv = np.zeros(nx*ny)
n0 = Uv.copy()  # Concentration of neurotransmitters. pop not allocated.
r0 = Uv.copy() + cFR0  # uniform concentration
b0 = Uv.copy()  # concentration of bounded receptors
n0[int(nx*ny/2)] = 5000


def f(n, r, b):
    # t = -k1*n*r + km1*b
    # t = np.where
    # return max(-k1*n*r + km1*b, -0.4)
    # TODO make max n*r and zero if not too slow
    return -k1*n*r + km1*b


timesteps = int(1000)

A, B = makeA2D(nx, ny, sigma)
# plt.spy(A)
# plt.show()
# plt.spy(B)
n, r, b, fvals = CN2D(A, B, n0, r0, b0, timesteps)

# %% debug plot
# for t in range(0, 7):
#     # plot_3D(nx, ny, n[t, :])
#     print('NT and bound:            {}\n'
#           'bound and free:          {}\n'
#           'NT and free and 2xBound: {}\n\n'
#           ''.format(np.sum(n[t, :])+np.sum(b[t, :]),
#                     np.sum(b[t, :]) + np.sum(r[t, :]),
#                     np.sum(n[t, :]) + np.sum(r[t, :]) + 2*np.sum(b[t, :])))

# print and plot vals
# print("steplength in time:", dt, "\n", "#"*50)
# for t in range(0, timesteps, int(timesteps/5)):
#     plot_3D(nx, ny, n[t, :])
#     plot_3D(nx, ny, r[t, :])
#     plot_3D(nx, ny, b[t, :])
#     print('NT                       {}\n'
#           'NT and bound:            {}\n'
#           'bound and free:          {}\n'
#           'NT and free and 2xBound: {}\n\n'
#           ''.format(np.sum(n[t, :]),
#                     np.sum(n[t, :])+np.sum(b[t, :]),
#                     np.sum(b[t, :]) + np.sum(r[t, :]),
#                     np.sum(n[t, :]) + np.sum(r[t, :]) + 2*np.sum(b[t, :])))

# plot(fvals)


def rowSum(a):
    pass
    l = a.shape[0]
    rs = np.zeros(l)
    ierror = []
    for i in range(l):
        if 0.9>np.sum(a[i, :]) or np.sum(a[i, :])>1.1:
            ierror.append(i)
    return ierror


# aiErr = rowSum(A)
# biErr = rowSum(B)
# print(aiErr)
# print(biErr)
# print(A[22, :])


# x = np.arange(0, nx*ny, 1)
# for t in range(0, timesteps, int(timesteps/5)):
#     plt.plot(x, n[t, :])
# print(A.toarray())
# print(B.toarray())

# %% 3D plot
def update(t, c, line, zlim):
    """updates 3D animation

    Args:
        t (int): Current timestep
        c (np.array): Concentration of substance
        line (?): line to be updated in 3D
    """
    # line.set_data()
    # line.set_3d_properties(c[t, :].reshape(nx, ny))

    ax.clear()
    z = c[t, :].reshape(nx, ny)
    plotset(z, zlim)
    plt.title("t = {}/{}".format(t, timesteps))
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1,
                           cmap=cm.coolwarm, linewidth=0, antialiased=False,
                           alpha=0.7)
    return surf,


def plotset(z, zlim):
    ax.set_xlim3d(0., 1.)
    ax.set_ylim3d(0., 1.)
    if zlim == None:
        ax.set_zlim3d(0, np.max(z))
    else:
        ax.set_zlim3d(0, zlim)
    ax.set_zlim3d(0, np.max(z))
    ax.set_autoscalez_on(False)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    cset = ax.contour(x, y, z, zdir='x', offset=0., cmap=cm.coolwarm)
    cset = ax.contour(x, y, z, zdir='y', offset=1., cmap=cm.coolwarm)
    cset = ax.contour(x, y, z, zdir='z', offset=-1., cmap=cm.coolwarm)


def plot3D(c, zlim=None, title = None):
    global x, y, z
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    x, y = np.meshgrid(x, y)
    c0 = c[0, :]
    z = c0.reshape((nx, ny))

    fig = plt.figure()
    global ax
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim3d(0., 1.)
    ax.set_ylim3d(0., 1.)
    if zlim == None:
        ax.set_zlim3d(0, np.max(c))
    else:
        ax.set_zlim3d(0, zlim)
    
    if title: 
        plt.title(title)

    cset = ax.contour(x, y, z, zdir='x', offset=0., cmap=cm.coolwarm)
    cset = ax.contour(x, y, z, zdir='y', offset=1., cmap=cm.coolwarm)
    cset = ax.contour(x, y, z, zdir='z', offset=-1., cmap=cm.coolwarm)

    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False, alpha=0.7)

    ani = animation.FuncAnimation(fig, update, fargs=(c, surf, zlim), frames=timesteps, interval=30, blit=False)
    plt.show()
    plt.close()


# plot3D(n, title = "Neurotransmitter concentration")
# plot3D(r, 1, title = "Receptor concentration")
# plot3D(b, 1, title = "Bound receptor concentration")

n_r = np.sum(r, axis = 1)
n_b = np.sum(b, axis = 1)
P = n_b[1:]/n_r[1:]  
  
print(np.min(P))
print(np.min(np.argwhere(P >= 0.5)))

plt.plot(n_r)
plt.plot(n_b)
plt.show()

plt.plot(P)
plt.show()
