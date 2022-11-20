import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, csr_matrix, csc_matrix
from scipy.sparse.linalg import spsolve, inv
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from matplotlib import cm
from tqdm import tqdm
import numpy as np
from alive_progress import alive_bar
import warnings


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
# warnings.filterwarnings("ignore")

# %% run options
run         = 1
showAni     = 0
printAns    = 0
saveFigs    = 0
matrixDebug = 0
fluxFindRun = 0
findTime    = 0
runNoFlux   = 0


# %% Functions
def f(n, r, b):
    # TODO make max n*r and zero if not too slow
    return -k1*n*r + km1*b


def abc(x0, y0, x, r):
    c = y0**2 + (x-x0)**2 - r**2
    b = -2*y0
    nom1 = -b + np.sqrt(b**2-4*c)
    nom2 = -b - np.sqrt(b**2-4*c)
    denom = 2
    return np.array([nom1, nom2])/denom


def discDistr(nx, ny, loc, r, fluxTot):
    m = np.zeros((nx, ny))
    ind = []
    for i in range(nx):
        for j in range(ny):
            if (i-loc[0])**2 + (j-loc[1])**2 <= r**2:
                m[i, j] = (r**2 - ((i-loc[0])**2 + (j-loc[1])**2))
                ind.append([i, j])
    # print(np.array(ind))
    ind = np.array(ind)
    # m[ind[:, 0], ind[:, 1]] = fluxTot/len(ind[:, 0])
    # print(round(m, 2))
    m = m/np.sum(m)*fluxTot
    return m.reshape(nx*ny)


def makeA_2DNeyman(nx: int, ny: int, sigma: float, drch=1):
    """Make block matrix A. (First make A = dia(A_d), then add A_u and A_l
       to A. Furthermore, correct each diagonal block boundaries and corners in 
       interior of A. Lastly fix two points.) (Made under assumption that rowsum
       of A,B must be 1.)

    Args:
        nx (int): # points in x
        ny (int): # points in y
        sigma (float): Coefficient (dt*D/(2dxdy)) Negative for B matrix.

    Returns:
        A: Block matrix A (or B)
    """

    # A (nxny,nxny) (Using A_d for the entire block first, then correct)
    offset = [-nx, -1, 0, 1, nx]  # Offset includes A_+-1
    valsA = [-sigma, -sigma, 1+4*sigma, -sigma, -sigma]
    A_dia = diags(valsA, offset, (nx*ny, nx*ny))  # guess this is right shape

    # A_u (nx,nx) (Corrects upper left and lover right block of A)
    offsetU = [-1, 0, 1]
    valsUA = [-sigma, 1+3*sigma, -sigma]
    Au_dia = diags(valsUA, offsetU, (nx, nx))

    # Convert to modifiable matrix type and modify
    A = csc_matrix(A_dia)
    Au = csc_matrix(Au_dia)

    # Correct corners
    Au[0, 0] = 1+2*sigma
    Au[nx-1, nx-1] = 1+2*sigma

    # Fix left/right and corner of each block except A_u and A_l
    for i in range(nx, nx*ny-ny, nx):
        A[i, i-1] = 0  # Left of grid
        A[i, i] = 1+3*sigma  # First diag (boundary)
        A[i+nx-1, i+nx-1] = 1+3*sigma  # Last diag (boundary)
        A[i+nx-1, i+nx] = 0  # Right of grid

    A[:nx, :ny] = Au  # add A_u block
    A[(nx**2-nx):, (nx**2-nx):] = Au  # Add A_l block

    # Fix point to the right/left of top right/top left point
    A[nx-1, nx] = 0  # Fix one fucker
    A[nx**2-nx, nx**2-nx-1] = 0  # Fix another fucker

    return A


def makeA_2DDirichlet(nx: int, ny: int, sigma: float):
    """Make block matrix A. (First make A = dia(A_d), then add A_u and A_l
       to A. Furthermore, correct each diagonal block boundaries and corners in 
       interior of A. Lastly fix two points.) (Made under assumption that rowsum
       of A,B must be 1.)

    Args:
        nx (int): # points in x
        ny (int): # points in y
        sigma (float): Coefficient (dt*D/(2dxdy)) Negative for B matrix.

    Returns:
        A: Block matrix A (or B)
    """

    # A (nxny,nxny) (Using A_d for the entire block first, then correct)
    offset = [-nx, -1, 0, 1, nx]  # Offset includes A_+-1
    valsA = [-sigma, -sigma, 1+4*sigma, -sigma, -sigma]
    A_dia = diags(valsA, offset, (nx*ny, nx*ny))  # guess this is right shape

    # Convert to modifiable matrix type and modify
    A = csc_matrix(A_dia)
    # A[0, 0] = 0
    # A[nx*ny-1, nx*ny-1] = 0

    # Fix block to the right/left of upper left/lower right
    # xi = np.array([i for i in range(nx)])
    # indices = np.array([i for i in range(nx)])
    # yi = np.array([nx+i for i in range(nx)])
    # A[indices, indices+nx] = 0
    # A[nx*(ny-1) + indices, nx*(ny-2) + indices] = 0

    for i in range(0, ny*(nx-1), nx):
        A[i, i-1] = 0  # Left of grid
        A[i+nx-1, i+nx] = 0  # Right of grid

    # A[nx-1, nx] = 0
    # A[nx**2-nx, nx**2-nx-1] = 0

    return A

# old docstring
# """Performs CN in 2D using matrix multiplication of sparse matrices.

# Args:
#     A (csr_matrix): Block matrix A (sparse)
#     B (csr_matrix): Block matrix A (sparse)
#     n0 (ndarray): Initital distribution of neurotransm. concentration
#     r0 (ndarray): Initial distr. of receptor concentration
#     b0 (ndarray): Init. distr. of bounded receptor concentration
#     ts (int, optional): Timesteps.
#     drch (int, optional): Dirichlet. Have zero conc. on boundaries.
#     kwargs (dictionary): can take specific n0, in flux, and disc location and radius

# Returns:
#     (n, r, b): Concentrations of n, r, b for all timesteps
# """


# def CN2D(A: csr_matrix, B: csr_matrix, n0: np.ndarray, r0: np.ndarray, b0: np.ndarray, ts: int, flux: int, nFlux=0):
def CN2D(A: csr_matrix, B: csr_matrix, ts: int, **kwargs):
    """Performs CN in 2D using matrix multiplication of sparse matrices.

    Args:
        A (csr_matrix): Block matrix A (sparse)
        B (csr_matrix): Block matrix A (sparse)
        ts (int, optional): Timesteps.
        kwargs (dictionary): can take specific n0, in flux, disc location and radius

    Returns:
        (n, r, b): Concentrations of n, r, b for all timesteps
    """

    # matrices for all timesteps
    n = np.zeros((ts, nx*ny))
    r = np.zeros((ts, nx*ny))
    b = np.zeros((ts, nx*ny))
    # n0 = discDistr(nx, ny, [nx/2, ny/2], 0.2*nx, 5000)
    # n[0, :], r[0, :], b[0, :] = n0, r0, b0

    rDisc = 0.8*nx/2
    loc = [nx/2, ny/2]  # center
    flux = 200
    for key, item in kwargs.items():
        match key:
            case 'n0':
                n[0, :] = discDistr(nx, ny, loc, rDisc, 5000)
            case 'radius':
                rDisc = item
            case 'location':
                loc = item
            case 'flux':
                flux = item
                print(flux)

    print('\n\nRun details:\n'
          'Grid (nx, ny):   {}\n'
          'Timesteps:       {}\n'
          'Total NT/step:   {}\n'
          'Total NT         {}\n'
          ''.format((nx, ny), ts, flux, flux*ts)
          )

    # init values
    m = discDistr(nx, ny, loc, rDisc, flux)
    # mSum = np.sum(m)
    # n[0, :] = m
    r[0, :] = 1e3*4*0.22**2/(nx*ny)  # uniform conc of receptors on entire grid

    # Precompute/allocate memory
    Ainv  = inv(A)
    AinvB = Ainv@B
    dtf   = np.zeros(nx*ny)

    with alive_bar(ts-1) as bar:
        for t in (range(ts-1)):
            dtf = Ainv@(dt*f(n[t, :], r[t, :], b[t, :]))
            n[t+1, :] = AinvB@n[t, :] + dtf +m
            r[t+1, :] = r[t, :] + dtf
            b[t+1, :] = b[t, :] - dtf
            n[t, [0, nx-1]] = 0
            for i in range(nx, nx*ny-nx, nx):
                # can make diag identity for this maybe
                n[t, [i, i+nx-1]] = 0
            bar()
    # for t in (range(ts-1)):
    #     dtf = Ainv@(dt*f(n[t, :], r[t, :], b[t, :]))
    #     n[t+1, :] = AinvB@n[t, :] + dtf +m
    #     r[t+1, :] = r[t, :] + dtf
    #     b[t+1, :] = b[t, :] - dtf
    #     n[t, [0, nx-1]] = 0
    #     for i in range(nx, nx*ny-nx, nx):
    #         # can make diag identity for this maybe
    #         n[t, [i, i+nx-1]] = 0
    return n, r, b


# %% 3D animate functions
def update(t, c, line, zlim):
    """updates 3D animation

    Args:
        t (int): Current timestep
        c (np.array): Concentration of substance
        line (?): line to be updated in 3D
    """
    ax.clear()
    z = c[t, :].reshape(nx, ny)
    plotset(z, zlim)
    plt.title("t = {}/{}".format(t, timesteps))
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1,
                           cmap='viridis', linewidth=0, antialiased=False,
                           alpha=0.7)
    return surf,


def plotset(z, zlim):
    ax.set_xlim3d(0., 1.)
    ax.set_ylim3d(0., 1.)
    if zlim == None:
        ax.set_zlim3d(0, np.max(z))
    else:
        ax.set_zlim3d(0, zlim)
    # ax.set_zlim3d(0, np.max(z))
    ax.set_autoscalez_on(False)
    # plt.ylabel('Y AXIS')
    # plt.xlabel('X AXIS')
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    cset = ax.contour(x, y, z, zdir='x', offset=0., cmap='viridis')
    cset = ax.contour(x, y, z, zdir='y', offset=1., cmap='viridis')
    cset = ax.contour(x, y, z, zdir='z', offset=-1., cmap='viridis')


def animate3D(c, zlim=None):
    global x, y, z
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    x, y = np.meshgrid(x, y)
    c0 = c[0, :]
    z = c0.reshape((nx, ny))

    fig = plt.figure()
    global ax
    ax = fig.add_subplot(111, projection='3d')
    plt.ylabel('y')
    plt.xlabel('x')
    ax.set_xlim3d(0., 1.)
    ax.set_ylim3d(0., 1.)
    if zlim == None:
        ax.set_zlim3d(0, np.max(c))
    else:
        ax.set_zlim3d(0, zlim)

    cset = ax.contour(x, y, z, zdir='x', offset=0., cmap='viridis')
    cset = ax.contour(x, y, z, zdir='y', offset=1., cmap='viridis')
    cset = ax.contour(x, y, z, zdir='z', offset=-1., cmap='viridis')

    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis',
                           linewidth=0, antialiased=False, alpha=0.7)

    ani = animation.FuncAnimation(fig, update, fargs=(c, surf, zlim), frames=timesteps, interval=1, blit=False)
    plt.show()


# %% Plot functions for figures to be saved
def plot_3D(Nx, Ny, ct, g=0, save=0, fn=None, title=0):
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    x, y = np.meshgrid(x, y)
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    if g:
        diff = ct.max() - ct.min()
        ax.axes.set_zlim3d(ct.min() - g*diff, ct.max() + g*diff)
    z = ct.reshape((Nx, Ny))
    ax.plot_surface(x, y, z, cmap="viridis")
    if title:
        plt.title(title)
    if save:
        plt.savefig('figures\\'+fn+'.pdf')
    else:
        plt.show()


def plotConcProgress(*args, **kwargs):
    for c in args:
        plt.plot(c)
    for k, v in kwargs.items():
        match k:
            case 'labels':
                plt.legend(v)
            case 'save':
                plt.savefig('figures\\'+v+'.pdf')
    plt.show()


# %% Units
height = 15e-9  # Height of cleft (15e-9)
radius = 220e-9  # Radius of cleft/membrane (220e-9)
D      = 8e-7  # Diffusion coefficient (8e-7)
k1     = 4e6  # Forward reaction coefficient (4e6)
km1    = 5  # Backward reaction coefficient (5)
# r0     = 152  # receptors on membrane (1000/(1e-6)^2 * 2*pi*(220e-9))
r0     = 192  # Square area
n0     = 500  # Neurotransmitters per vesicle pop

# Units in micro
# height = 15e-3  # Height of cleft (15e-9)
# radius = 220e-3  # Radius of cleft/membrane (220e-9)
# D = 8e-7*1e12  # Diffusion coefficient (8e-7)
# k1 = 4e6  # Forward reaction coefficient (4e6)
# km1 = 5  # Backward reaction coefficient (5)
# # r0 = 152  # receptors on membrane (1000/(1e-6)^2 * 2*pi*(220e-9))
# r0 = 192  # Square area
# n0 = 5000  # Neurotransmitters per vesicle pop

# %% run single
# Discretization
# nx = 51  # Discretization in x
# ny = 51  # Dicretization in y
# dx = 2*radius/nx
# dy = 2*radius/ny
# dt = dx**2*1e6  # (dx*dy?)  # TODO scale


# ################# scaling
# # D = 0.01
# kappa = D*dt  # TODO scale?
# # sigma = 1
# sigma = kappa / (2*dx*dy)

# make grid and init state
# ma   = 4*radius**2  # membrane area
# cFR0 = 1e3*4*0.22**2/(nx*ny)  # r0/ma  # concentration of free receptors at t0 (1e-3)
# Uv   = np.zeros(nx*ny)
# cn0   = Uv.copy()  # Concentration of neurotransmitters. pop not allocated.
# cr0   = Uv.copy() + cFR0  # uniform concentration
# cb0   = Uv.copy()  # concentration of bounded receptors
# cn0[int(nx*ny/2)] = n0


# timesteps = int(500)

# A = makeA_2DNeyman(nx, ny, sigma)
# B = makeA_2DNeyman(nx, ny, -sigma)
# A = makeA_2DDirichlet(nx, ny, sigma)
# B = makeA_2DDirichlet(nx, ny, -sigma)

# debug matrix
if matrixDebug:
    plt.spy(A)
    plt.grid()
    plt.show()
    # plt.spy(B)


timesteps = int(500)
nx = 21  # Discretization in x
ny = 21  # Dicretization in y
dx = 2*radius/nx
dy = 2*radius/ny
dt = dx**2*1e6  # (dx*dy?)  # TODO scale


# scaling
# D = 0.01
kappa = D*dt  # TODO scale?
# sigma = 1
sigma = kappa / (2*dx*dy)

rDisc = 0.7*nx/2
loc = [nx/2, nx/2]  # defaults to center
flux = 200

A = makeA_2DDirichlet(nx, ny, sigma)
B = makeA_2DDirichlet(nx, ny, -sigma)

if run:
    # n, r, b = CN2D(
    #     A, B, cr0, cb0, timesteps,
    #     radius=radius, location=loc, flux=flux
    # )
    # n, r, b = CN2D(
    #     A, B, timesteps,
    #     radius=rDisc, location=loc, flux=flux
    # )
    n, r, b = CN2D(
        A, B, timesteps,
        radius=rDisc, location=loc, flux=0, n0=1
    )


# %% temp debug leakage
if run:
    # animate3D(b)
    animate3D(n, zlim=1)
    # plot_3D(nx, ny, n[0])
    plotConcProgress(np.sum(n, axis=1))
    plotConcProgress(np.sum(b, axis=1))

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


if showAni:
    animate3D(n)
    # animate3D(r, 1)
    # animate3D(b, 1)

    # plt.plot(bFrac)
    # plt.show()

# %% Critical times (Answers)
if printAns:  # print and compute misc answers
    print(
        'run informatio:\n'
        'Discretization:\n'
        '   (nx, ny, ts) = ({},{}, {})\n'
        '   (dx, dy, dt) = ({:.3e},{:.3e}, {:.3e})\n'
        ''.format(
            nx, ny, timesteps,
            dx, dy, dt
        )
    )

    nTot = np.sum(n, axis=1)
    rTot = np.sum(r, axis=1)
    bTot = np.sum(b, axis=1)
    ts = np.arange(timesteps)
    plt.plot(ts, bTot)
    plt.plot(ts, rTot)
    plt.hlines(y=r0/2, xmin=0, xmax=timesteps)
    plt.show()

    signal = np.argmax(bTot > r0/2)
    print('When is signal transmitted?\n'
          'Values:\n'
          'index {}\n'
          'time  {}'.format(signal, signal*dt))


# %% Save figures
if saveFigs:
    i = 100
    plot_3D(nx, ny, n[i, :], 0.2, 1, 'CN2D_'+str(i))
    plot_3D(nx, ny, n[signal, :], 0.2, 1, "CN2D_Signal")
    plot_3D(nx, ny, n[-1, :], 0.2, 1, "CN2D_last")

    plotConcProgress(rTot, bTot, labels=['r', 'b'], save="CN2D_conc_r&b")
    plotConcProgress(
        rTot, bTot, nTot-n0+r0,
        labels=[r'$r$', r'$b$', r'$n-n_0+r_0$'],
        save="CN2D_conc_nr&b",
        linestyle=['-', '-', 'dashed']
    )

# %% Find flux
# radius = 220e-9  # Radius of cleft/membrane (220e-9)

# discretization
nx = 31  # Discretization in x
ny = 31  # Dicretization in y
dx = 2*radius/nx
dy = 2*radius/ny
dt = dx**2*1e8  # (dx*dy?)  # TODO scale

timesteps = int(500)

# fluxs[0] = 10

# nTots = {}
# rTots = {}
# bTots = {}
ns = {}
rs = {}
bs = {}

nTemp = np.zeros((timesteps, nx**2))
rTemp = np.zeros((timesteps, nx**2))
bTemp = np.zeros((timesteps, nx**2))

rDisc = nx/2
loc = [nx/2, nx/2]  # defaults to center
# loc = [nx/2 + 0.25*nx, nx/2 + 0.25*nx]  # defaults to center
# flux = 200
fluxs = np.arange(0.1, 1.1, 0.2)

if fluxFindRun:
    A = makeA_2DDirichlet(nx, ny, sigma)
    B = makeA_2DDirichlet(nx, ny, -sigma)
    runs = {}
    for fl in fluxs:
        nTemp, rTemp, bTemp = CN2D(A, B, timesteps, radius=rDisc, location=loc, flux=fl)
        ns[str(fl)], rs[str(fl)], bs[str(fl)] =  nTemp.copy(), rTemp.copy(), bTemp.copy()


# for f, run in runs.items():

#     nTots[str(flux)] = np.sum(runs[flux][0], axis=1)


# %% Answers for flux find
# plot_3D(nx,ny, ns[str(fluxs[1])][1,:])
nTots = {}
rTots = {}
bTots = {}
if fluxFindRun:
    for fl in fluxs:
        # animate3D(ns[str(fl)])
        nTots[str(fl)] = np.sum(ns[str(fl)], axis=1)
        rTots[str(fl)] = np.sum(rs[str(fl)], axis=1)
        bTots[str(fl)] = np.sum(bs[str(fl)], axis=1)

# animate3D(ns[str(fluxs[0])])

# %% Save find flux figs


def plotFluxFind(cTots: dict, ts: int, dt: float, **kwargs):
    """plots stuff

    Args:
        cTots (dict): dict of concentrations for different fluxes
        ts (int): # of timesteps
        dt (float): u know
        show (int): Whether to show plots
        kwargs: save=fn, hlines=y_value
    """
    x = np.arange(0, ts, 1)*dt
    labs = []
    for flux, c in cTots.items():
        plt.plot(x, c)
        labs.append(str(flux))

    plt.legend(labs)

    for k, v in kwargs.items():
        match k:
            case 'hlines':
                plt.hline(v, x[0], x[-1])
            case 'save':
                plt.savefig('figures\\'+v+'.pdf')
            case 'show':
                plt.show()


if fluxFindRun:  # fluxFindSave
    # plotFluxFind(nTots, timesteps, dt, show=1)
    # plotFluxFind(rTots, timesteps, dt, show=1)
    plotFluxFind(bTots, timesteps, dt, show=1)

# %% Find time
nx = 31  # Discretization in x
ny = 31  # Dicretization in y
dx = 2*radius/nx
dy = 2*radius/ny

timesteps = 10000
nTotal = 5000
dtMax = dx**2*1e6
dtMin = dx**2*1e5
dts = np.arange(dtMin, dtMax+0.1*dtMax, (dtMax-dtMin)/1)
nTime = {}
rTime = {}
bTime = {}
bLast = []

if findTime:
    for dt in dts:
        sigma = D*dt / (2*dx*dy)

        A = makeA_2DDirichlet(nx, ny, sigma)
        B = makeA_2DDirichlet(nx, ny, -sigma)

        nTemp, rTemp, bTemp = CN2D(A, B, timesteps, flux=nTotal/timesteps)
        nTime[str(dt)] = nTemp
        rTime[str(dt)] = rTemp
        bTime[str(dt)] = bTemp
        bLast.append(np.sum(bTemp[-1, :]))

# %% remove cell caption
time = timesteps*np.array(dts)
plt.plot(time, bLast)
plt.hlines(192/2, time.min(), time.max(), linestyles='dashed')
plt.show()

# %%
