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

k1     = 4e6  # Forward reaction coefficient (4e6)
km1    = 5  # Backward reaction coefficient (5)


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


def CNstep(n, r, b, Ainv, AinvB, dt, t, flux=0):
    nx = int(np.sqrt(Ainv.shape[0]))
    dtf = Ainv@(dt*f(n[t, :], r[t, :], b[t, :]))
    n[t+1, :] = AinvB@n[t, :] + dtf + flux
    r[t+1, :] = r[t, :] + dtf
    b[t+1, :] = b[t, :] - dtf
    n[t, [0, nx-1]] = 0
    for i in range(nx, nx*nx-nx, nx):
        # can make diag identity for this maybe
        n[t, [i, i+nx-1]] = 0


def CN2D(nx, ny, sigma, dt, ts: int, flux=0, **kwargs):
    """Performs CN in 2D using matrix multiplication of sparse matrices.

    Args:
        A (csr_matrix): Block matrix A (sparse)
        B (csr_matrix): Block matrix A (sparse)
        ts (int, optional): Timesteps.
        kwargs (dictionary): can take specific n0, in flux, disc location and radius

    Returns:
        (n, r, b): Concentrations of n, r, b for all timesteps
    """
    A = makeA_2DDirichlet(nx, ny, sigma)
    B = makeA_2DDirichlet(nx, ny, -sigma)
    # matrices for all timesteps
    n = np.zeros((ts, nx*ny))
    r = np.zeros((ts, nx*ny))
    b = np.zeros((ts, nx*ny))
    # n0 = discDistr(nx, ny, [nx/2, ny/2], 0.2*nx, 5000)
    # n[0, :], r[0, :], b[0, :] = n0, r0, b0

    rDisc = nx/2
    loc   = [nx/2, ny/2]  # center
    # init values
    # flux = 0
    fluxts=ts
    m = discDistr(nx, ny, loc, rDisc, flux)
    # mSum = np.sum(m)
    n[0, :] = m
    r[0, :] = 1e3*4*0.22**2/(nx*ny)  # uniform conc of receptors on entire grid

    # Kwargs handling
    for key, value in kwargs.items():
        match key:
            case 'n0': n[0, :] = value
            case 'radius': rDisc = value
            case 'location': loc = value
            # case 'flux': flux = value  # don't need to be case?
            case 'fluxts':
                fluxts = value
                # print('Total NT:', flux*fluxts)
            case other as o:
                print('Ey, "{}" is not correct, duh.'.format(o))

    print('\n\nRun details:\n'
          'Grid (nx, ny):   {}\n'
          'Timesteps:       {}\n'
          'Total NT/step:   {}\n'
          'Total NT         {}\n'
          'NT/sec           {:.2e}'
          ''.format((nx, ny), ts, flux, flux*fluxts, flux/dt)
          )

    # Precompute/allocate memory
    Ainv  = inv(A)
    AinvB = Ainv@B
    dtf   = np.zeros(nx*ny)

    with alive_bar(ts-1) as bar:
        if fluxts:
            for t in (range(fluxts)):
                CNstep(n, r, b, Ainv, AinvB, dt, t, m)
                bar()
            m = np.zeros(nx**2)

        for t in range(fluxts, ts-1):
            CNstep(n, r, b, Ainv, AinvB, dt, t, m)
            # for i in range(nx, nx*ny-nx, nx):
            #     # can make diag identity for this maybe
            #     n[t, [i, i+nx-1]] = 0
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
