import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, csc_matrix
from scipy.sparse.linalg import spsolve, inv
from numericalMethod import *
from plotFncs import *

# offset = np.array([-1, 0, 1])

# d1 = diags([1, 2, 1], offset, (4, 4))
# d2 = diags([2, 1, 2], offset, (4, 4))
# d3 = np.hstack((d1.toarray(), d2.toarray()))
# # print(d3)

# # %% sparse stack
# d3s = hstack([d1, d2])
# d3sArr = d3s.toarray()
# print(d3sArr)


# # %% stacked and sparse stacked matrix mult
# n1 = d3s.shape[1]
# print(n1)
# v = np.array([1 for _ in range(n1)])
# m = d3s@v
# print(m)

# n0=r0=b0 = np.ones(5)

# n = r = b = np.zeros((3, 5))
# n[0, :], r[0, :], b[0, :] = n0, r0, b0
# print(n)
# print(r)
# print(b)
# print("hello world")


# %% make 3D A and invert
# import time
# nz = 30
# nx = ny = nz
# # nx = ny = 15*nz
# offset = [-2*nx, -nx, -1, 0, 1, nx, 2*nx]
# A = diags([-1, -1, -1, 5, -1, -1, -1], offset, (nx*ny*nz, nx*ny*nz))
# # plt.spy(A)
# # plt.show()
# B=csc_matrix(A)
# s = time.time()
# Ainv = inv(B)
# print(time.time() - s)


# %% circle distr.
# def plot_3D(Nx, Ny, ct, g=0, save=0, fn=None, title=0):
#     x = np.linspace(0, 1, Nx)
#     y = np.linspace(0, 1, Ny)
#     x, y = np.meshgrid(x, y)
#     fig = plt.figure()
#     ax = fig.gca(projection="3d")
#     if g:
#         diff = ct.max() - ct.min()
#         ax.axes.set_zlim3d(ct.min() - g*diff, ct.max() + g*diff)
#     # z = ct.reshape((Nx, Ny))
#     z= ct
#     ax.plot_surface(x, y, z, cmap="viridis")
#     if title:
#         plt.title(title)
#     if save:
#         plt.savefig('figures\\'+fn+'.pdf')
#     else:
#         plt.show()


def abc(x0, y0, x, r):
    c = y0**2 + (x-x0)**2 - r**2
    b = -2*y0
    nom1 = -b + np.sqrt(b**2-4*c)
    nom2 = -b - np.sqrt(b**2-4*c)
    denom = 2
    return np.array([nom1, nom2])/denom


def discDistr(nx, ny, prec, r, loc):
    x = np.linspace(loc[0]-r, loc[1]+r, prec)
    y = abc(loc[0], loc[1], x, r)

    if True:
        plt.plot(x, y[0])
        plt.plot(x, y[1])
        plt.hlines(loc[0], 0, nx)
        plt.vlines(loc[1], 0, ny)
        plt.grid()
        plt.show()
    m = np.zeros((nx, ny))
    for i in range(len(x)):
        m[int(x[i]), [int(y[0, i]), int(y[1, i])]] = 1
    return x, y, m


# plt.plot(x, y[0])
# plt.plot(x, y[1])
# plt.grid()
# plt.show()
# nx=31
# ny=31
# x, y, m = discDistr(nx, ny, nx*ny, (nx/2)-2, [nx/2, nx/2])

# plot_3D(nx, ny, m)
# mArr = m.reshape(nx*ny)
# plt.plot(mArr)
# plt.show()
# print(mArr)

# %%
# def discDistr(nx, ny, loc, r, fluxTot):
#     m = np.zeros((nx, ny))
#     ind = []
#     for i in range(nx):
#         for j in range(ny):
#             if (i-loc[0])**2 + (j-loc[1])**2 <= r**2:
#                 m[i, j] = 1
#                 ind.append([i, j])
#     # print(np.array(ind))
#     ind = np.array(ind)
#     m[ind[:, 0], ind[:, 1]] = fluxTot/len(ind[:, 0])
#     # print(round(m, 2))

#     # make area = 1
#     return m


# nx = 15
# ny = 15
# loc = [(nx-1)/2, (ny-1)/2]
# r = (nx/2)*0.5

# discDistr(nx, ny, loc, r, 3)
# print(m)

# %% plot ticks

# x = np.linspace(-10, 10, 35)
# y = x**2
# plt.plot(x, y)
# plt.xticks([-5, 0, 5])
# plt.show()


# %%
radius    = 220e-9
D         = 8e-7
time      = 1e-5
timesteps = 100
# dt        = time/timesteps
dt        = 1e-9
nx        = 31  # Discretization in x
ny        = 31  # Dicretization in y
dx        = 2*radius/nx
dy        = 2*radius/ny
sigma     = D*dt / (2*dx*dy)
rDisc     = 0.1*nx/2  # defaults to width of grid
loc       = [nx/2, nx/2]  # defaults to center
flux      = 20

# n, r, b = CN2D(nx, ny, sigma, dt, timesteps, flux=flux, fluxts=timesteps,
#                radius=rDisc)
# np.set_printoptions(linewidth=160, precision=2)
# for i in range(timesteps):
#     #     plot_3D(nx, ny, n[i, :], title=i)
#     print(np.reshape(n[i, :], (nx, ny)))
#     print()
# # plt.show()
# plotConcProgress({'n': n})
# animate3D(n, interval=700, timesteps=timesteps-1, zlim=6)


# %% sin and cos distr of R
def rSinCos(nx, ny, R=192):
    r = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            r[i, j] = np.sin(i) + np.cos(j) + 2
    r = R*r/np.sum(r)
    print(np.sum(r))
    return np.reshape(r, nx*ny)


nr, rr, br = CN2D(nx, ny, sigma, dt, timesteps, flux=flux, fluxts=timesteps,
                  radius=nx/2, r0=rSinCos(nx, ny))
n, r, b = CN2D(nx, ny, sigma, dt, timesteps, flux=flux, fluxts=timesteps,
               radius=nx/2)

plotConcProgress({'br': br, 'b': b})
plotConcProgress({'rr': rr, 'r': r})
plt.show()
