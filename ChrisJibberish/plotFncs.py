import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from matplotlib import cm


# %% 3D animate functions
def update(t, c, line, zlim):
    """updates 3D animation

    Args:
        t (int): Current timestep
        c (np.array): Concentration of substance
        line (?): line to be updated in 3D
    """
    ax.clear()
    z = c[t, :].reshape(aniNx, aniNy)
    plotset(z, zlim)
    plt.title("t = {}/{}".format(t, aniTs))
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
    ax.set_autoscalez_on(False)
    # plt.ylabel('Y AXIS')
    # plt.xlabel('X AXIS')
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    cset = ax.contour(x, y, z, zdir='x', offset=0., cmap='viridis')
    cset = ax.contour(x, y, z, zdir='y', offset=1., cmap='viridis')
    cset = ax.contour(x, y, z, zdir='z', offset=-1., cmap='viridis')


def animate3D(c, nx=0, ny=0, zlim=None, timesteps=500):
    """Animates concentration assuming quadratic grid. If not quadratic, grid 
    size in x, nx, and y, ny, must be supplied.

    Args:
        c (numpy.array): Concentration (timesteps, nx*ny)
        nx (int, optional): Grid size in x. Defaults to 0.
        ny (int, optional): Grid size in y. Defaults to 0.
        zlim (int, optional): Add concentration max limit to animation. Defaults to None.
        timesteps (int, optional): Number of frames (not used). Defaults to 500.
    """
    global x, y, z, aniNx, aniNy, aniTs
    aniTs = len(c[:, 0])
    if nx==0 and ny ==0:
        aniNx = aniNy = int(np.sqrt(len(c[0, :])))

    x = np.linspace(0, 1, aniNx)
    y = np.linspace(0, 1, aniNy)
    x, y = np.meshgrid(x, y)
    c0 = c[0, :]
    z = c0.reshape((aniNx, aniNy))

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

    ani = animation.FuncAnimation(fig, update, fargs=(c, surf, zlim),
                                  frames=timesteps, interval=1, blit=False)
    plt.show()


# %% flux for a certain time
def plotFluxTermination(cTots: dict, ts: int, dt: float, **kwargs):
    """Plots concentration for different times of flux termination.

    Args:
        cTots (dict): dict of concentrations for different fluxes
        ts (int): # of timesteps
        dt (float): Timestep size
        show (Any, optional): Whether to show plots
        kwargs: save=fn, hlines=y_value
    """
    x = np.arange(0, ts, 1)*dt
    labs = []
    for ft, c in cTots.items():
        cTot = np.sum(c, axis=1)
        plt.plot(x, cTot, label=ft)
        # labs.append(str(ft))

    # plt.legend(labs)
    plt.legend()

    for k, v in kwargs.items():
        match k:
            case 'axis':
                plt.xlabel(v[0])
                plt.ylabel(v[1])
            case 'hlines':
                plt.hlines(v, x[0], x[-1], linestyles='dashed', colors="gray")
            case 'vlines':
                for xt, c in cTots.items():
                    plt.vlines(int(xt)*dt, 0, np.sum(c[int(xt)]), colors='gray', linestyles='dashed')
                pass  # add? for stuff in args
            case 'save':
                plt.savefig('figures\\'+v+'.pdf')
            case 'show':
                plt.show()

# %% Plot functions for figures to be saved


def plot_3D(Nx, Ny, ct, g=0, save=0, fn=None, title=0, zlab=0):
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
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    if title:
        plt.title(title)
    if save:
        plt.savefig('figures\\'+fn+'.pdf')
    else:
        plt.show()
    # plt.show()


def saveNTcritPlots(nx, ny, n, b, dn=1e-2, other=[], otherLabs=[]):
    grad = np.gradient(n, axis=0)
    diff = np.abs(n[:-1, :] - n[1:, :])
    diffSum = np.sum(diff, axis=1)
    gradSum = np.sum(grad, axis=1)

    # plt.plot(diffSum)
    # plt.plot(gradSum)
    # plt.show()
    # print(np.sum(gradSum - diffSum))
    stable = np.argmax(diffSum<dn)
    signal = np.argmax(np.sum(b, axis=1)>192/2)
    plot_3D(nx, ny, n[0, :], save=1, fn="CN2D_n_initial")
    print('stable:', stable, 'signal:', signal)
    plot_3D(nx, ny, n[stable, :], save=1, fn="CN2D_n_stable_" + str(stable))
    plot_3D(nx, ny, n[signal, :], save=1, fn="CN2D_n_signal" + str(signal))
    for i in range(len(other)):
        plot_3D(nx, ny, n[other[i], :], save=1, fn=otherLabs[i])


def plotConcProgress(cDict, x=0, **kwargs):
    if type(x) == int:
        x = np.arange(0, len(next(iter(cDict.values()))))
    for key, c in cDict.items():
        cTot = np.sum(c, axis=1)
        # x = np.linspace(0, float(key), len(cTot))
        plt.plot(x, cTot, label=key)

    plt.legend()

    for k, v in kwargs.items():
        match k:
            case 'hlines':
                plt.hlines(192/2, 0, max(x), linestyles='dashed', colors='gray')
            case 'axis':
                plt.xlabel(v[0])
                plt.ylabel(v[1])
            case 'save':
                plt.savefig('figures\\'+v+'.pdf')

    plt.show()
