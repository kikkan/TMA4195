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


def animate3D(c, zlim=None, timesteps=500):
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


# %% flux for a certain time
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
