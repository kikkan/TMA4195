from numericalMethod import *
from plotFncs import *


""" 
* Noe galt med FindTime. Tror det ligger i funksjonskallet. den lekker for hardt. se sammenlignet med run.
"""

# %% Options
np.set_printoptions(linewidth=160)
# warnings.filterwarnings("ignore")

# %% run options
run         = 0
showAni     = 0
printAns    = 0
saveFigs    = 0
matrixDebug = 0
fluxFindRun = 0
findTime    = 1
runNoFlux   = 0

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


def animate3D(c, zlim=None, timesteps=500, interval=30):
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

    ani = animation.FuncAnimation(fig, update, fargs=(c, surf, zlim), frames=timesteps, interval=interval, blit=False)
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

# %% Discretization
nx = 51  # Discretization in x
ny = 51  # Dicretization in y
dx = 2*radius/nx
dy = 2*radius/ny
dt = dx**2*1e6  # (dx*dy?)  # TODO scale


# %% scaling
# D = 0.01
kappa = D*dt  # TODO scale?
# sigma = 1
sigma = kappa / (2*dx*dy)

# %% make grid and init state
ma   = 4*radius**2  # membrane area
cFR0 = 1e3*4*0.22**2/(nx*ny)  # r0/ma  # concentration of free receptors at t0 (1e-3)
Uv   = np.zeros(nx*ny)
cn0   = Uv.copy()  # Concentration of neurotransmitters. pop not allocated.
cr0   = Uv.copy() + cFR0  # uniform concentration
cb0   = Uv.copy()  # concentration of bounded receptors
cn0[int(nx*ny/2)] = n0


timesteps = int(500)

# A = makeA_2DNeyman(nx, ny, sigma)
# B = makeA_2DNeyman(nx, ny, -sigma)
A = makeA_2DDirichlet(nx, ny, sigma)
B = makeA_2DDirichlet(nx, ny, -sigma)

if matrixDebug:
    plt.spy(A)
    plt.grid()
    plt.show()
    # plt.spy(B)

# %% run
timesteps = int(500)
nx = 31  # Discretization in x
ny = 31  # Dicretization in y
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
    #     nx, ny, sigma, dt, timesteps,
    #     radius=rDisc, location=loc, flux=0, n0=discDistr(nx, ny, loc, rDisc, 5000)
    # )

    n, r, b = CN2D(
        nx, ny, sigma, dt, timesteps,
        radius=rDisc, location=loc, flux=50, fluxts=100
    )


# %% temp debug leakage
if run:
    # animate3D(b)
    animate3D(n)
    # plot_3D(nx, ny, n[0])
    plotConcProgress(np.sum(n, axis=1))
    # plotConcProgress(np.sum(b, axis=1))

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
        nTemp, rTemp, bTemp = CN2D(nx, ny, sigma, dt, timesteps, radius=rDisc, location=loc, flux=fl)
        ns[str(fl)], rs[str(fl)], bs[str(fl)] =  nTemp.copy(), rTemp.copy(), bTemp.copy()


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


if fluxFindRun:  # fluxFindSave
    # plotFluxFind(nTots, timesteps, dt, show=1)
    # plotFluxFind(rTots, timesteps, dt, show=1)
    plotFluxFind(bTots, timesteps, dt, show=1)

# %% Find time old

# old
# nx = 31  # Discretization in x
# ny = 31  # Dicretization in y
# dx = 2*radius/nx
# dy = 2*radius/ny

# times = np.arange(3e-5, 3.5e-5, 2e-5)  # total time
# timesLabs = ['{:.2e}'.format(t) for t in times]
# # print(timesLabs)
# timesteps = 500
# dts = times / timesteps
# nTotal = 5000
# # dtMax = dx**2*1e6
# # dtMin = dx**2*1e5
# # dts = np.arange(dtMin, dtMax+0.1*dtMax, (dtMax-dtMin)/1)
# nTime = {}
# rTime = {}
# bTime = {}
# bLast = []

# nTots = {}
# rTots = {}
# bTots = {}

# if findTime:
#     for tEnd, dt in zip(times, dts):
#         sigma = D*dt / (2*dx*dy)

#         A = makeA_2DDirichlet(nx, ny, sigma)
#         B = makeA_2DDirichlet(nx, ny, -sigma)

#         nTemp, rTemp, bTemp = CN2D(nx, ny, sigma, dt, timesteps, flux=nTotal/timesteps)
#         # nTime[str(dt)] = nTemp.copy()
#         # rTime[str(dt)] = rTemp.copy()
#         # bTime[str(dt)] = bTemp.copy()
#         bLast.append(np.sum(bTemp[-1, :]))
#         bTots[str(tEnd)] = np.sum(bTemp, axis=1)

# %% Flux certain time

nx = 21  # Discretization in x
ny = 21  # Dicretization in y
dx = 2*radius/nx
dy = 2*radius/ny

time = 1e-6
timesteps = 2000
dt = time/timesteps
# fluxTimes = np.arange(0.1*time, 0.5*time, 0.1*time)                         # set end time for flux
# fluxts = np.array([int(ft/dt) for ft in fluxTimes], dtype=int)              # depend on fluxTimes
fluxts = np.arange(1/5*timesteps, 2/5*timesteps, 0.5/5*timesteps, dtype=int)  # set # timesteps for flux
nTotal = 5000

nTime = {}
rTime = {}
bTime = {}
bLast = []

nTots = {}
rTots = {}
bTots = {}

if findTime:
    # for tEnd, dt in zip(times, dts):
    for ft in fluxts:
        sigma = D*dt / (2*dx*dy)

        # A = makeA_2DDirichlet(nx, ny, sigma)
        # B = makeA_2DDirichlet(nx, ny, -sigma)

        nTemp, rTemp, bTemp = CN2D(
            nx, ny, sigma, dt, timesteps,
            flux=nTotal/ft, fluxts=ft
        )
        nTime[str(ft)] = nTemp.copy()
        rTime[str(ft)] = rTemp.copy()
        bTime[str(ft)] = bTemp.copy()
        bLast.append(np.sum(bTemp[-1, :]))
        nTots[str(ft)] = np.sum(nTemp, axis=1)
        bTots[str(ft)] = np.sum(bTemp, axis=1)

    # n, r, b = CN2D(
    #     nx, ny, sigma, dt, timesteps,
    #     radius=rDisc, location=loc, flux=50, fluxts=100
    # )

# %% remove cell caption
# if findTime:
    # time = timesteps*np.array(dts)
    # plt.plot(time, bLast)
    # plt.hlines(192/2, times.min(), times.max(), linestyles='dashed')
    # plt.show()
    # for lab, tEnd, bTot in zip(timesLabs, times, bTots.values()):
    # x = np.linspace(0, time, timesteps)
    # for ft, bTot in zip(fluxts, bTots.values()):
    #     plt.plot(x, bTot, label='{:.0f}'.format(ft))
    # plt.hlines(192/2, 0, time, linestyles='dashed')
    # plt.ylim((0, 192/2+1))
    # plt.legend()
    # plt.show()

    # plotFluxFind(bTots, timesteps, dt, show=1)
    # for c in nTime.values():
    #     animate3D(c, timesteps=range(0, timesteps, 5), interval=0.1, zlim=np.max(c))

    # %% No flux
if findTime:
    # for b in nTots.values():
    #     plotConcProgress(b)
    plotConcProgress(*nTots.values())
    plotConcProgress(*bTots.values())
