from numericalMethod import *
from plotFncs import *


# %% Options
np.set_printoptions(linewidth=160)
warnings.filterwarnings("ignore")

# %% run options
runSingle              = 0
singleRunAni           = 0
singleRunPrintAns      = 0
singleRunSaveFigs      = 0
matrixDebug            = 0

fluxTermination        = 0
variableFlux           = 0
constNTdiffTermination = 0
variableRadius         = 0
neymannBound           = 1


# %% Units
height = 15e-9     # Height of cleft               (15e-9) (not used)
radius = 220e-9    # Radius of cleft/membrane      (220e-9)
D      = 8e-7      # Diffusion coefficient         (8e-7)
k1     = 4e6       # Forward reaction coefficient  (4e6)
km1    = 5         # Backward reaction coefficient (5)
r0     = 152       # receptors on membrane         (1000/(1e-6)^2 * 2*pi*(220e-9))
r0     = 192       # Square area                   (192)
n0     = 5000      # Neurotransmitters per vesicle pop (5000)

# %% Matrix visualization
if matrixDebug:
    A = makeA_2DDirichlet(21, 21, 2)
    A = makeA_2DNeyman(21, 21, 2)
    plt.spy(A)
    plt.grid()
    plt.show()


# %% Standard discretization and values
std_time      = 1e-5
std_timesteps = 2000
std_dt        = std_time/std_timesteps
std_nx        = 31  # Discretization in x
std_ny        = 31  # Dicretization in y
std_dx        = 2*radius/std_nx
std_dy        = 2*radius/std_ny
# std_dt = std_dx**2*1e8

std_sigma = D*std_dt / (2*std_dx*std_dy)

std_rDisc = std_nx/2  # defaults to width of grid
std_loc = [std_nx/2, std_nx/2]  # defaults to center
std_flux = 20

# %% Run single
if runSingle:
    # A single run using standard values
    print('Run single')
    n, r, b = CN2D(
        std_nx, std_ny, std_sigma, std_dt, std_timesteps,
        radius=std_rDisc, location=std_loc, flux=std_flux, fluxts=std_timesteps
    )
    st = stable(n)
    si = signal(b)

    print(
        '\n\nrun settings:\n'
        'Grid: (nx,ny) =         ({}, {})\n'
        'Timesteps, dt, endTime: {}, {:.3e}, {:.3e}\n'
        'flux:                   {}\n'
        'stable:                 {}\n'
        'signal:                 {}'
        ''.format(std_nx, std_ny, std_timesteps, std_dt, std_dt*std_timesteps,
                  std_flux, st, si)
    )

    with open('values\\runSingle.txt', 'w') as f:
        f.write(
            'Grid: (nx,ny) =         ({}, {})\n'
            'Timesteps, dt, endTime: {}, {:.3e}, {:.3e}\n'
            'flux:                   {}\n'
            'stable:                 {}\n'
            'signal:                 {}'
            ''.format(std_nx, std_ny, std_timesteps, std_dt, std_dt*std_timesteps,
                      std_flux, st, si)
        )

    saveNTcritPlots(std_nx, std_ny, n, st, si, fnAdd='n')
    plt.show()
    saveNTcritPlots(std_nx, std_ny, b, st, si, fnAdd='b')
    plt.show()
    saveNTcritPlots(std_nx, std_ny, r, st, si, fnAdd='r')
    plt.show()


# %% Animation
if singleRunAni:
    animate3D(n)
    # animate3D(r, 1)
    # animate3D(b, 1)


# %% Critical times (Answers)
if singleRunPrintAns:  # print and compute misc answers
    print(
        'run informatio:\n'
        'Discretization:\n'
        '   (nx, ny, ts) = ({},{}, {})\n'
        '   (dx, dy, dt) = ({:.3e},{:.3e}, {:.3e})\n'
        ''.format(
            std_nx, std_ny, std_timesteps,
            std_dx, std_dy, std_dt
        )
    )

    nTot = np.sum(n, axis=1)
    rTot = np.sum(r, axis=1)
    bTot = np.sum(b, axis=1)
    ts = np.arange(std_timesteps)
    plt.plot(ts, bTot)
    plt.plot(ts, rTot)
    plt.hlines(y=r0/2, xmin=0, xmax=std_timesteps)
    plt.show()

    signal = np.argmax(bTot > r0/2)
    print('When is signal transmitted?\n'
          'Values:\n'
          'index {}\n'
          'time  {}'.format(signal, signal*std_dt))


# %% Save single run figures
if singleRunSaveFigs:
    plot_3D(std_nx, std_ny, n[0, :], 0.2, 1, 'CN2D_initial')
    plot_3D(std_nx, std_ny, n[signal, :], 0.2, 1, "CN2D_Signal")
    plot_3D(std_nx, std_ny, n[-1, :], 0.2, 1, "CN2D_last")

    plotConcProgress(rTot, bTot, labels=['r', 'b'], save="CN2D_conc_r&b")
    plotConcProgress(
        rTot, bTot, nTot-n0+r0,
        labels=[r'$r$', r'$b$', r'$n-n_0+r_0$'],
        save="CN2D_conc_nr&b",
        linestyle=['-', '-', 'dashed']
    )

# %% Flux termination
time = 1e-5
timesteps = 2000
nx = 31  # Discretization in x
ny = 31  # Dicretization in y
dx = 2*radius/nx
dy = 2*radius/ny
# dt = dx**2*1e6
dt = time/timesteps
sigma = D*dt/(2*dx*dy)


ns = {}
rs = {}
bs = {}

rDisc = nx/2
loc = [nx/2, nx/2]  # defaults to center

fluxTimes = np.arange(0.2*timesteps, 0.91*timesteps, 0.15*timesteps, dtype=int)
flux=10

if fluxTermination:
    print('Run {} times. TermTimes {}'.format(len(fluxTimes), fluxTimes))
    for ft in fluxTimes:
        ns[str(ft)], rs[str(ft)], bs[str(ft)] = CN2D(
            nx, ny, sigma, dt, timesteps,
            radius=rDisc, location=loc, flux=flux, fluxts=ft)

    # animate3D(next(iter(ns.values())))

    ########################### save and plot
    plotFluxTermination(
        bs, timesteps, dt,
        hlines=192/2, axis=(r'$s$', r'Bounded receptors'), vlines=1, ticks=1,
        save='CN2D_b_fluxForATime',
        show=1
    )

    fileTermination = open('values\\fluxTermination.txt', 'w')
    fileTermination.write(
        'Grid: (nx,ny) =    ({}, {})\n'
        'Timesteps, dt:     {}, {:.3e}\n'
        'Termination times: {}\n'
        ''.format(nx, ny, timesteps, dt, fluxTimes)
    )
    fileTermination.close()

    # plot
    # plotFluxTermination(
    #     bs, timesteps, dt,
    #     hlines=192/2, axis=(r'$s$', r'Bounded receptors'), vlines=1,
    #     show=1
    # )

    print(
        '\n\nrun settings:\n'
        'Grid: (nx,ny) =    ({}, {})\n'
        'Timesteps, dt:     {}, {:.3e}\n'
        'Termination times: {}\n'
        ''.format(nx, ny, timesteps, dt, fluxTimes)
    )


# %% Variable flux
if variableFlux:
    nx = 31  # Discretization in x
    ny = 31  # Dicretization in y
    dx = 2*radius/nx
    dy = 2*radius/ny

    fluxs = np.arange(5, 26, 5)  # varying
    timesteps = 2000  # fixed
    time = 1e-5
    # keys = ['{:.2e}'.format(t) for t in times]
    keys = ['{}'.format(f) for f in fluxs]
    dt = time/timesteps
    # dt = dx*dy*1e6
    # time = dt*timesteps

    nTime = {}
    rTime = {}
    bTime = {}

    print('Running {} times. Fluxes {}'.format(len(fluxs), fluxs))
    for f in fluxs:
        sigma = D*dt/(2*dx*dy)
        # key = '{:.2e}'.format(tEnd)
        nTemp, rTemp, bTemp = CN2D(
            nx, ny, sigma, dt, timesteps,
            flux=f, fluxts=timesteps)
        nTime[str(f)], rTime[str(f)], bTime[str(f)] = nTemp.copy(), rTemp.copy(), bTemp.copy()

    # animate3D(bTime[str(f)])
    x = np.linspace(0, time, timesteps)
    plotConcProgress(nTime, x=x, axis=(r'$s$', r'Neurotransmitters'),
                     save="CN2D_n_variableFluxNoDiff"
                     )
    plt.figure()
    plotConcProgress(bTime, x=x, axis=(r'$s$', r'Bounded receptors'), hlines=1,
                     save="CN2D_b_variableFluxNoDiff"
                     )
    plt.figure()
    plotConcProgress(rTime, x=x, axis=(r'$s$', r'Free receptors'), hlines=1,
                     save="CN2D_r_variableFluxNoDiff"
                     )

    plt.show()
    # plt.close()

    fileVarFlux = open('values\\varFlux.txt', 'w')
    fileVarFlux.write(
        'Grid: (nx,ny) =    ({}, {})\n'
        'Timesteps, dt:     {}, {:.3e}\n'
        'Fluxes:            {}\n'
        ''.format(nx, ny, timesteps, dt, fluxs)
    )
    fileVarFlux.close()

    print(
        '\n\nrun settings:\n'
        'Grid: (nx,ny) = ({}, {})\n'
        'Timesteps, dt: {}, {:.3e}\n'
        'Fluxes: {}\n'
        ''.format(nx, ny, timesteps, dt, fluxs)
    )

# %% Total number of NT for different termination times
if constNTdiffTermination:
    N = 12500

    nx = 31  # Discretization in x
    ny = 31  # Dicretization in y
    dx = 2*radius/nx
    dy = 2*radius/ny

    timesteps = 2000  # fixed
    time = 1e-5  # set end time
    dt = time/timesteps
    sigma = D*dt/(2*dx*dy)
    terminations = np.linspace(0.1*timesteps, 0.9*timesteps, 5, dtype=int)  # varying

    keys = ['{:.1f}'.format(term) for term in terminations]

    ns = {}
    rs = {}
    bs = {}

    print('Run {} times'.format(len(terminations)))
    for termTime, key in zip(terminations, keys):
        ns[key], rs[key], bs[key] = CN2D(nx, ny, sigma, dt, timesteps,
                                         flux=N/termTime, fluxts=termTime)

    # animate3D(bs[keys[0]])
    plotFluxTermination(
        ns, timesteps, dt, vlines=1,
        axis=(r's', r'Neurotransmitters'), ticks=1,
        save='CN2D_n_constNTdiffTerm',
        # show=1
    )
    plt.figure()
    plotFluxTermination(
        rs, timesteps, dt, vlines=1, hlines=192/2,
        axis=(r's', r'Free receptors'), ticks=1,
        save='CN2D_r_constNTdiffTerm',
        # show=1
    )
    plt.figure()
    plotFluxTermination(
        bs, timesteps, dt, vlines=1, hlines=192/2, ticks=1,
        axis=(r's', r'Bounded receptors'),
        save='CN2D_b_constNTdiffTerm',
        # show=1
    )
    plt.show()
    # plt.close()

    print(
        '\n\nrun settings:\n'
        'Grid: (nx,ny) =    ({}, {})\n'
        'Timesteps, dt:     {}, {:.3e}\n'
        'Terminations:      {}\n'
        'Total NT:          {}'
        ''.format(nx, ny, timesteps, dt, keys, N)
    )
    fileConstNTtermination = open('values\\constNTdiffTermination.txt', 'w')
    fileConstNTtermination.write(
        'Grid: (nx,ny) =    ({}, {})\n'
        'Timesteps, dt:     {}, {:.3e}\n'
        'Terminations:      {}\n'
        'Total NT:          {}'
        ''.format(nx, ny, timesteps, dt, keys, N)
    )
    fileConstNTtermination.close()

# %% Variable radius
if variableRadius:
    time = std_time
    timesteps = std_timesteps
    nx=std_nx
    ny=std_ny
    dt=std_dt
    sigma=std_sigma
    flux = std_flux

    rDiscs = np.linspace(0.1, 1, 5)*nx
    keys = ['{:.2e}'.format(r) for r in rDiscs]

    ns = {}
    rs = {}
    bs = {}

    print('Run variable radius {} times'.format(len(rDiscs)))
    for r, key in zip(rDiscs, keys):
        ns[key], rs[key], bs[key] = CN2D(nx, ny, sigma, dt, timesteps, flux=flux, fluxts=timesteps,
                                         radius=r)

    x = np.linspace(0, time, timesteps)
    plotConcProgress(ns, x=x, axis=(r'$s$', r'Neurotransmitters'),
                     save="CN2D_n_variableRadius"
                     )
    plt.figure()
    plotConcProgress(bs, x=x, axis=(r'$s$', r'Bounded receptors'), hlines=1,
                     save="CN2D_b_variableRadius"
                     )
    plt.figure()
    plotConcProgress(rs, x=x, axis=(r'$s$', r'Free receptors'), hlines=1,
                     save="CN2D_r_variableRadius"
                     )

    plt.show()


# %% Neymann boundary conditions
if neymannBound:
    time = 1e-6
    ts = 1000
    dt = time/ts
    nx = ny=31
    dx = dy = 2*radius/nx
    sigma = D*dt/(2*dx*dy)
    flux=500

    # n0 = discDistr(nx, nx, [nx, nx]/2, nx/2, 500)
    n, r, b = CN2D(nx, ny, sigma, dt, ts,
                   flux=flux, fluxts=1, makeA=makeA_2DNeyman)

    ngrad = np.abs(np.gradient(n, axis=0))
    gradsum = np.sum(ngrad, axis=1)
    st = np.argmax(gradsum<0.5)
    si = np.argmax(np.sum(b, axis=1)>192/2)
    print(st, si)
    print(stable(n, 0.5), signal(b))  # could use this instead
    # exit()
    runInfo(
        1, 1, fn='values\\neymannBoundaries.txt',
        grid=[nx, ny], time=time, timesteps=ts, dt=dt, flux=flux, stable=st, signal=si
    )

    # plt.plot(gradsum)
    # plt.vlines(stable, 0, max(gradsum))
    # plt.show()

    # animate3D(n, timesteps=ts)
    plotConcProgress({r'N': n, r'B': b, r'R': r}, hlines=192/2, axis=[r'Timesteps', 'Number of particles'], save='CN2D_Neymann_concProgress')
    plt.show()

    # saveNTcritPlots(nx, ny, n, st, si, fnAdd='Neymann_n', zlim=[0, np.max(n)*(1.+0.1)])
    # # plt.show()
    # saveNTcritPlots(nx, ny, b, st, si, fnAdd='Neymann_b', zlim=[0, np.max(b)*(1.+0.1)])
    # # plt.show()
    # saveNTcritPlots(nx, ny, r, st, si, fnAdd='Neymann_r', zlim=[0, np.max(r)*(1.+0.1)])
    # # plt.show()
