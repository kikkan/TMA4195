from numericalMethod import *
from plotFncs import *


# %% Options
np.set_printoptions(linewidth=160)
warnings.filterwarnings("ignore")

# %% run options
runSingle          = 0
singleRunAni       = 0
singleRunPrintAns  = 0
singleRunSaveFigs  = 0
matrixDebug        = 0
fluxTermination    = 0
variableFlux       = 0
totalNToverVarTime = 1


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


# %% Standard discretization
std_time = 1e-5
std_timesteps = 5000
# dt = time/timesteps
std_nx = 31  # Discretization in x
std_ny = 31  # Dicretization in y
std_dx = 2*radius/std_nx
std_dy = 2*radius/std_ny
std_dt = std_dx**2*1e6
std_sigma = D*std_dt / (2*std_dx*std_dy)

std_rDisc = std_nx/2
std_loc = [std_nx/2, std_nx/2]  # defaults to center
std_flux = 5

# %% Run single
if runSingle:
    # A single run using standard values
    n, r, b = CN2D(
        std_nx, std_ny, std_sigma, std_dt, std_timesteps,
        radius=std_rDisc, location=std_loc, flux=std_flux, fluxts=std_timesteps
    )
    animate3D(b)
    # plotConcProgress({str(flux): b})
    saveNTcritPlots(std_nx, std_ny, n, b)


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

# %% Find flux
nx = 31  # Discretization in x
ny = 31  # Dicretization in y
dx = 2*radius/nx
dy = 2*radius/ny
dt = dx**2*1e6  # (dx*dy?)  # TODO scale
sigma = D*dt/(2*dx*dy)

timesteps = 5000

ns = {}
rs = {}
bs = {}

rDisc = nx/2
loc = [nx/2, nx/2]  # defaults to center

fluxTimes = np.arange(0.1*timesteps, 0.91*timesteps, 0.1*timesteps, dtype=int)
flux=10

if fluxTermination:
    print('Run {} times'.format(len(fluxTimes)))
    for ft in fluxTimes:
        ns[str(ft)], rs[str(ft)], bs[str(ft)] = CN2D(
            nx, ny, sigma, dt, timesteps,
            radius=rDisc, location=loc, flux=flux, fluxts=ft)

    # animate3D(next(iter(ns.values())))

    ########################### save and plot
    plotFluxTermination(
        bs, timesteps, dt,
        hlines=192/2, axis=(r'$s$', r'Bounded receptors'), vlines=1,
        save='CN2D_b_fluxForATime', show=1
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
    time = 1e-6
    # keys = ['{:.2e}'.format(t) for t in times]
    keys = ['{}'.format(f) for f in fluxs]
    # dt = time/timesteps
    dt = dx*dy*1e6
    time = dt*timesteps

    nTime = {}
    rTime = {}
    bTime = {}

    print('Running {} times'.format(len(fluxs)))
    for f in fluxs:
        sigma = D*dt/(2*dx*dy)
        # key = '{:.2e}'.format(tEnd)
        nTemp, rTemp, bTemp = CN2D(
            nx, ny, sigma, dt, timesteps,
            flux=f, fluxts=timesteps)
        nTime[str(f)], rTime[str(f)], bTime[str(f)] = nTemp.copy(), rTemp.copy(), bTemp.copy()

    # animate3D(bTime[str(f)])
    x = np.linspace(0, time, timesteps)
    plotConcProgress(nTime, x=x, axis=(r'$s$', r'Neurotransmitters'), save="CN2D_n_variableFluxNoDiff")
    plotConcProgress(bTime, x=x, axis=(r'$s$', r'Bounded receptors'), hlines=1, save="CN2D_b_variableFluxNoDiff")
    plotConcProgress(rTime, x=x, axis=(r'$s$', r'Free receptors'), hlines=1, save="CN2D_r_variableFluxNoDiff")
    """Variable flux no diff uses no diffusion of r and b properly."""

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
if totalNToverVarTime:
    N = 15000

    nx = 31  # Discretization in x
    ny = 31  # Dicretization in y
    dx = 2*radius/nx
    dy = 2*radius/ny

    timesteps = 2000  # fixed
    time = 1e-5  # set end time
    dt = time/timesteps
    sigma = D*dt/(2*dx*dy)
    terminations = np.linspace(0.1*timesteps, 0.9*timesteps, 5, dtype=int)  # varying

    keys = ['{:.2e}'.format(term) for term in terminations]

    ns = {}
    rs = {}
    bs = {}

    for termTime, key in zip(terminations, keys):
        # key = '{:.2e}'.format(tEnd)
        ns[key], rs[key], bs[key] = CN2D(
            nx, ny, sigma, dt, timesteps,
            flux=N/termTime, fluxts=termTime)

        # animate3D(bs[keys[0]])
