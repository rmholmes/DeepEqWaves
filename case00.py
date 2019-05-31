"""
Dedalus script for 2D Inertio-gravity waves on an equatorial beta plane 
with non-traditional Coriolis accelearion.

This script uses a Fourier basis in the x direction with periodic boundary
conditions.

This script can be ran serially or in parallel, and uses the built-in analysis
framework to save data snapshots in HDF5 files.  The `merge.py` script in this
folder can be used to merge distributed analysis sets from parallel runs,
and the `plot_2d_series.py` script can be used to plot the snapshots.

To run, merge, and plot using 4 processes, for instance, you could use:
    $ mpiexec -n 4 python3 igw.py
    $ mpiexec -n 4 python3 merge.py snapshots
    $ mpiexec -n 4 python3 plot_2d_series.py snapshots/*.h5

"""

import numpy as np
from mpi4py import MPI
from scipy.special import erf
import time

from dedalus import public as de
from dedalus.extras import flow_tools

import logging
logger = logging.getLogger(__name__)

# directories:
rundir = '/short/e14/rmh561/dedalus/DeepEqWaves/rundir/'

# Parameters
Ly, Lz = (800., 4.) # units = 1km

# Create bases and domain
y_basis = de.Fourier('y', 512, interval=(-Ly, Ly), dealias=3/2)
z_basis = de.Chebyshev('z', 256, interval=(0, Lz), dealias=3/2)
domain = de.Domain([y_basis, z_basis], grid_dtype=np.float64)

# Periodic β
ε = 0.03
y = domain.grid(0)
z = domain.grid(1)
Y = domain.new_field()
Y['g'] = Ly*(y/Ly - (2/np.pi)*np.arctan(ε*np.tan(np.pi*y/(2*Ly))))/(1-ε)

# Sponge layer
SL0,dLy = 1/240,Ly/10
SL = domain.new_field()
SL['g'] = SL0*np.exp(-(Ly*np.cos(np.pi*y/(2*Ly))/dLy)**2)

# Forcing
Amp, ay, az, σy, σz = 0.02, 0, 3.8, 30, 0.2
F0 = domain.new_field()
F0['g'] = Amp*np.exp(-((y-ay)/σy)**2)*(1+erf((z-az)/σz))

Ω = 2*np.pi/24 # units = 1 hour
f = 2*Ω
β = f/6378 # earth radius = 6378km

#Stratification profile
B = domain.new_field()
Btop, α = 80, 0.6 # Low Stratification Test Case
#Btop, α = 325, 0.75 # Exponential Fit to Observed
B['g'] = Btop*np.exp((z-Lz)/α)
Bbot   = Btop*np.exp(-Lz/α)

# Background stratification for linear simulation
N2bak = domain.new_field()
N2bak.meta['y']['constant'] = True
N2bak['g'] = Btop*np.exp((z-Lz)/α)/α

N2ref = domain.new_field()
N2ref.meta['y']['constant'] = True
N2ref['g'] = (Btop/α)*(z/Lz)**(Lz//α) # because (1 + x/n)**n ~ exp(x)

# Linear or Non-linear:
LIN = True

# 2D Boussinesq hydrodynamics
problem = de.IVP(domain, variables=['p','b','u','v','w','bz','uz','ζ'])
problem.meta[:]['z']['dirichlet'] = True

problem.parameters['κy']  = 1e-4  #RA: Equivalent 2.77e-2 m2s-1
problem.parameters['νy']  = 1e-4
problem.parameters['κz']  = 1e-7  #RA: Equivalent 2.77e-5 m2s-1
problem.parameters['νz']  = 1e-7
problem.parameters['f']  = f
problem.parameters['ω']  = Ω/10 # 10 days
problem.parameters['βY'] = β*Y
problem.parameters['N2bak'] = N2bak
problem.parameters['N2ref'] = N2ref
problem.parameters['α'] = α
problem.parameters['F0'] =  F0
problem.parameters['SL'] = SL
problem.parameters['Bbot'] = Bbot
problem.parameters['Btop'] = Btop
problem.add_equation("dy(v) + dz(w) = 0")
if LIN:
    problem.add_equation("dt(b) - κy*d(b,y=2) - κz*( dz(bz) - bz/α ) + w*N2ref = - w*(N2bak - N2ref)  - SL*b")
    problem.add_equation("dt(u) + f*w - νy*d(u,y=2) - νz*dz(uz) =  βY*v - SL*u")
    problem.add_equation("dt(v) - (νy-νz)*d(v,y=2) + νz*dz(ζ) + dy(p) = -βY*u - SL*v + F0*sin(ω*t)")
    problem.add_equation("dt(w) - f*u - (νy-νz)*d(w,y=2) - νz*dy(ζ) + dz(p) - b = - SL*w")
else:
    problem.add_equation("dt(b)       -      κy*d(b,y=2) - κz*dz(bz)  + w*N2ref =        -(v*dy(b) + w*(bz-N2ref))")
    problem.add_equation("dt(u) + f*w -      νy*d(u,y=2) - νz*dz(uz)            =  βY*v  -(v*dy(u) + w*uz)      - SL*u")
    problem.add_equation("dt(v)       - (νy-νz)*d(v,y=2) + νz*dz(ζ) + dy(p)     = -βY*u  + ζ*w                  - SL*v + F0*sin(ω*t)")
    problem.add_equation("dt(w) - f*u - (νy-νz)*d(w,y=2) - νz*dy(ζ) + dz(p) - b =        - ζ*v                  - SL*w")
problem.add_equation("bz - dz(b) = 0")
problem.add_equation("uz - dz(u) = 0")
problem.add_equation("ζ + dz(v) - dy(w) = 0")
# RA: Advection terms are expressed using ζ by transforming pressure
# P -> P - 1/2*w^2 - 1/2*v^2
if LIN:
    problem.add_bc("left(bz) = 0")
    problem.add_bc("right(b) = 0")
else:
    problem.add_bc("left(b) = Bbot")
    problem.add_bc("right(b) = Btop")
problem.add_bc("left(u) = 0")
problem.add_bc("left(v) = 0")
problem.add_bc("left(w) = 0")
problem.add_bc("right(uz) = 0")
problem.add_bc("right(ζ) = 0")
problem.add_bc("right(w) = 0", condition="(ny != 0)")
problem.add_bc("right(p) = 0", condition="(ny == 0)")

# Build solver
solver = problem.build_solver(de.timesteppers.RK222)
logger.info('Solver built')

# Initial conditions
if !LIN:
    b = solver.state['b']
    p = solver.state['p']
    bz = solver.state['bz']
    b['g'] = B['g']
    p['g'] = α*(B['g']-Btop)
    b.differentiate('z', out=bz)

# Integration parameters
solver.stop_sim_time = np.inf
solver.stop_wall_time = 6*60*60.
solver.stop_iteration = np.inf

# Analysis
snapshots = solver.evaluator.add_file_handler(rundir + 'snapshots', iter=20, max_writes=50)
snapshots.add_task("b")
snapshots.add_task("bz")
snapshots.add_task("p")
snapshots.add_task("u")
snapshots.add_task("uz")
snapshots.add_task("v")
snapshots.add_task("w")
snapshots.add_task("ζ")
if LIN:
    snapshots.add_task("(uz**2+ζ**2)/abs(N2bak+bz)",name='invRi')
else:
    snapshots.add_task("(uz**2+ζ**2)/abs(bz)",name='invRi')

energies = solver.evaluator.add_file_handler(rundir + 'energies', iter=20, max_writes=50)
energies.add_task("0.5*u*u",name="Energy-x")
energies.add_task("0.5*v*v",name="Energy-y")
energies.add_task("0.5*w*w",name="Energy-z")
if LIN:
    energies.add_task("0.5*b*b/N2bak",name="Energy-B")

dissipation = solver.evaluator.add_file_handler(rundir + 'dissipation', iter=20, max_writes=50)
dissipation.add_task("νy*dy(u)**2",name='u-y-diss')
dissipation.add_task("νy*dy(v)**2",name='v-y-diss')
dissipation.add_task("νy*dy(w)**2",name='w-y-diss')
dissipation.add_task("νz*ζ**2",name='vorticity-diss')
dissipation.add_task("νz*uz**2",name='u-z-diss')
if LIN:
    dissipation.add_task("(κy/N2bak)*dy(b)**2",name='b-y-diss')
    dissipation.add_task("(κz/N2bak)*bz**2",name='b-z-diss')

dt=0.25
# CFL
#CFL = flow_tools.CFL(solver, initial_dt=0.1, cadence=10, safety=1,
#                     max_change=1.5, min_change=0.5, max_dt=0.1)
#CFL.add_velocities(('v','w'))

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
if LIN:
    flow.add_property("sqrt((uz**2+ζ**2)/abs(N2bak+bz))", name='root_inv_Ri')
else:
    flow.add_property("sqrt((uz**2+ζ**2)/abs(bz))", name='root_inv_Ri')

# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.ok:
        #        dt = CFL.compute_dt()
        solver.step(dt)
        if (solver.iteration-1) % 10 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            logger.info('Max 1/sqrt(Ri) = %f' %flow.max('root_inv_Ri'))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))
