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


# Parameters
Ly, Lz = (800., 4.) # units = 1km

# Create bases and domain
y_basis = de.Fourier('y', 512, interval=(-Ly, Ly), dealias=3/2)
z_basis = de.Chebyshev('z', 256, interval=(0, Lz), dealias=3/2)
domain = de.Domain([y_basis, z_basis], grid_dtype=np.float64)

ε = 0.03
y = domain.grid(0)
z = domain.grid(1)
Y = domain.new_field()
Y['g'] = Ly*(y/Ly - (2/np.pi)*np.arctan(ε*np.tan(np.pi*y/(2*Ly))))/(1-ε)

Amp, ay, az, σy, σz = 0.001, 0, 3.8, 30, 0.2
Fy = domain.new_field()
Fy['g'] = Amp*np.exp(-((y-ay)/σy)**2)*(1+erf((z-az)/σz))
# RA: Go to divergenceless forcing? Needs to be below surface.

Ω = 2*np.pi/24 # units = 1 hour
f = 2*Ω
β = f/6300 # earth radius = 6300km

B = domain.new_field()
Btop = 80
α  = 0.6
B['g'] = Btop*np.exp((z-Lz)/α)
Bbot   = Btop*np.exp(-Lz/α)

DBref = domain.new_field()
DBref.meta['y']['constant'] = True
DBref['g'] = Btop*(z/Lz)**6   #RQ: This isn't that close, is it an
                              #    issue?


# 2D Boussinesq hydrodynamics
problem = de.IVP(domain, variables=['p','b','u','v','w','bz','uz','ζ'])
problem.meta[:]['z']['dirichlet'] = True

# problem.parameters['κz']  = 1e-6  #RA: Equivalent 2.77e-4 m2s-1
# problem.parameters['νz']  = 5e-6  #RA: Equivalent 1.38e-3 m2s-1
problem.parameters['κy']  = 1e-4  #RA: Equivalent 2.77e-2 m2s-1
problem.parameters['νy']  = 5e-4  #RA: Equivalent 1.38e-1 m2s-1
                                  #RQ: Why pick Pr = 5 - better
                                  #    numerically?
                                  #RQ: Vertical kappa/nu seem 2 orders
                                  #    of magnitude too strong?
problem.parameters['κz']  = 1e-7  #RA: Equivalent 2.77e-5 m2s-1
problem.parameters['νz']  = 5e-7  #RA: Equivalent 1.38e-4 m2s-1
problem.parameters['f']  = f
problem.parameters['ω']  = Ω/10
problem.parameters['βY'] = β*Y
problem.parameters['DB'] = DBref
problem.parameters['Fy'] =  Fy
problem.parameters['Bbot'] = Bbot
problem.parameters['Btop'] = Btop
problem.add_equation("dy(v) + dz(w) = 0")
problem.add_equation("dt(b)       -      κy*d(b,y=2) - κz*dz(bz)     + w*DB =        -(v*dy(b) + w*(bz-DB))")
problem.add_equation("dt(u) + f*w -      νy*d(u,y=2) - νz*dz(uz)            =  βY*v  -(v*dy(u) + w*uz)")
problem.add_equation("dt(v)       - (νy-νz)*d(v,y=2) + νz*dz(ζ) + dy(p)     = -βY*u  + ζ*w + Fy*sin(ω*t)")
problem.add_equation("dt(w) - f*u - (νy-νz)*d(w,y=2) - νz*dy(ζ) + dz(p) - b =        - ζ*v")
problem.add_equation("bz - dz(b) = 0")
problem.add_equation("uz - dz(u) = 0")
problem.add_equation("ζ + dz(v) - dy(w) = 0")
# RA: Advection terms are expressed using ζ by transforming pressure
# P -> P - 1/2*w^2 - 1/2*v^2
problem.add_bc("left(b) = Bbot")
problem.add_bc("left(u) = 0")
problem.add_bc("left(v) = 0")
problem.add_bc("left(w) = 0")
problem.add_bc("right(b) = Btop")
problem.add_bc("right(uz) = 0")
problem.add_bc("right(ζ) = 0")
problem.add_bc("right(w) = 0", condition="(ny != 0)") #RQ: Why need ny !=0
problem.add_bc("right(p) = 0", condition="(ny == 0)")

# Build solver
solver = problem.build_solver(de.timesteppers.RK222)
logger.info('Solver built')

# Initial conditions
b = solver.state['b']
p = solver.state['p']
bz = solver.state['bz']

b['g'] = B['g']
p['g'] = α*(B['g']-Btop)
b.differentiate('z', out=bz)

# Integration parameters
solver.stop_sim_time = np.inf
solver.stop_wall_time = 4*60*60.
solver.stop_iteration = np.inf

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', iter=60, max_writes=50)
snapshots.add_task("b")
snapshots.add_task("bz")
snapshots.add_task("p")
snapshots.add_task("u")
snapshots.add_task("uz")
snapshots.add_task("v")
snapshots.add_task("w")
snapshots.add_task("ζ")

dt=0.5
# CFL
#CFL = flow_tools.CFL(solver, initial_dt=0.1, cadence=10, safety=1,
#                     max_change=1.5, min_change=0.5, max_dt=0.1)
#CFL.add_velocities(('v','w'))

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("abs(ζ)/f", name='Ro')
# This isn't a good measure of non-linearity (except at the Equator?)
# because we would expect this to be large!

# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.ok:
        #        dt = CFL.compute_dt()
        solver.step(dt)
        if (solver.iteration-1) % 10 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            logger.info('Max Ro = %f' %flow.max('Ro'))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))
