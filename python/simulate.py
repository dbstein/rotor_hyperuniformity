import numpy as np
import scipy as sp
import scipy.integrate
import sys

# import kernels
from kernels import apply_euler_grad, apply_euler_grad_steric, compute_euler_hamiltonian
from kernels import apply_qsg_grad, apply_qsg_grad_steric, compute_qsg_hamiltonian
from kernels import get_SD_functions
# convenience functions
from kernels import divide_state
# import data generation functions
from generate_initial_data import get_initial_data

################################################################################
# parameters

N = 1000             # number of particles
r = 0.3              # radius of ensemble
kernel = 'euler'     # kernel to use ('euler', 'qsg', 'sd')
sd_number = 0.5      # saffman-delbruck number (only used if kernel=='sd')
steric = True        # whether to include steric interactions
rSteric = 0.01       # radius where steric interactions turn on (only used if steric==True)
wSteric = 1000000.0  # strength of steric interactions (only used if steric==True)
random_seed = -1     # random seed for setting initial conditions (set to -1 to not set a seed)
omega1 = 1           # size of circulation for first half of the particles
omega2 = 1           # size of ciculation for second half of the particles
max_time = 0.05      # time to timestep to
error_tol = 1.0e-6   # error tolerance for DOP853 Timestepper

################################################################################
# construct vector for circulations

halfN = N // 2
omega = np.repeat([omega1, omega2], halfN)

################################################################################
# setup kernels

if kernel == 'euler':
	def compute_hamiltonian(X):
		return compute_euler_hamiltonian(X, omega)
	if steric:
		def apply_grad(X):
			return apply_euler_grad_steric(X, omega, rSteric, wSteric)
	else:
		def apply_grad(X):
			return apply_euler_grad(X, omega)
elif kernel == 'qsg':
	def compute_hamiltonian(X):
		return compute_qsg_hamiltonian(X, omega)
	if steric:
		def apply_grad(X):
			return apply_qsg_grad_steric(X, omega, rSteric, wSteric)
	else:
		def apply_grad(X):
			return apply_qsg_grad(X, omega)
elif kernel == 'sd':
	apply_sd_grad, apply_sd_grad_steric, apply_sd_kernel, compute_sd_hamiltonian \
		= get_SD_functions(sd_number)
	def compute_hamiltonian(X):
		return compute_sd_hamiltonian(X, omega)
	if steric:
		def apply_grad(X):
			return apply_sd_grad_steric(X, omega, rSteric, wSteric)
	else:
		def apply_grad(X):
			return apply_sd_grad(X, omega)
else:
	raise ValueError("'kernel' must be one of 'euler', 'qsg', or 'sd'")

################################################################################
# generate initial data

initial_x, initial_y = get_initial_data(N, r, random_seed)
X = np.concatenate([initial_x, initial_y])
H0 = compute_hamiltonian(X) # initial Hamiltonian

################################################################################
# timestep

def fun(t, X):
    return apply_grad(X)
stepper = sp.integrate.DOP853(fun, 0.0, X, max_time, rtol=error_tol, atol=error_tol)

Hs = []
ts = []
X0 = []
Y0 = []
while stepper.status != 'finished':
	stepper.step()
	H = compute_hamiltonian(stepper.y)
	HE = np.abs(H-H0)/np.abs(H0)
	print('... t = {:0.8f}'.format(stepper.t), 'H err is: {:0.2e}'.format(HE), end='\r')
	sys.stdout.flush()
	Hs.append(H)
	ts.append(stepper.t)
	X0.append(stepper.y[0])
	Y0.append(stepper.y[N])

################################################################################
# analysis

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ion()
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath,amssymb}'
from analyze import compute_structure_factor

# extract the state
X = stepper.y
x, y = divide_state(X)

# plot the initial and final configurations
fig, [ax0, ax1] = plt.subplots(1, 2, figsize=(12,6))
ax0.scatter(initial_x, initial_y, color='blue')
ax1.scatter(x, y, color='red')
ax0.set_title('Initial configuration')
ax1.set_title('Final state')
for ax in [ax0, ax1]:
	ax.set_aspect('equal')

# compute the structure factor
Sq, rSq, rv = compute_structure_factor(x, y, r)

# plot the structure factor
fig, [ax0, ax1] = plt.subplots(1, 2, figsize=(12,6))
ax0.imshow(Sq)
ax0.set(xticks=[], yticks=[], xticklabels=[], yticklabels=[])
ax0.set(title='Structure factor')
ax1.plot(rv, rSq, color='black', linewidth=3)
ax1.set(xscale='log', yscale='log')
ax1.set(xlabel=r'$q$', ylabel=r'$\langle S(q)\rangle$')
ax1.set(title='Radially averaged structure factor')

# plot a single trajectory and the error in H
fig, [ax0, ax1] = plt.subplots(1, 2, figsize=(12,6))
ax0.plot(X0, Y0, color='black')
ax0.set(xlabel=r'$x$', ylabel=r'$y$')
ax0.set(title='Single-rotor trajectory')
HE = np.abs(np.array(Hs)-H0)/np.abs(H0)
ax1.plot(ts, HE, color='black', linewidth=3)
ax1.set(title='Relative error in the Hamiltonian')
ax1.set(xlabel=r'$t$')

# issue a message to the user if steric interactions are on
if steric:
	print("\nNote: since Steric interactions are on, the Hamiltonian will not be conserved!")
