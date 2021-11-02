import numpy as np
import numba

################################################################################
# convenience application functions

def divide_state(X):
    N = X.size // 2
    return X[:N], X[N:]
def apply_grad(X, omega, grad_looper):
    pot = np.zeros_like(X)
    X1, X2 = divide_state(X)
    P1, P2 = divide_state(pot)
    grad_looper(X1, X2, P1, P2, omega)
    return pot
def apply_grad_steric(X, omega, rSteric, wSteric, grad_looper):
    pot = np.zeros_like(X)
    X1, X2 = divide_state(X)
    P1, P2 = divide_state(pot)
    grad_looper(X1, X2, P1, P2, omega, rSteric, wSteric)
    return pot
def apply_kernel(X, omega, kernel_looper):
    X1, X2 = divide_state(X)
    pot = np.zeros_like(X1)
    kernel_looper(X1, X2, pot, omega)
    return pot
def compute_hamiltonian(X, omega, kernel):
    return np.sum(kernel(X, omega)*omega)

################################################################################
# Functions to add in Steric interactions

@numba.njit(inline='always')
def steric(r, rSteric, wSteric):
    if r < rSteric:
        idr = 1.0 / r
        fSteric = wSteric * (rSteric * idr - 1.0)
        return fSteric
    else:
        return 0.0

@numba.njit(inline='always')
def steric_core(dx, dy, r, potx, poty, i, rSteric, wSteric):
    fSteric = steric(r, rSteric, wSteric)
    potx[i] += dx*fSteric
    poty[i] += dy*fSteric

################################################################################
# Grad Kernel Looper

def get_grad_looper(grad_core):

    @numba.njit(parallel=True)
    def grad_looper(sx, sy, potx, poty, omega):
        sn = sx.shape[0]
        for i in numba.prange(sn):
            for j in range(sn):
                if i != j:
                    grad_core(sx, sy, i, j, potx, poty, omega)

    @numba.njit(parallel=True)
    def grad_looper_steric(sx, sy, potx, poty, omega, rSteric, wSteric):
        sn = sx.shape[0]
        for i in numba.prange(sn):
            for j in range(sn):
                if i != j:
                    dx, dy, r = grad_core(sx, sy, i, j, potx, poty, omega)
                    steric_core(dx, dy, r, potx, poty, i, rSteric, wSteric)

    return grad_looper, grad_looper_steric

################################################################################
# Kernel Looper

def get_kernel_looper(kernel_core):

    @numba.njit(parallel=True)
    def kernel_looper(sx, sy, pot, omega):
        sn = sx.size
        pn = pot.size
        for i in numba.prange(pn):
            for j in range(sn):
                if i != j:
                    kernel_core(sx, sy, i, j, pot, omega)

    return kernel_looper

################################################################################
# compute x/y signed distances
@numba.njit(inline='always')
def _dist_core(sx, sy, i, j):
    dx = sx[i] - sx[j]
    dy = sy[i] - sy[j]
    r = np.hypot(dx, dy)
    return dx, dy, r

################################################################################
# Numba-jitted Kernels for Euler-vortices

@numba.njit(inline='always')
def _euler_grad_core(sx, sy, i, j, potx, poty, omega):
    dx, dy, r = _dist_core(sx, sy, i, j)
    id2 = 1.0/(r*r)
    potx[i] -= dy*id2*omega[j]
    poty[i] += dx*id2*omega[j]
    return dx, dy, r
@numba.njit(inline='always')
def _euler_kernel_core(sx, sy, i, j, pot, omega):
    dx, dy, r = _dist_core(sx, sy, i, j)
    pot[i] += np.log(r)*omega[j]

_euler_grad, _euler_grad_steric = get_grad_looper(_euler_grad_core)
_euler_kernel = get_kernel_looper(_euler_kernel_core)
def apply_euler_grad(X, omega):
    return apply_grad(X, omega, _euler_grad)
def apply_euler_grad_steric(X, omega, rSteric, wSteric):
    return apply_grad_steric(X, omega, rSteric, wSteric, _euler_grad_steric)
def apply_euler_kernel(X, omega):
    return apply_kernel(X, omega, _euler_kernel)
def compute_euler_hamiltonian(X, omega):
    return compute_hamiltonian(X, omega, apply_euler_kernel)

################################################################################
# Numba-jitted Kernels for QSG-vortices

@numba.njit(inline='always')
def _qsg_grad_core(sx, sy, i, j, potx, poty, omega):
    dx, dy, r = _dist_core(sx, sy, i, j)
    ir3 = 1.0 / (r*r*r)
    potx[i] -= dy*ir3*omega[j]
    poty[i] += dx*ir3*omega[j]
    return dx, dy, r
@numba.njit(inline='always')
def _qsg_kernel_core(sx, sy, i, j, pot, omega):
    dx, dy, r = _dist_core(sx, sy, i, j)
    pot[i] += omega[j] / r

_qsg_grad, _qsg_grad_steric = get_grad_looper(_qsg_grad_core)
_qsg_kernel = get_kernel_looper(_qsg_kernel_core)
def apply_qsg_grad(X, omega):
    return apply_grad(X, omega, _qsg_grad)
def apply_qsg_grad_steric(X, omega, rSteric, wSteric):
    return apply_grad_steric(X, omega, rSteric, wSteric, _qsg_grad_steric)
def apply_qsg_kernel(X, omega):
    return apply_kernel(X, omega, _qsg_kernel)
def compute_qsg_hamiltonian(X, omega):
    return compute_hamiltonian(X, omega, apply_qsg_kernel)

################################################################################
# Numba-jitted Kernel for general Saffman-Delbruck Number

def get_SD_functions(ll):

    from saffman import _gf, _gp

    @numba.njit(inline='always')
    def _sd_grad_core(sx, sy, i, j, potx, poty, omega):
        dx, dy, r = _dist_core(sx, sy, i, j)
        pot = omega[j] * _gp(r/ll) / (r * ll)
        potx[i] -= dy*pot
        poty[i] += dx*pot
        return dx, dy, r
    @numba.njit(inline='always')
    def _sd_kernel_core(sx, sy, i, j, pot, omega):
        dx, dy, r = _dist_core(sx, sy, i, j)
        pot[i] += omega[j] * _gf(r/ll)

    _sd_grad, _sd_grad_steric = get_grad_looper(_sd_grad_core)
    _sd_kernel = get_kernel_looper(_sd_kernel_core)
    def apply_sd_grad(X, omega):
        return apply_grad(X, omega, _sd_grad)
    def apply_sd_grad_steric(X, omega, rSteric, wSteric):
        return apply_grad_steric(X, omega, rSteric, wSteric, _sd_grad_steric)
    def apply_sd_kernel(X, omega):
        return apply_kernel(X, omega, _sd_kernel)
    def compute_sd_hamiltonian(X, omega):
        return compute_hamiltonian(X, omega, apply_sd_kernel)

    return apply_sd_grad, apply_sd_grad_steric, apply_sd_kernel, compute_sd_hamiltonian




