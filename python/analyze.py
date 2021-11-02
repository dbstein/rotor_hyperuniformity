import numpy as np
import numba
from finufft import nufft2d1

@numba.njit
def accumulate(inds, accumulator, number, fsq):
    for i in range(inds.size):
        accumulator[inds[i]] += fsq[i]
        number[inds[i]] += 1

def AngularAverage(Sq, nAvg, r):
    """
    Compute an annular average of Sq over nAvg bins of r
    """
    r, fsq = r.ravel(), Sq.ravel()
    rdiv = r.max() / nAvg
    accumulator = np.zeros(nAvg, dtype=float)
    number = np.zeros(nAvg, dtype=int)
    inds = (r / rdiv).astype(int)
    inds[inds == nAvg] -= 1 # fix up the max value
    accumulate(inds, accumulator, number, fsq)
    # calculate the average and return
    return accumulator / number, np.arange(nAvg)*rdiv

def screen(xs, ys, xbounds, ybounds):
    """
    Keep only those points xs/ys/ws that lie with xbounds/ybounds
    """
    good_x = np.logical_and(xs >= xbounds[0], xs < xbounds[1])
    good_y = np.logical_and(ys >= ybounds[0], ys < ybounds[1])
    good = np.logical_and(good_x, good_y)
    xs = xs[good]
    ys = ys[good]
    return xs, ys

def compute_structure_factor(x, y, r, NF=512):
    # reduce to a fully covered region of space
    extent = 2*r/3.0
    xs, ys = screen(x, y, [-extent, extent], [-extent, extent])
    ws = np.ones_like(xs)
    # scale to [0, 2*pi]
    xs = (xs + extent)  * (2*np.pi/(2*extent))
    ys = (ys + extent)  * (2*np.pi/(2*extent))
    # compute structure factor via NUFFT (onto NF frequencies)
    out = nufft2d1(xs, ys, ws.astype(complex), NF, eps=1e-14, isign=-1)
    out[NF//2, NF//2] = 0.0
    Sq = np.abs(out)**2
    Sq /= xs.size
    # compute frequencies
    h = 2*extent/NF
    qv = np.fft.fftfreq(NF, h/(2*np.pi))
    qx, qy = np.meshgrid(qv, qv, indexing='ij')
    qm = np.hypot(qx, qy)
    qm = np.fft.fftshift(qm)
    # average radially
    rSq, rv = AngularAverage(Sq, int(NF/2), qm)
    # return the structure factor, radially averaged structure factor
    # and the modes associated with the bins for radially averaging
    return Sq, rSq, rv
