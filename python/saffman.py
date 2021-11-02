import numpy as np
import scipy as sp
from scipy.special import struve, yn

from function_generator import FunctionGenerator

################################################################################
# this module creates a fast, jittable function to evaluate the greens-function
# for any given Saffman-Delbruck number

# slow version of the unscaled greens function
def _slow_greens(x):
    H = struve(0, x)
    Y = yn(0, x)
    return 0.25*(H-Y)
# fast version of the unscaled greens function
gf = FunctionGenerator(_slow_greens, a=1e-30, b=100, tol=1e-12)
# jit-compilable core of the fast version
_gf = gf.get_base_function()

# slow version of the derivative of the unscaled greens functions
def _slow_gp(x):
    H = struve(-1, x)
    Y = yn(1, x)
    return 0.25*(H+Y)
# fast version of the derivative
gp = FunctionGenerator(_slow_gp, a=1e-30, b=100, tol=1e-12)
# jit-compilable core of the fast version
_gp = gp.get_base_function()


