import numpy as np
import scipy as sc
from scipy.sparse import csr_matrix
from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray

from poly import mvmonos, powers

cff_cnt = [10, 10]

d = 0.5
delta = 0.2
rho = 1000
K = 2030
E = 200000
c2 = 1/(rho/K + (rho*d)/(delta*E))


def mvmonoss(x, powers, shift_ind, cff_cnt, diff=None):
    lzeros = sum((cff_cnt[i] for i in range(shift_ind)))
    rzeros = sum((cff_cnt[i] for i in range(shift_ind + 1, len(cff_cnt))))
    monos = mvmonos(x, powers, diff)
    lzeros = np.zeros((len(x), lzeros))
    rzeros = np.zeros((len(x), rzeros))
    return np.hstack([lzeros, monos, rzeros])


def nodes(*grid_base):
    """
    Make list of nodes from given space of points
    """
    grid = np.meshgrid(*grid_base)
    grid_flat = map(lambda x: x.flatten(), grid)
    return np.vstack(list(grid_flat)).T


def make_id(i,j):
    return i*xreg + j

def shifted(cffs,shift):
    pcount = len(cffs)
    psize = len(cffs[0])
    lzeros = sc.zeros((pcount, psize * shift))
    rzeros = sc.zeros((pcount, (max_reg - shift-1) * psize))
    cffs = sc.hstack([lzeros,cffs,rzeros])
    return cffs

def eq1(*grid_base):
    in_pts = nodes(*grid_base)

    dpdx = mvmonoss(in_pts, powers(3, 2), 0, cff_cnt, [1, 0])
    dvdt = mvmonoss(in_pts, powers(3, 2), 1, cff_cnt, [0, 1])

    monos = dpdx + rho*dvdt

    rhs = np.full(len(monos), 0)
    cff = np.full(len(monos), 1)
    return monos, rhs, cff

def eq2(*grid_base):
    in_pts = nodes(*grid_base)

    dpdt = mvmonoss(in_pts, powers(3, 2), 0, cff_cnt, [0, 1])
    dvdt = mvmonoss(in_pts, powers(3, 2), 1, cff_cnt, [1, 0])

    
    monos = dpdx - c2* rho*dvdt

    rhs = np.full(len(monos), 0)
    cff = np.full(len(monos), 1)
    return monos, rhs, cff
