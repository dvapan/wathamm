import numpy as np
import scipy as sc
from scipy.sparse import csr_matrix
from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray

from poly import mvmonos, powers


from constants import *


ppwrs = powers(max_poly_degree, 2)
psize = len(ppwrs)


cff_cnt = [psize,psize]


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

def eq1_left(pts, p_cf=None):
    dpdx = mvmonoss(pts, ppwrs, 0, cff_cnt, [0, 1])
    if p_cf is not None:
        dpdx = dpdx.dot(p_cf)
    return dpdx

def eq1_right(pts, v_cf=None):
    v = mvmonoss(pts, ppwrs, 1, cff_cnt, [0, 0])
    dvdt = mvmonoss(pts, ppwrs, 1, cff_cnt, [1, 0])
    if v_cf is not None:
        v = v.dot(v_cf)
        dvdt = dvdt.dot(v_cf)
    return -rho*(dvdt + lmd*v/(2*d))

def eq2_left(pts, p_cf=None):
    dpdt = mvmonoss(pts, ppwrs, 0, cff_cnt, [1, 0])
    if p_cf is not None:
        dpdt = dpdt.dot(p_cf)
    return dpdt

def eq2_right(pts, v_cf=None):
    dvdx = mvmonoss(pts, ppwrs, 1, cff_cnt, [0, 1])
    if v_cf is not None:
        dvdx = dvdx.dot(v_cf)

    return c2*rho*dvdx

def eq1(*grid_base):
    in_pts = nodes(*grid_base)

    monos = eq1_left(in_pts) - eq1_right(in_pts)

    rhs = np.full(len(monos), 0)
    cff = np.full(len(monos), 100)
    return monos, rhs, cff


def eq2(*grid_base):
    in_pts = nodes(*grid_base)
    
    monos = eq2_left(in_pts) - eq2_right(in_pts)

    rhs = np.full(len(monos), 0)
    cff = np.full(len(monos), 10000)
    return monos, rhs, cff

def boundary_val(val,eps, ind, *grid_base):
    """
    Boundary points
    """
    sb_pts_x0 = nodes(*grid_base)
    monos = mvmonoss(sb_pts_x0, ppwrs, ind, cff_cnt)
    rhs = np.full(len(monos), val)
    cff = np.full(len(monos), eps)
    return monos, rhs, cff

def boundary_fnc(fnc,eps, ind, *grid_base):
    """
    Boundary points
    """
    sb_pts_x0 = nodes(*grid_base)
    monos = mvmonoss(sb_pts_x0, ppwrs, ind, cff_cnt)
    rhs = np.apply_along_axis(fnc, 1, sb_pts_x0)
    cff = np.full(len(monos), eps)
    return monos, rhs, cff



def shifted(cffs,shift):
    pcount = len(cffs)
    psize = len(cffs[0])
    lzeros = np.zeros((pcount, psize * shift))
    rzeros = np.zeros((pcount, (max_reg - shift-1) * psize))
    cffs = np.hstack([lzeros,cffs,rzeros])
    return cffs



def vs(pts):
    t,x = pts
    if 1 - 1/timeclose*t > 0:
        return 1 - 1/timeclose*t
    else:
        return 0
        

    
if __name__ == "__main__":
    monos = []
    rhs = []
    cff = []
    for i in range(treg):
        for j in range(xreg):
            conditions = (eq1(T_part[i], X_part[j]),
                          eq2(T_part[i], X_part[j]))

            ind = make_id(i, j)
            for m, r, c in conditions:
                m = shifted(m, ind)
                monos.append(m)
                rhs.append(r)
                cff.append(c)

    for i in range(treg):    
        m,r,c = boundary_fnc(vs,0.001, 1, T_part[i],X_part[xreg - 1][-1])
        # m,r,c = boundary_val(v0,0.01, 1, T_part[i],X_part[xreg - 1][-1])
        ind = make_id(i, 0)
        # m = shifted(m, ind)
        monos.append(m)
        rhs.append(r)
        cff.append(c)

    for j in range(xreg):
        m,r,c = boundary_val(p0,1000, 0, T_part[0][0], X_part[j])
        ind = make_id(0, j)
        # m = shifted(m, ind)
        monos.append(m)
        rhs.append(r)
        cff.append(c)

    for j in range(xreg):
        m,r,c = boundary_val(v0,0.001, 1, T_part[0][0], X_part[j])
        ind = make_id(0, j)
        # m = shifted(m, ind)
        monos.append(m)
        rhs.append(r)
        cff.append(c)


    A = sc.vstack(monos)

    rhs = np.hstack(rhs) 
    cff = np.hstack(cff).reshape(-1, 1)

    s = CyClpSimplex()

    s.logLevel = 3

    lp_dim = A.shape[1] + 1
    A /= cff
    rhs /= cff.reshape(-1)
    ones = np.ones_like(cff)
    A1 = np.hstack([A, ones])
    A2 = np.hstack([-A, ones])

    x = s.addVariable('x', lp_dim)

    A1 = np.matrix(A1)
    A2 = np.matrix(A2)
    nnz = np.count_nonzero(A1)+np.count_nonzero(A2)

    b1 = CyLPArray(rhs)
    b2 = CyLPArray(-rhs)

    s += A1 * x >= b1
    s += A2 * x >= b2

    s += x[lp_dim - 1] >= 0
    # s += x[lp_dim - 1] <= 1
    s.objective = x[lp_dim - 1]

    print ("TASK SIZE:")
    print ("XCOUNT:",lp_dim)
    print ("GXCOUNT:",len(rhs)+len(rhs))
    nnz = np.count_nonzero(A1)+np.count_nonzero(A2)
    aec = len(rhs)*lp_dim*2
    print("nonzeros:",nnz, aec, nnz/aec)

    print("START")
    s.primal()
    outx = s.primalVariableSolution['x']
    pc = sc.split(outx[:-1],max_reg)
    np.savetxt("test_cff", pc)
