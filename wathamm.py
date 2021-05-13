import numpy as np
import scipy as sc
from scipy.sparse import csr_matrix
from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray

from poly import mvmonos, powers


from constants import *


cff_cnt = [10, 10]


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
    dvdx = mvmonoss(in_pts, powers(3, 2), 1, cff_cnt, [1, 0])

    
    monos = dpdt - c2*rho*dvdx

    rhs = np.full(len(monos), 0)
    cff = np.full(len(monos), 1)
    return monos, rhs, cff

def boundary_val(val, ind, *grid_base):
    """
    Boundary points
    """
    sb_pts_x0 = nodes(*grid_base)
    monos = mvmonoss(sb_pts_x0, powers(3, 2), ind, cff_cnt)
    rhs = np.full(len(monos), val)
    cff = np.full(len(monos), 1)
    return monos, rhs, cff

def boundary_fnc(fnc, ind, *grid_base):
    """
    Boundary points
    """
    sb_pts_x0 = nodes(*grid_base)
    monos = mvmonoss(sb_pts_x0, powers(3, 2), ind, cff_cnt)
    rhs = np.apply_along_axis(fnc, 1, sb_pts_x0)
    # print(rhs)
    cff = np.full(len(monos), 1)
    return monos, rhs, cff



def shifted(cffs,shift):
    pcount = len(cffs)
    psize = len(cffs[0])
    lzeros = np.zeros((pcount, psize * shift))
    rzeros = np.zeros((pcount, (max_reg - shift-1) * psize))
    cffs = np.hstack([lzeros,cffs,rzeros])
    return cffs



    
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


def vs(pts):
    t,x = pts
    if t < timeclose:
        return v0*(timeclose - t)
    else:
        return 0


for i in range(treg):    
    m,r,c = boundary_fnc(vs, 1, T_part[i],X_part[xreg - 1][-1])
    ind = make_id(i, 0)
    m = shifted(m, ind)
    monos.append(m)
    rhs.append(r)
    cff.append(c)

for j in range(xreg):
    m,r,c = boundary_val(p0, 0, T_part[0][0], X_part[i])
    ind = make_id(0, j)
    m = shifted(m, ind)
    monos.append(m)
    rhs.append(r)
    cff.append(c)



# def betw_blocks(pws, gind,dind, pind, R=None):
#     i, j = gind
#     di,dj = dind
#     if di > 0:
#         Ti1 = -1
#         Ti2 = 0
#     else:
#         Ti1 = 0
#         Ti2 = -1
#     ind = make_id(i, j)
#     if R is None:
#         grid_base = T_part[i][Ti1], X_part[j]
#     else:
#         grid_base = T_part[i][Ti1], X_part[j],R
#     ptr_bnd = nodes(*grid_base)
#     val = mvmonoss(ptr_bnd, pws, pind, cff_cnt)
#     val = shifted(val, ind)

#     ni, nj = i+di, j
#     indn = make_id(ni, nj)
#     if R is None:
#         grid_basen = T_part[ni][Ti2], X_part[nj]
#     else:
#         grid_basen = T_part[ni][Ti2], X_part[nj], R
#     ptr_bndn = nodes(*grid_basen)
#     valn = mvmonoss(ptr_bndn, pws, pind, cff_cnt)
#     valn = shifted(valn, indn)

#     monos = []

#     monos.append(valn - val)

#     if dj > 0:
#         Tj1 = -1
#         Tj2 = 0
#     else:
#         Tj1 = 0
#         Tj2 = -1
#     if R is None:
#         grid_base = T_part[i], X_part[j][Tj1]
#     else:
#         grid_base = T_part[i], X_part[j][Tj1],R
#     ptr_bnd = nodes(*grid_base)
#     val = mvmonoss(ptr_bnd, pws, pind, cff_cnt)
#     val = shifted(val, ind)

#     ni, nj = i, j+dj
#     indn = make_id(ni, nj)
#     if R is None:
#         grid_basen = T_part[ni], X_part[nj][Tj2]
#     else:
#         grid_basen = T_part[ni], X_part[nj][Tj2], R

#     ptr_bndn = nodes(*grid_basen)
#     valn = mvmonoss(ptr_bndn, pws, pind, cff_cnt)
#     valn = shifted(valn, indn)

#     monos.append(valn - val)
#     monos = np.vstack(monos)
#     rhs = np.full(len(monos), 0)
#     cff = np.full(len(monos), 1)
#     return monos, rhs, cff


# conditions = []
# for i in range(treg - 1):
#     for j in range(xreg - 1):
#         conditions.append(betw_blocks(powers(3, 2), (i, j),(1,1), 0))
# for i in range(1,treg):
#     for j in range(1, xreg):
#         conditions.append(betw_blocks(powers(3, 2), (i, j),(-1,-1), 1))

    

A = sc.vstack(monos)

rhs = np.hstack(rhs)
cff = np.hstack(cff).reshape(-1, 1)

s = CyClpSimplex()
lp_dim = A.shape[1] + 1

A1 = np.hstack([A, cff])
A2 = np.hstack([-A, cff])

x = s.addVariable('x', lp_dim)
A1 = np.matrix(A1)
A2 = np.matrix(A2)
nnz = np.count_nonzero(A1)+np.count_nonzero(A2)

b1 = CyLPArray(rhs)
b2 = CyLPArray(-rhs)

s += A1 * x >= b1
s += A2 * x >= b2

s += x[lp_dim - 1] >= 0
s += x[lp_dim - 1] <= 10**6
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
