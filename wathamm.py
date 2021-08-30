"""
Расчет мономиального базиса для переменных
"""
import numpy as np
import scipy as sc
from scipy.sparse import csr_matrix
from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray

from poly import mvmonos, powers


from constants import *
import pandas as pd

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

def eq1_left(pts, cf=None):
    dpdx = mvmonoss(pts, ppwrs, 0, cff_cnt, [0, 1])
    if cf is not None:
        dpdx = dpdx.dot(cf)
    return dpdx/1000

def eq1_right(pts, cf=None):
    v = mvmonoss(pts, ppwrs, 1, cff_cnt, [0, 0])*1000
    dvdt = mvmonoss(pts, ppwrs, 1, cff_cnt, [1, 0])
    if cf is not None:
        v = v.dot(cf)
        dvdt = dvdt.dot(cf)
    return -rho*(dvdt + lmd*v/(2*d))

def eq2_left(pts, cf=None):
    dpdt = mvmonoss(pts, ppwrs, 0, cff_cnt, [1, 0])
    if cf is not None:
        dpdt = dpdt.dot(cf)
    return dpdt

def eq2_right(pts, cf=None):
    dvdx = mvmonoss(pts, ppwrs, 1, cff_cnt, [0, 1])
    if cf is not None:
        dvdx = dvdx.dot(cf)

    return -c2*rho*dvdx/1000

def eq1(*grid_base, cf=None):
    in_pts = nodes(*grid_base)

    monos = eq1_left(in_pts) - eq1_right(in_pts)

    rhs = np.full(len(monos), 0)
    if cf is None:
        cff = np.full(len(monos), 100)
    else:
        left_bal = np.abs(eq1_left(in_pts, cf))
        right_bal = np.abs(eq1_right(in_pts, cf))
        cff = 0.01*np.maximum(left_bal, right_bal)
        print("eq1")
        print(cff)
    return monos, rhs, cff, ["eq1"]*len(monos)


def eq2(*grid_base, cf=None):
    in_pts = nodes(*grid_base)
    
    monos = eq2_left(in_pts) - eq2_right(in_pts)

    rhs = np.full(len(monos), 0)
    if cf is None:
        cff = np.full(len(monos), 10000)
    else:
        left_bal = np.abs(eq2_left(in_pts, cf))
        right_bal = np.abs(eq2_right(in_pts, cf))
        cff = 0.01*np.maximum(left_bal, right_bal)
        print("eq2")
        print(cff)

    return monos, rhs, cff, ["eq2"]*len(monos)        


def boundary_val(val,eps, ind, *grid_base):
    """
    Boundary points
    """
    sb_pts_x0 = nodes(*grid_base)
    monos = mvmonoss(sb_pts_x0, ppwrs, ind, cff_cnt)
    rhs = np.full(len(monos), val)
    cff = np.full(len(monos), eps)
    return monos, rhs, cff, ["bnd_val"]*len(monos)        

def boundary_fnc(fnc,eps, ind, *grid_base):
    """
    Boundary points
    """
    sb_pts_x0 = nodes(*grid_base)
    monos = mvmonoss(sb_pts_x0, ppwrs, ind, cff_cnt)
    rhs = np.apply_along_axis(fnc, 1, sb_pts_x0)
    cff = np.full(len(monos), eps)
    return monos, rhs, cff, ["bnd_fnc"]*len(monos)        



def shifted(cffs,shift):
    pcount = len(cffs)
    psize = len(cffs[0])
    lzeros = np.zeros((pcount, psize * shift))
    rzeros = np.zeros((pcount, (max_reg - shift-1) * psize))
    cffs = np.hstack([lzeros,cffs,rzeros])
    return cffs



def vs(pts):
    t,x = pts
    k = timeclose*time
    if 1 - 1/k*t > 0:
        return 1 - 1/k*t
    else:
        return 0
        
i = 1


def count_points(poly_coeff=None):
    monos = []
    rhs = []
    cff = []
    cnst_type = []
    for i in range(treg):
        for j in range(xreg):
            conditions = (eq1(T_part[i], X_part[j], cf=poly_coeff),
                          eq2(T_part[i], X_part[j], cf=poly_coeff))

            ind = make_id(i, j)
            for m, r, c, t in conditions:
                m = shifted(m, ind)
                monos.append(m)
                rhs.append(r)
                cff.append(c)
                cnst_type.append(t)

    for i in range(treg):    
        m,r,c,t = boundary_fnc(vs,0.03, 1, T_part[i],X_part[xreg - 1][-1])
        # m,r,c = boundary_val(v0,0.01, 1, T_part[i],X_part[xreg - 1][-1])
        ind = make_id(i, 0)
        # m = shifted(m, ind)
        monos.append(m)
        rhs.append(r)
        cff.append(c)
        cnst_type.append(t)
        
    for j in range(xreg):
        m,r,c,t = boundary_val(p0,10000, 0, T_part[0][0], X_part[j])
        ind = make_id(0, j)
        # m = shifted(m, ind)
        monos.append(m)
        rhs.append(r)
        cff.append(c)
        cnst_type.append(t)

    for i in range(treg):
        m,r,c,t = boundary_val(p0,10000, 0, T_part[i], X_part[0][0])
        ind = make_id(i, 0)
        # m = shifted(m, ind)
        monos.append(m)
        rhs.append(r)
        cff.append(c)
        cnst_type.append(t)

        
    for j in range(xreg):
        m,r,c,t = boundary_val(v0,0.03, 1, T_part[0][0], X_part[j])
        ind = make_id(0, j)
        # m = shifted(m, ind)
        monos.append(m)
        rhs.append(r)
        cff.append(c)
        cnst_type.append(t)

        
    monos = sc.vstack(monos)
    rhs = np.hstack(rhs) 
    cff = np.hstack(cff)
    rhs /= cff
    monos /= cff.reshape(-1,1)

    cnst_type = np.hstack(cnst_type)
    print(cnst_type)
    
    return monos, rhs, cnst_type
    



def count():
    import os.path
    # if os.path.isfile("test_cff"):
    #     pc = np.loadtxt("test_cff")    
    # else:
    #     pc = None
    pc = None

    monos, rhs, ct = count_points(poly_coeff=pc)

    s = CyClpSimplex()

    s.logLevel = 3

    ct = np.hstack([ct,ct])
    
    lp_dim = monos.shape[1] + 1
    ones = np.ones((len(monos),1))

    
    A1 = np.hstack([monos, ones])
    A2 = np.hstack([-monos, ones])

    x = s.addVariable('x', lp_dim)

    A = np.vstack([A1,A2])            

    A = np.matrix(A)
    nnz = np.count_nonzero(A1)+np.count_nonzero(A2)

    b = np.hstack([rhs, -rhs])
    
    b = CyLPArray(b)

    s += A * x >= b

    s += x[lp_dim - 1] >= 0
    # s += x[lp_dim - 1] <= 1
    s.objective = x[lp_dim - 1]

    print ("TASK SIZE:")
    print ("XCOUNT:",lp_dim)
    print ("GXCOUNT:",len(b))
    nnz = np.count_nonzero(A)
    aec = len(rhs)*lp_dim
    print("nonzeros:",nnz, aec, nnz/aec)
    print("START")
    s.primal()
    outx = s.primalVariableSolution['x']
    pc = sc.split(outx[:-1],max_reg)[0]
    np.savetxt("test_cff", pc)
    print(pc)

    k = list(s.primalConstraintSolution.keys())[0]
    k2 =list(s.dualConstraintSolution.keys())[0]
    
    data = {
        "type":ct,
        "resd":s.primalConstraintSolution[k],
        "dual":s.dualConstraintSolution[k2]
    }
    
    df = pd.DataFrame(data)
    df.to_csv('out.csv', header = True, index = False)

    
    
if __name__ == "__main__":
    count()
