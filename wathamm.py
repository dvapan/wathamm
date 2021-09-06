import numpy as np
import scipy as sc
from scipy.sparse import csr_matrix
from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray

from poly import mvmonos, powers

import logging


from constants import *
import pandas as pd


import matplotlib.pyplot as plt

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
    k = timeclose*total_time
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
    


def solve_simplex(A, rhs, ct=None, logLevel=0):
    s = CyClpSimplex()
    s.logLevel = logLevel
    lp_dim = A.shape[1] 

   
    x = s.addVariable('x', lp_dim)
    A = np.matrix(A)
    rhs = CyLPArray(rhs)

    s += A * x >= rhs

    s += x[lp_dim - 1] >= 0
    s.objective = x[lp_dim - 1]

    nnz = np.count_nonzero(A)
    print (f"TASK SIZE XCOUNT: {lp_dim} GXCOUNT: {len(rhs)}")

    s.primal()
    k = list(s.primalConstraintSolution.keys())
    k2 =list(s.dualConstraintSolution.keys())
    q = s.dualConstraintSolution[k2[0]]
    print(f"{s.getStatusString()} objective: {s.objectiveValue}")
    print("nonzeros rhs:",np.count_nonzero(s.primalConstraintSolution[k[0]]))
    print("nonzeros dual:",np.count_nonzero(s.dualConstraintSolution[k2[0]]))

    if ct is not None:
        data = {
            "type":ct,
            "resd":s.primalConstraintSolution[k],
            "dual":s.dualConstraintSolution[k2]
        }
    
        df = pd.DataFrame(data)
        df.to_csv('out.csv', header = True, index = False)

    
    return s.primalVariableSolution['x']


def count(num_cnst_add, eps=0.01):
    import os.path
    import sys
    # if os.path.isfile("test_cff"):
    #     pc = np.loadtxt("test_cff")    
    # else:
    #     pc = None
    pc = None
    ofile = sys.argv[1]

    monos, rhs, ct = count_points(poly_coeff=pc)

    ct = np.hstack([ct,ct])
    
    lp_dim = monos.shape[1] + 1
    ones = np.ones((len(monos),1))

    
    A1 = np.hstack([monos, ones])
    A2 = np.hstack([-monos, ones])
    A = np.vstack([A1,A2])

    rhs = np.hstack([rhs,-rhs])

    m1 = lp_dim*2

    indeces = np.random.choice(len(A), m1, replace=False)

    task_A = A[indeces]
    task_rhs = rhs[indeces]

    run = True
    itnum = 0
    while run:
        stime = time.time()
        outx = solve_simplex(task_A, task_rhs, logLevel=1)
        t = time.time() - stime

        otkl = np.dot(A,outx) - rhs

        itnum += 1
        i = np.argmin(otkl)
        logging.info(f"iter {itnum}; {t:.2f} s")
        logging.debug("count otkl < 0: {} / {}".format(len(otkl[otkl < 0]), len(otkl)))
        logging.debug("count otkl < -{}: {} / {}".format(eps,len(otkl[otkl < -eps]), len(otkl)))
        logging.debug(f"count active constraints {len()}")
        logging.debug(f"fx: {outx[-1]} {otkl[i]}")

        if abs(np.min(otkl)) < eps:
            run = False
            break
        worst_A = A[otkl.argsort()][:num_cnst_add]
        worst_rhs = rhs[otkl.argsort()][:num_cnst_add]
        task_A = np.vstack([task_A, worst_A])
        task_rhs = np.hstack([task_rhs, worst_rhs])

        nonactive_constr = otkl[abs(otkl) >= 0.001]
        active_constr = otkl[abs(otkl) < 0.001]

    ofile += f"p{max_poly_degree}nc{num_cnst_add}"
    np.savetxt(ofile, outx)
    print(outx)




def test():
    import time

    times = []
    f = open("times.dat", "w")
    f.write("n t\n")
    f.close()
    num_cnst_add = list(range(100,2000,50))
    for i,n in enumerate(num_cnst_add):   
        stime = time.time()
        count(n)
        t = time.time() - stime
        f = open("times.dat", "a")
        f.write(f"{n} {t}\n")
        f.close()
        logging.debug("total time {} seconds".format(t) )
    f.close()

if __name__ == "__main__":
    import time
    logging.basicConfig(filename='wathamm.log', level=logging.DEBUG,
                        format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H-%M-%S')
    stime = time.time()
    count(1000)
    t = time.time() - stime
    logging.debug("total time {} seconds".format(t) )
