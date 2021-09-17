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
    return dpdx

def eq1_right(pts, cf=None):
    v = mvmonoss(pts, ppwrs, 1, cff_cnt, [0, 0])
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

    return -c2*rho*dvdx

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
        cff = np.full(len(monos), 100000)
    else:
        left_bal = np.abs(eq2_left(in_pts, cf))
        right_bal = np.abs(eq2_right(in_pts, cf))
        cff = 0.01*np.maximum(left_bal, right_bal)
        print("eq2")
        print(cff)

    return monos, rhs, cff, ["eq2"]*len(monos)        

def make_cnst_name(first, second=None):
    cnst_name = first
    if second != None:
        cnst_name += "_" + name
    return cnst_name
    

def boundary_val(val,eps, ind, *grid_base, name=None):
    """
    Boundary points
    """
    sb_pts_x0 = nodes(*grid_base)
    monos = mvmonoss(sb_pts_x0, ppwrs, ind, cff_cnt)
    rhs = np.full(len(monos), val)
    cff = np.full(len(monos), eps)
    return monos, rhs, cff, [make_cnst_name("bnd_val",name)]*len(monos)

def boundary_fnc(fnc,eps, ind,  *grid_base, name=None):
    """
    Boundary points
    """
    sb_pts_x0 = nodes(*grid_base)
    monos = mvmonoss(sb_pts_x0, ppwrs, ind, cff_cnt)
    rhs = np.apply_along_axis(fnc, 1, sb_pts_x0)
    print(rhs)
    cff = np.full(len(monos), eps)
    return monos, rhs, cff, [make_cnst_name("bnd_fnc",name)]*len(monos)        


def betw_blocks(pws, gind,dind, pind, eps, name=None):
    i, j = gind
    di,dj = dind
    ind = make_id(i, j)
    monos = []

    if i < treg - 1:
        grid_base = T_part[i][-1], X_part[j]
        ptr_bnd = nodes(*grid_base)
        val = mvmonoss(ptr_bnd, pws, pind, cff_cnt)
        val = shifted(val, ind)

        ni, nj = i+di, j
        indn = make_id(ni, nj)
        grid_basen = T_part[ni][0], X_part[nj]
        ptr_bndn = nodes(*grid_basen)
        valn = mvmonoss(ptr_bndn, pws, pind, cff_cnt)
        valn = shifted(valn, indn)

        monos.append(valn - val)
        monos.append(val - valn)
    if j < xreg - 1:
        grid_base = T_part[i], X_part[j][-1]
        ptr_bnd = nodes(*grid_base)
        val = mvmonoss(ptr_bnd, pws, pind, cff_cnt)
        val = shifted(val, ind)

        ni, nj = i, j+dj
        indn = make_id(ni, nj)
        grid_basen = T_part[ni], X_part[nj][0]
    
        ptr_bndn = nodes(*grid_basen)
        valn = mvmonoss(ptr_bndn, pws, pind, cff_cnt)
        valn = shifted(valn, indn)

        monos.append(valn - val)
        monos.append(val - valn)
    monos = np.vstack(monos)
    rhs = np.full(len(monos), 0)
    cff = np.full(len(monos), eps)
    return monos, rhs, cff, [make_cnst_name("betw_blocks",name)]*len(monos)        



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

def ps(pts):
    t,x = pts
    return p0 - rho*v0*lmd/(2*d)*x

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
        m,r,c,t = boundary_fnc(vs,0.01, 1, T_part[i],X_part[xreg - 1][-1])
        ind = make_id(i, xreg-1)
        m = shifted(m, ind)
        monos.append(m)
        rhs.append(r)
        cff.append(c)
        cnst_type.append(t)
        
    for j in range(xreg):
        m,r,c,t = boundary_fnc(ps,10000, 0, T_part[0][0], X_part[j])
#        m,r,c,t = boundary_val(p0,100000, 0, T_part[0][0], X_part[j])

        ind = make_id(0, j)
        m = shifted(m, ind)
        monos.append(m)
        rhs.append(r)
        cff.append(c)
        cnst_type.append(t)

    for i in range(treg):
        m,r,c,t = boundary_val(p0,10000, 0, T_part[i], X_part[0][0])
        ind = make_id(i, 0)
        m = shifted(m, ind)
        monos.append(m)
        rhs.append(r)
        cff.append(c)
        cnst_type.append(t)

        
    for j in range(xreg):
        m,r,c,t = boundary_val(v0,0.01, 1, T_part[0][0], X_part[j])
        ind = make_id(0, j)
        m = shifted(m, ind)
        monos.append(m)
        rhs.append(r)
        cff.append(c)
        cnst_type.append(t)


        
    # conditions = []
    # for i in range(treg):
    #     for j in range(xreg):
    #         if i < treg - 1 or j < xreg - 1:
    #             #pressure connect blocks
    #             conditions.append(betw_blocks(ppwrs, (i, j),(1,1), 0, 10000))
    #             #velocity connect blocks
    #             conditions.append(betw_blocks(ppwrs, (i, j),(1,1), 1, 0.01))
    # for m, r, c,t in conditions:
    #     monos.append(m)
    #     rhs.append(r)
    #     cff.append(c)
    #     cnst_type.append(t)

    monos = np.vstack(monos)
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
    logging.debug(f"TASK SIZE XCOUNT: {lp_dim} GXCOUNT: {len(rhs)}")

    s.primal()
    k = list(s.primalConstraintSolution.keys())[0]
    k2 =list(s.dualConstraintSolution.keys())[0]
    q = s.dualConstraintSolution[k2]
    logging.debug(f"{s.getStatusString()} objective: {s.objectiveValue}")
    logging.debug(f"nonzeros rhs: {np.count_nonzero(s.primalConstraintSolution[k])}")
    logging.debug(f"nonzeros dual: {np.count_nonzero(s.dualConstraintSolution[k2])}")

    if ct is not None:
        data = {
            "type":ct,
            "resd":s.primalConstraintSolution[k],
            "dual":s.dualConstraintSolution[k2]
        }
    
        df = pd.DataFrame(data)
        df.to_csv('out.csv', header = True, index = False)

    
    return s.primalVariableSolution['x']


def count(num_cnst_add=None, eps=0.01):
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

    m1 = lp_dim*1
    num_cnst_add = m1
    fix_idx = np.any(np.vstack([ct=="bnd_fnc",ct == "bnd_val"]), axis=0)
    fixed_points = A[fix_idx][::2]
    nonfixed_points = A[~fix_idx]

    nfix_idx = np.random.choice(len(nonfixed_points), num_cnst_add, replace=False)
    nonfixed_points = nonfixed_points[nfix_idx]

    print(fixed_points.shape)
    print(nonfixed_points.shape)

    task_A = np.vstack([fixed_points,nonfixed_points])
    task_rhs = np.hstack([rhs[fix_idx][::2], rhs[nfix_idx]])

    run = True
    itnum = 0
    bs = None
    while run:
        stime = time.time()

        outx = solve_simplex(task_A, task_rhs, logLevel=1)
        t = time.time() - stime

        otkl = np.dot(A,outx) - rhs

        itnum += 1
        i = np.argmin(otkl)
        logging.info(f"iter {itnum}; {t:.2f} s")
        logging.debug(f"count otkl < 0: {len(otkl[otkl < 0])} / {len(otkl)}")
        logging.debug(f"count otkl < -{eps}: {len(otkl[otkl < -eps])} / {len(otkl)}")
        # logging.debug(f"count active constraints {len()}")
        logging.debug(f"fx: {outx[-1]} {otkl[i]}")

        if abs(np.min(otkl)) < eps:
            run = False
            break

        num_cnst_add = max(num_cnst_add, int(np.round(len(task_A)*0.05)))
        num_cnst_add = min(num_cnst_add, len(otkl[otkl < -eps]))
        logging.debug(f"num_cnst_add: {num_cnst_add}")


        worst_A = A[otkl.argsort()][:num_cnst_add]
        worst_rhs = rhs[otkl.argsort()][:num_cnst_add]

        otkl = np.dot(task_A,outx) - task_rhs

        
        nact_A = task_A[abs(otkl) >= eps]
        nact_rhs = task_rhs[abs(otkl) >= eps]
        act_A = task_A[abs(otkl) < eps]
        act_rhs = task_rhs[abs(otkl) < eps]
        print(nact_A.shape)
        print(act_A.shape)
        q = np.dot(nact_A[:,:-1], act_A[:,:-1].T)
        print(q.shape)
        w1 = np.all(q >= eps, axis = 1)
        w2 = np.all(q <= eps, axis = 1)

        w = w1 | w2
        w = ~w

        la = len(task_A)
        task_A = np.vstack([act_A,nact_A[w]])
        task_rhs = np.hstack([act_rhs,nact_rhs[w]])
        dl = la - len(task_A)
        logging.debug(f" filtered: {dl} constraints")


        task_A = np.vstack([task_A, worst_A])
        task_rhs = np.hstack([task_rhs, worst_rhs])

        # otkl = np.dot(task_A,outx) - task_rhs

        
        

    ofile += f"p{max_poly_degree}"
    np.savetxt(ofile, outx)
    print(outx)


def count_base(num_cnst_add=None, eps=0.01):
    import os.path
    import sys
    # if os.path.isfile("test_cff"):
    #     pc = np.loadtxt("test_cff")    
    # else:
    #     pc = None
    pc = None
    ofile = sys.argv[1]

    monos, rhs, ct = count_points(poly_coeff=pc)
    num_cnst_add = len(monos[0])*4
    print("###############",num_cnst_add)


    ct = np.hstack([ct,ct])
    
    lp_dim = monos.shape[1] + 1
    ones = np.ones((len(monos),1))

    
    A1 = np.hstack([monos, ones])
    A2 = np.hstack([-monos, ones])
    task_A = np.vstack([A1,A2])

    task_rhs = np.hstack([rhs,-rhs])

    stime = time.time()

    outx = solve_simplex(task_A, task_rhs, ct=ct, logLevel=1)

    t = time.time() - stime

    ofile += f"p{max_poly_degree}xr{xreg}tr{treg}"
    np.savetxt(ofile, outx)



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
    logging.basicConfig(filename="wathamm.log",level=logging.DEBUG,
                        format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H-%M-%S')
    stime = time.time()
    count()
    t = time.time() - stime
    logging.debug("total time {} seconds".format(t) )
