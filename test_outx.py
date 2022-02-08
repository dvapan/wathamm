import numpy as np
import logging
from model import count_points
from model import nodes,make_id
from model import ppwrs,psize,cff_cnt,mvmonoss
import solvers.simplex as simplex
import solvers.solve_constractions_cone as constr_cone
import solvers.iterate_simplex as iterate_simplex

from constants import *
from scipy.sparse import coo_matrix
import sys

def count(params, outx):
    pprx = params["pprx"]
    pprt = params["pprt"]
    xreg = params["xreg"]
    treg = params["treg"]
    is_run = True
    totalx = xreg*pprx - xreg + 1
    totalt = treg*pprt - treg + 1
    X = np.linspace(0, length, totalx)
    T = np.linspace(0, total_time, totalt)
    X_part = list(mit.windowed(X,n=pprx,step=pprx - 1))
    T_part = list(mit.windowed(T,n=pprt,step=pprt - 1))
    lxreg = X_part[0][-1] - X_part[0][0]
    ltreg = T_part[0][-1] - T_part[0][0]
    bsize = sum(cff_cnt)
    v_lst = []
    p_lst = []
    for i in range(treg):
        for j in range(xreg):
            ind = make_id(i, j, params)
            pts = nodes(T_part[i],X_part[j])
            cf = outx[ind*bsize:(ind+1)*bsize]
            tv = mvmonoss(pts, ppwrs, 1, cff_cnt, [0, 0])
            tp = mvmonoss(pts, ppwrs, 0, cff_cnt, [0, 0])
            ttv = tv.dot(cf)
            ttp = tp.dot(cf)
            v_lst.append(ttv)
            p_lst.append(ttp)
    v = np.hstack(v_lst)
    p = np.hstack(p_lst)
    print(v)
    monos, rhs, ct = count_points(params, v)

    ct = np.hstack([ct,ct])
    
    ones = np.ones((len(monos),1))

    A1 = np.hstack([monos, ones])
    A2 = np.hstack([-monos, ones])
    task_A = np.vstack([A1,A2])

    task_rhs = np.hstack([rhs,-rhs])

    stime = time.time()

    resd = task_A.dot(outx) - task_rhs
    resd = np.array(resd).flatten()
    print(resd)

    
    idx = resd.argsort()

    print(f"optimal objective: {outx[-1]}")
    print("worst residuals:")
    print(f"{resd[idx]}")
    print(f"{ct[idx]}")
    print(f"{len(resd[resd < -0.01])} / {len(resd)}")

if __name__ == "__main__":
    import time
    import argparse
    import sys

    logging.basicConfig(level=logging.DEBUG,
            format='%(asctime)s %(message)s', 
            datefmt='%Y-%m-%d %H-%M-%S')
    parser = argparse.ArgumentParser()
    parser.add_argument("--xreg", default=1,type=int)
    parser.add_argument("--treg", default=1,type=int)
    parser.add_argument("--pprx", default=7,type=int)
    parser.add_argument("--pprt", default=7,type=int)
    parser.add_argument("filename")
    args = parser.parse_args(sys.argv[1:])
    params = vars(args)

    stime = time.time()
    outx = np.loadtxt(args.filename)
    count(params, outx, a=1)
    t = time.time() - stime
    logging.debug("total time {} seconds".format(t) )
