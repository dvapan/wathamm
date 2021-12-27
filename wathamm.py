import numpy as np
import logging
from model import count_points
import solvers.simplex as simplex
import solvers.solve_constractions_cone as constr_cone
import solvers.iterate_simplex as iterate_simplex

from scipy.sparse import coo_matrix

def count(params, eps=0.01, pc=None):
    space = prepare_space(params)

    monos, rhs, ct = count_points(poly_coeff=pc)

    ct = np.hstack([ct,ct])
    
    ones = np.ones((len(monos),1))

    A1 = np.hstack([monos, ones])
    A2 = np.hstack([-monos, ones])
    task_A = np.vstack([A1,A2])

    task_rhs = np.hstack([rhs,-rhs])

    stime = time.time()

    outx = simplex.solve(task_A, task_rhs, ct=ct, logLevel=1)
#    outx = constr_cone.solve(task_A, task_rhs, ct=ct) #
#    outx = iterate_simplex.solve(task_A, task_rhs, ct=None, logLevel=1)

#    print("result:",outx[-1])
#    np.savetxt("xdata.vec", outx)

def save_task(task_A, task_rhs):
    logging.debug("BEGIN SAVING")

    logging.debug("SAVING MTX")
    sp_task_A = coo_matrix(task_A)
    rows = sp_task_A.row
    cols = sp_task_A.col
    vals = sp_task_A.data
    out = np.hstack([rows.reshape(-1, 1),
                     cols.reshape(-1, 1),
                     vals.reshape(-1, 1)])
    np.savetxt('cnstr.txt', out, fmt="%.8g")
    logging.debug("SAVING RHS")
    np.savetxt("rhs.txt", task_rhs.reshape(-1,1), fmt="%.8g")
    np.savetxt('cnstr_types.txt', ct, fmt="%s")

def save_basis(task_A, task_rhs):
    from collections import Counter
    outx, _, lmd = simplex.solve(task_A, task_rhs, ct=None, 
            logLevel=1, extnd=True)
    A = task_A[lmd != 0]
    rhs = task_rhs[lmd != 0]
    TA = abs(A) < 1.0e-5
    TTA = np.any(TA,axis=1)
    print(Counter(TTA))
    print(np.nonzero(A[:,0]))
    np.savetxt("task_A.txt",A,fmt="%.8g")
    logging.debug("SAVING MTX")
    sp_task_A = coo_matrix(A)
    rows = sp_task_A.row
    cols = sp_task_A.col
    vals = sp_task_A.data
    out = np.hstack([rows.reshape(-1, 1),
                     cols.reshape(-1, 1),
                     vals.reshape(-1, 1)])
    np.savetxt('A_basis.txt', out, fmt="%.8g")
    np.savetxt('b_basis.txt', rhs.reshape(-1,1), fmt="%.8g")
    np.savetxt("xdata.txt", outx, fmt="%.8g")

#xreg,treg = 1,1
#max_reg = xreg*treg
#max_poly_degree = 5
#pprx = 5                      # Точек на регион
#pprt = 5

def prepare_space(params):
    xreg = params['xreg']
    treg = params['treg']
    pprx = params['pprx']
    pprt = params['pprt']
    totalx = xreg*pprx - xreg + 1
    totalt = treg*pprt - treg + 1
    dx = length/xreg
    dt = total_time/treg
    X = np.linspace(0, length, totalx)
    T = np.linspace(0, total_time, totalt)
    X_part = list(mit.windowed(X,n=pprx,step=pprx - 1))
    T_part = list(mit.windowed(T,n=pprt,step=pprt - 1))
    return X_part, T_part

if __name__ == "__main__":
    import time
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H-%M-%S')
    params = {
            'xreg'   : 6,
            'treg'   : 12,
            'pol_deg': 3,
            'pprx'   : 6,
            'pprt'   : 6,
            }

    stime = time.time()
    count(params)
    t = time.time() - stime
    logging.debug("total time {} seconds".format(t) )
