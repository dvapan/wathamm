import numpy as np
import logging
from model import count_points
import solvers.simplex as simplex
import solvers.solve_constractions_cone as constr_cone
import solvers.iterate_simplex as iterate_simplex

from scipy.sparse import coo_matrix



def count(eps=0.01):
    import os.path
    import sys
    import scipy.io as sio
    # if os.path.isfile("test_cff"):
    #     pc = np.loadtxt("test_cff")    
    # else:
    #     pc = None
    pc = None
    # ofile = sys.argv[1]

    monos, rhs, ct = count_points(poly_coeff=pc)

    ct = np.hstack([ct,ct])
    
    ones = np.ones((len(monos),1))

    A1 = np.hstack([monos, ones])
    A2 = np.hstack([-monos, ones])
    task_A = np.vstack([A1,A2])

    task_rhs = np.hstack([rhs,-rhs])

    stime = time.time()

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

#    from collections import Counter
#    outx, _, lmd = simplex.solve(task_A, task_rhs, ct=None, 
#            logLevel=1, extnd=True)
#    A = task_A[lmd != 0]
#    rhs = task_rhs[lmd != 0]
#    TA = abs(A) < 1.0e-5
#    TTA = np.any(TA,axis=1)
#    print(Counter(TTA))
#    print(np.nonzero(A[:,0]))
#    np.savetxt("task_A.txt",A,fmt="%.8g")
#    logging.debug("SAVING MTX")
#    sp_task_A = coo_matrix(A)
#    rows = sp_task_A.row
#    cols = sp_task_A.col
#    vals = sp_task_A.data
#    out = np.hstack([rows.reshape(-1, 1),
#                     cols.reshape(-1, 1),
#                     vals.reshape(-1, 1)])
#    np.savetxt('A_basis.txt', out, fmt="%.8g")
#    np.savetxt('b_basis.txt', rhs.reshape(-1,1), fmt="%.8g")
#    np.savetxt("xdata.vec", outx, fmt="%.8g")

    # outx = simplex.solve(task_A, task_rhs, ct=None, logLevel=1, mps_file=ofile)
#    outx = simplex.solve(task_A, task_rhs, ct=ct, logLevel=1)
#    outx = constr_cone.solve(task_A, task_rhs, ct=ct) #
    # outx = iterate_simplex.solve(task_A, task_rhs, ct=None, logLevel=1)

#    print("result:",outx[-1])
#    np.savetxt("xdata.vec", outx)

if __name__ == "__main__":
    # import sys
    # np.set_printoptions(threshold=sys.maxsize)
    import time
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H-%M-%S')
    stime = time.time()
    count()
    t = time.time() - stime
    logging.debug("total time {} seconds".format(t) )
