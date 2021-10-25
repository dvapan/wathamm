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
    # if os.path.isfile("test_cff"):
    #     pc = np.loadtxt("test_cff")    
    # else:
    #     pc = None
    pc = None
    ofile = sys.argv[1]

    monos, rhs, ct = count_points(poly_coeff=pc)

    ct = np.hstack([ct,ct])
    
    ones = np.ones((len(monos),1))

    A1 = np.hstack([monos, ones])
    A2 = np.hstack([-monos, ones])
    task_A = np.vstack([A1,A2])

    task_rhs = np.hstack([rhs,-rhs])

    stime = time.time()

    # outx = simplex.solve(task_A, task_rhs, ct=None, logLevel=1)
    outx = simplex.solve(task_A, task_rhs, ct=None, logLevel=1, mps_file=ofile)
    # outx = constr_cone.solve(task_A, task_rhs, ct=ct)
    # outx = iterate_simplex.solve(task_A, task_rhs, ct=None, logLevel=1)

    #print("result:",outx[-1])
    #np.savetxt(ofile, outx)

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
