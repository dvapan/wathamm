import numpy as np
import logging
from model import count_points
import solvers.simplex as simplex
import solvers.solve_constractions_cone as constr_cone
import solvers.iterate_simplex as iterate_simplex

from scipy.sparse import coo_matrix


def count(eps=0.01):
   
    outx = np.loadtxt("xdata.txt")

    monos, rhs, ct = count_points(30,30,poly_coeff=None)

    ct = np.hstack([ct,ct])
    
    ones = np.ones((len(monos),1))

    A1 = np.hstack([monos, ones])
    A2 = np.hstack([-monos, ones])
    task_A = np.vstack([A1,A2])

    task_rhs = np.hstack([rhs,-rhs])

    stime = time.time()

    resd = abs(np.dot(task_A,outx) - task_rhs)
    
    idx = resd.argsort()

    logging.info(f"worst residuals: {resd[idx[::-1]]}")
    logging.info(f"worst residuals: {ct[idx[::-1]]}")

if __name__ == "__main__":
    import time
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H-%M-%S')
    stime = time.time()
    count()
    t = time.time() - stime
    logging.debug("total time {} seconds".format(t) )
