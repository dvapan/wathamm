import numpy as np
import logging
from model import count_points
import solvers.simplex as simplex
import solvers.solve_constractions_cone as constr_cone
import solvers.iterate_simplex as iterate_simplex

from scipy.sparse import coo_matrix


def count(params, eps=0.01):
    monos, rhs, ct = count_points(7,7)

    ct = np.hstack([ct,ct])
    
    ones = np.ones((len(monos),1))

    A1 = np.hstack([monos, ones])
    A2 = np.hstack([-monos, ones])
    task_A = np.vstack([A1,A2])

    task_rhs = np.hstack([rhs,-rhs])

    stime = time.time()
    print(task_A.shape)
    outx = simplex.solve(task_A, task_rhs, ct=ct, logLevel=1)

    monos, rhs, ct = count_points(7,7,poly_coeff=outx)
    ct = np.hstack([ct,ct])
    
    ones = np.ones((len(monos),1))

    A1 = np.hstack([monos, ones])
    A2 = np.hstack([-monos, ones])
    task_A = np.vstack([A1,A2])

    task_rhs = np.hstack([rhs,-rhs])
    

    outx = simplex.solve(task_A, task_rhs, ct=ct, logLevel=1)

    np.savetxt("xdata.txt", outx)
    logging.info('start testing')
    monos, rhs, ct = count_points(20,20)

    ct = np.hstack([ct,ct])
    
    ones = np.ones((len(monos),1))

    A1 = np.hstack([monos, ones])
    A2 = np.hstack([-monos, ones])
    task_A = np.vstack([A1,A2])

    task_rhs = np.hstack([rhs,-rhs])

    stime = time.time()

    resd = abs(np.dot(task_A,outx) - task_rhs)
    
    idx = resd.argsort()

    logging.info(f"worst  residuals: {resd[idx[::-1]]}")
    np.savetxt("xdata.txt", outx)

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
