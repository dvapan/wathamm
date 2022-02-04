import numpy as np
from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray

import logging

import pandas as pd

def solve(A, rhs, ct=None, logLevel=0, extnd=False, basis=False, mps_file=None):
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
    if not mps_file is None:
        s.writeMps(mps_file)
        return None
    logging.debug(f"TASK SIZE XCOUNT: {lp_dim} GXCOUNT: {len(rhs)}")

    s.primal()
    k = list(s.primalConstraintSolution.keys())[0]
    k2 =list(s.dualConstraintSolution.keys())[0]
    q = s.dualConstraintSolution[k2]
    logging.debug(f"{s.getStatusString()} objective: {s.objectiveValue}")

    if extnd and not basis:
        return s.primalVariableSolution['x'],s.primalConstraintSolution[k],s.dualConstraintSolution[k2]
    elif not extnd and basis:
        basis = s.getBasisStatus()
        return s.primalVariableSolution['x'],*basis
    else:
        return s.primalVariableSolution['x']
