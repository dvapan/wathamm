import numpy as np
import logging
from model import count_points
from model import nodes,make_id
from model import ppwrs,psize,cff_cnt,mvmonoss
import solvers.simplex as simplex
import solvers.solve_constractions_cone as constr_cone
import solvers.iterate_simplex as iterate_simplex

from scipy.sparse import coo_matrix

from constants import *

def test(pprx,pprt,outx):
    monos, rhs, ct,cff = count_points(pprx,pprt,pc=outx)
    ct = np.hstack([ct,ct])

    ones = np.ones((len(monos),1))

    A1 = np.hstack([monos, ones])
    A2 = np.hstack([-monos, ones])
    task_A = np.vstack([A1,A2])

    task_rhs = np.hstack([rhs,-rhs])

    resd = np.dot(task_A,outx) - task_rhs
    
    idx = resd.argsort()

    logging.info(f"worst residuals: {resd[idx]}")
    logging.info(f"worst residuals: {ct[idx]}")
    return resd[idx][0], len(resd[resd < -0.01])


def count(params, eps=0.01):
    itcnt = 0
    outx = None
    pprx = 7
    pprt = 7
    is_run = True
    v0 = None
    totalx = xreg*pprx - xreg + 1
    totalt = treg*pprt - treg + 1
    X = np.linspace(0, length, totalx)
    T = np.linspace(0, total_time, totalt)
    X_part = list(mit.windowed(X,n=pprx,step=pprx - 1))
    T_part = list(mit.windowed(T,n=pprt,step=pprt - 1))
    lxreg = X_part[0][-1] - X_part[0][0]
    ltreg = T_part[0][-1] - T_part[0][0]
    bsize = sum(cff_cnt)
    outx_old = None
    outxx = []
    f = open('v.txt','w')
    f.close()
    f = open('vu.txt','w')
    f.close()
    f = open('dv.txt','w')
    f.close()
    v_old = None
    while is_run  or itcnt == 0:
        itcnt += 1
        logging.info(f"ITER: {itcnt}")
        stime = time.time()
        cff_old = None
        is_refin = True
        pc_cff = outx
        sdcf = None
        refit = 0
        monos, rhs, ct, cff = count_points(pprx,pprt,
                pc=outx,pco=outx_old,pc_cff=pc_cff)
        print(monos.shape)
        ones = np.ones((len(monos),1))

        A1 = np.hstack([monos, ones])
        A2 = np.hstack([-monos, ones])
        task_A = np.vstack([A1,A2])

        task_rhs = np.hstack([rhs,-rhs])

#        logging.info("start saving")
#        np.savetxt("A.dat",task_A.flatten())
#        np.savetxt("b.dat",task_rhs)
#        logging.info("end saving")
        outx = simplex.solve(task_A, task_rhs, ct=ct, logLevel=1)
#        logging.info("START REFIN")
#        while is_refin:
#            refit+=1
#            if len(outxx)>=2:
#                ppoutx = outxx[-2]
#                poutx = outxx[-1]
#            elif len(outxx)==1:
#                ppoutx=None
#                poutx = outxx[-1]
#            else:
#                ppoutx=None
#                poutx=None
#            monos, rhs, ct, cff = count_points(pprx,pprt,
#                    pc=poutx,pco=ppoutx,pc_cff=pc_cff)
#            ct = np.hstack([ct,ct])
#            logging.info(f"{min(cff)},{max(cff)}")
#            np.savetxt(f"ttt{refit}.dat",cff)
#            ones = np.ones((len(monos),1))
#
#            A1 = np.hstack([monos, ones])
#            A2 = np.hstack([-monos, ones])
#            task_A = np.vstack([A1,A2])
#
#            task_rhs = np.hstack([rhs,-rhs])
#
#            pc_cff = simplex.solve(task_A, task_rhs, ct=ct, logLevel=1)
#            np.savetxt(f"xdata_ref_{refit}.txt", pc_cff)
#            if cff_old is None:
#                cff_old = cff
#            else:
#                delta_cff = abs((cff - cff_old)/cff_old)
#                np.savetxt("ddd.dat",delta_cff)
#                indr = np.argmax(delta_cff)
#                is_refin =  delta_cff[indr] > 0.01
#                logging.info(f"delta_cff[{indr}]: {delta_cff[indr]}")
#                logging.info(f"{cff_old[indr]} | {cff[indr]}")
#                cff_old = cff
#
#        outx = pc_cff
        opt = test(pprx, pprt, outx)

        np.savetxt(f"xdata_{itcnt}.txt", outx)
        
        v_lst = []
        p_lst = []
        for i in range(treg):
            for j in range(xreg):
                ind = make_id(i, j)
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
        ind = 0
        if v0 is None:
            vu= v*a
            delta_v = abs(vu*a)
            ind = np.argmax(delta_v)
            logging.info(f"delta_v[{ind}]: {delta_v[ind]}")
            v0 = vu
        else:
            vu = (v-v0)*a+v0
            delta_v = abs(vu-v0)
            ind = np.argmax(delta_v)
            logging.info(f"delta_v[{ind}]: {delta_v[ind]}")
            v0 = vu
        if v_old is None:
            delta_v_c = abs(v)
            indr = np.argmax(delta_v_c)
            logging.info(f"delta_v_c[{indr}]: {delta_v_c[indr]}")
            logging.debug(f"0 | {v[indr]}")
        else:
            delta_v_c = abs(v - v_old)
            indr = np.argmax(delta_v_c)
            is_run =  delta_v_c[indr] > 0.01
            logging.info(f"delta_v_c[{indr}]: {delta_v_c[indr]}")
            logging.debug(f"{v_old[indr]} | {v[indr]}")
        logging.debug(f"max_v: {np.max(vu)} | {np.max(v)}")
        logging.debug(f"min_v: {np.min(vu)} | {np.min(v)}")
        logging.debug(f"v    : {vu[10]} | {v[10]}")
        logging.debug(f"max_p: {np.max(p)}")
        logging.debug(f"min_p: {np.min(p)}")
        v_old = v
        outxx.append(outx)
        f = open('dv.txt','a')
        f.write(f"{opt}\n")
        f.close()
        f = open('v.txt','a')
        f.write(f"{v}\n")
        f.close()
        f = open('vu.txt','a')
        f.write(f"{vu}\n")
        f.close()
        t = time.time() - stime
        logging.debug("iter time {} seconds".format(t) )
    np.savetxt(f"xdata_{itcnt}.txt", outx)

if __name__ == "__main__":
    import time
    logging.basicConfig(filename="wathamm.log", level=logging.DEBUG,
                        format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H-%M-%S')
    params = {
            'xreg'   : 6,
            'treg'   : 12,
            'pol_deg': 3,
            'pprx'   : 6,
            'pprt'   : 6,
            }
    logging.info("*"*40)
    logging.info("START")
    stime = time.time()
    count(params)
    t = time.time() - stime
    logging.debug("total time {} seconds".format(t) )
