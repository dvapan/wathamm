import logging
from model import count_points 
from model import nodes,make_id
from model import ppwrs,psize,cff_cnt,mvmonoss
import solvers.simplex as simplex
import solvers.solve_constractions_cone as constr_cone
import solvers.iterate_simplex as iterate_simplex

from scipy.sparse import coo_matrix

from constants import *

def test(params, outx,itcnt):
    monos, rhs, ct = count_points(params, None)
    ct = np.hstack([ct,ct])

    ones = np.ones((len(monos),1))

    A1 = np.hstack([monos, ones])
    A2 = np.hstack([-monos, ones])
    task_A = np.vstack([A1,A2])

    task_rhs = np.hstack([rhs,-rhs])

    resd = np.dot(task_A,outx) - task_rhs
    
    idx = resd.argsort()

    return resd[idx][0], len(resd[resd < -0.01])

def sv_vals_ct(tp,itcnt,val,ct):
    f = open(f"{tp}/{itcnt}.dat",'w')
    for e in zip(val,ct):
        f.write("{} {}\n".format(*e))
    f.close()


def count(params, eps=0.01):
    itcnt = 0
    outx = None
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
    outx_old = None
    outx = None 
    f = open('dv.txt','w')
    f.close()
    v_0 = None
    cff = None
    v_old = None
    cff_old = None
    is_refin = True
    a = a_0
    sqp = None
    lnp = None
    lnp2 = None
    task_old = None
    while is_run  or itcnt == 0:
        itcnt += 1
        logging.info(f"ITER: {itcnt}")
        stime = time.time()
        sdcf = None
        refit = 0
        monos, rhs, ct, sqp, lnp, lnp2 = count_points(params, v_0=v_0, a=a,
                sqp0=sqp, lnp0=lnp, lnp20=lnp2)
        ct = np.hstack([ct,ct])
        if cff_old is None:
            cff_old = np.copy(cff)
        ones = np.ones((len(monos),1))

        A1 = np.hstack([monos, ones])
        A2 = np.hstack([-monos, ones])
        task_A = np.vstack([A1,A2])

        task_rhs = np.hstack([rhs,-rhs])
        print(np.max(task_A),np.min(task_A))
        print((f"TASK SIZE XCOUNT: {task_A.shape[1]} " 
                      f"GXCOUNT: {task_A.shape[0]}"))
        outx = simplex.solve(task_A, task_rhs, ct=ct, 
                logLevel=1,outx=outx)
        
#        opt = test(params, outx, itcnt)

        np.savetxt(f"xdata_{itcnt}.dat", outx)
        
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
        ind = 0
        if v_0 is None:
            vu= v*a
            #vu= v
            delta_v = abs(vu)
            ind = np.argmax(delta_v)
            logging.info(f"delta_v[{ind}]: {delta_v[ind]}")
            logging.info(f"delta_v avg: {np.average(delta_v)}")
            v_0 = vu
        else:
            delta_v = abs(v-v_0)
            vu = (v-v0)*a+v
            #vu = v
            ind = np.argmax(delta_v)
            is_run = delta_v[ind] > accs["v"]
            logging.info(f"delta_v[{ind}]: {delta_v[ind]}")
            logging.info(f"delta_v avg: {np.average(delta_v)}")
            v_0 = vu
        logging.info(f"current a: {a}")
        logging.debug(f"max_v: {np.max(v_0)} | {np.max(v)}")
        logging.debug(f"min_v: {np.min(v_0)} | {np.min(v)}")
        logging.debug(f"max_p: {np.max(p)}")
        logging.debug(f"min_p: {np.min(p)}")
        
        f = open('dv.txt','a')
        f.write(f"{itcnt} {outx[-1]} {delta_v[ind]}\n")
        f.close()

        t = time.time() - stime
        logging.debug("iter time {} seconds".format(t) )
    np.savetxt(f"xdata.txt", outx)

if __name__ == "__main__":
    import time
    import argparse
    import sys
    np.set_printoptions(threshold=sys.maxsize)
    logging.basicConfig(filename="cpm.log",
            level=logging.DEBUG,
            format='%(asctime)s %(message)s', 
            datefmt='%Y-%m-%d %H-%M-%S')
    parser = argparse.ArgumentParser()
    parser.add_argument("--xreg", default=1,type=int)
    parser.add_argument("--treg", default=1,type=int)
    parser.add_argument("--pprx", default=7,type=int)
    parser.add_argument("--pprt", default=7,type=int)
    args = parser.parse_args(sys.argv[1:])
    params = vars(args)

    logging.info("*"*40)
    logging.info("START")
    stime = time.time()
    count(params)
    t = time.time() - stime
    logging.debug("total time {} seconds".format(t) )
