import numpy as np
import scipy.sparse as sps

from poly import mvmonos, powers
from constants import *

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


def make_id(i,j,p):
    return i*p["xreg"] + j

def shifted(cffs,shift,p):
    pcount = len(cffs)
    psize = len(cffs[0])
    max_reg = p["xreg"]*p["treg"]
    lzeros = np.zeros((pcount, psize * shift))
    rzeros = np.zeros((pcount, (max_reg - shift-1) * psize))
    cffs = np.hstack([lzeros,cffs,rzeros])
    return cffs

def make_cnst_name(first, second=None):
    cnst_name = first
    if second != None:
        cnst_name += "_" + name
    return cnst_name


def boundary_val(val,eps, ind, *grid_base, name=None, cf_cff=None):
    """
    Boundary points
    """
    pts = nodes(*grid_base)
    left = mvmonoss(pts, ppwrs, ind, cff_cnt)
    right = np.zeros_like(left)
    rhs = np.full(len(pts), val)
    cff = np.full(len(pts), eps)
    return left, right, rhs, cff, [make_cnst_name("bnd_val",name)]*len(pts)

def boundary_fnc(fnc,eps, ind,  *grid_base, name=None,cf_cff=None):
    """
    Boundary points
    """
    pts = nodes(*grid_base)
    left = mvmonoss(pts, ppwrs, ind, cff_cnt)
    right = np.zeros_like(left)
    rhs = np.apply_along_axis(fnc, 1, pts)
    cff = np.full(len(pts), eps)
    return left, right, rhs, cff, [make_cnst_name("bnd_fnc",name)]*len(pts)


def betw_blocks(pws, gind,dind, pind, eps, X_part, T_part,params):
    xreg = params["xreg"]
    treg = params["treg"]
    i, j = gind
    di,dj = dind
    ind = make_id(i, j, params)
    monos = []
    lv = []
    rv = []

    if i < treg - 1:
        grid_base = T_part[i][-1], X_part[j]
        ptr_bnd = nodes(*grid_base)
        val = mvmonoss(ptr_bnd, pws, pind, cff_cnt)
        val = shifted(val, ind, params)
        lv.append(val)

        ni, nj = i+di, j
        indn = make_id(ni, nj, params)
        grid_basen = T_part[ni][0], X_part[nj]
        ptr_bndn = nodes(*grid_basen)
        valn = mvmonoss(ptr_bndn, pws, pind, cff_cnt)
        valn = shifted(valn, indn, params)
        rv.append(valn)
        monos.append(valn - val)
    if j < xreg - 1:
        grid_base = T_part[i], X_part[j][-1]
        ptr_bnd = nodes(*grid_base)
        val = mvmonoss(ptr_bnd, pws, pind, cff_cnt)
        val = shifted(val, ind, params)
        lv.append(val)

        ni, nj = i, j+dj
        indn = make_id(ni, nj, params)
        grid_basen = T_part[ni], X_part[nj][0]

        ptr_bndn = nodes(*grid_basen)
        valn = mvmonoss(ptr_bndn, pws, pind, cff_cnt)
        valn = shifted(valn, indn, params)
        rv.append(valn)
        
        monos.append(valn - val)
    lv = np.vstack(lv)
    rv = np.vstack(rv)
    monos = np.vstack(monos)
    rhs = np.full(len(monos), 0)
    cff = np.full(len(monos), eps)
    return lv, rv, rhs, cff, ["betw_blocks"]*len(monos)




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


def count_points(params, v_0_=None, cff0=None, a=None, sqp0=None):
    lvals = []
    rvals = []
    monos = []
    rhs = []
    cff = []
    cnst_type = []
    xreg = params["xreg"]
    treg = params["treg"]
    pprx = params["pprx"]
    pprt = params["pprt"]
    totalx = xreg*pprx - xreg + 1
    totalt = treg*pprt - treg + 1
    X = np.linspace(0, length, totalx)
    T = np.linspace(0, total_time, totalt)
    X_part = list(mit.windowed(X,n=pprx,step=pprx - 1))
    T_part = list(mit.windowed(T,n=pprt,step=pprt - 1))
    bsize = sum(cff_cnt)
    refine_vals = True

    v_old = None
    dpdx = []
    dpdt = []
    v    = []
    v_0  = []
    dvdx = []
    dvdt = []
    print('start prepare')
    sh_v = 0
    for i in range(treg):
        for j in range(xreg):
            ind = make_id(i, j, params)
            grid_base = T_part[i], X_part[j]
            pts = nodes(*grid_base)
            p1 = mvmonoss(pts, ppwrs, 0, cff_cnt, [0, 1]) 
            dpdx.append(sps.csr_matrix(shifted(p1, ind, params)))
            p2 = mvmonoss(pts, ppwrs, 0, cff_cnt, [1, 0])
            dpdt.append(sps.csr_matrix(shifted(p2, ind, params)))
            p3 = mvmonoss(pts, ppwrs, 1, cff_cnt, [0, 0]) 
            v.append(sps.csr_matrix(shifted(p3, ind, params)))
            p4 = mvmonoss(pts, ppwrs, 1, cff_cnt, [0, 1])
            dvdx.append(sps.csr_matrix(shifted(p4, ind, params)))
            p5 = mvmonoss(pts, ppwrs, 1, cff_cnt, [1, 0])
            dvdt.append(sps.csr_matrix(shifted(p5, ind, params)))
            if v_0_ is None:
                p = np.zeros_like(p3)                 
            else:
                p = np.zeros_like(p3) 
                p[:,psize] = v_0_[sh_v:sh_v+len(pts)]
            v_0.append(sps.csr_matrix(shifted(p, ind, params)))
            sh_v += len(pts)
    print('end prepare')

    dpdx = sps.vstack(dpdx)
    dpdt = sps.vstack(dpdt)
    v    = sps.vstack(v) 
    dvdx = sps.vstack(dvdx)
    dvdt = sps.vstack(dvdt)
    v_0  = sps.vstack(v_0)

    num_points = dpdx.shape[0]#totalt*totalx

    lm1 = dpdx
    sqp = lmd*v_0.multiply(np.abs(v_0))/(2*d) 
    if sqp0 is not None:
        sqp = (sqp - sqp0)*a + sqp0
    lnp = lmd*(v-v_0)/d
    rm1 = -rho*(dvdt + sqp + lnp)
    lm2 = dpdt
    rm2 = -c2*rho*dvdx

    m1 = lm1 - rm1
    m2 = lm2 - rm2

    r1 = np.full(num_points, 0)
    cff1 = np.full(num_points, accs["eq1"])
    ct1 =np.full(num_points,"eq1")

    r2 = np.full(num_points, 0)
    cff2 = np.full(num_points, accs["eq2"])
    ct2 =np.full(num_points,"eq2")

    lvals.append(lm1)
    rvals.append(rm1)
    monos.append(m1)
    rhs.append(r1)
    cff.append(cff1)
    cnst_type.append([f"{q}-{j}x{i}" for q in ct1] )

    lvals.append(lm2)
    rvals.append(rm2)
    monos.append(m2)
    rhs.append(r2)
    cff.append(cff2)
    cnst_type.append([f"{q}-{j}x{i}" for q in ct2] )

    for i in range(treg):
        ind = make_id(i, xreg-1, params)
        lm,rm,r,c,t = boundary_fnc(vs,accs["v"], 1, T_part[i],X_part[xreg - 1][-1])
        lm = sps.csr_matrix(shifted(lm, ind, params))
        rm = sps.csr_matrix(shifted(rm, ind, params))
        m = lm - rm
#        lvals.append(lm)
#        rvals.append(rm)
        monos.append(m)
        rhs.append(r)
        cff.append(c)
        cnst_type.append([f"{q}-{xreg - 1}x{i}-vel-on-bound" for q in t])

    for j in range(xreg):
        ind = make_id(0, j, params)
        lm,rm,r,c,t = boundary_fnc(ps,accs["p"], 0, T_part[0][0], X_part[j])
        lm = sps.csr_matrix(shifted(lm, ind, params))
        rm = sps.csr_matrix(shifted(rm, ind, params))
        m = lm - rm
#        lvals.append(lm)
#        rvals.append(rm)
        monos.append(m)
        rhs.append(r)
        cff.append(c)
        cnst_type.append([f"{q}-{j}x{0}-pressure-start-time" for q in t])

    for i in range(treg):
        ind = make_id(i, 0, params)
        lm,rm,r,c,t = boundary_val(p0,accs["p"], 0, T_part[i], X_part[0][0])
        lm = sps.csr_matrix(shifted(lm, ind, params))
        rm = sps.csr_matrix(shifted(rm, ind, params))
        m = lm - rm
#        lvals.append(lm)
#        rvals.append(rm)
        monos.append(m)
        rhs.append(r)
        cff.append(c)
        cnst_type.append([f"{q}-{0}x{i}-pressure-start-pos" for q in t])


    for j in range(xreg):
        ind = make_id(0, j, params)
        lm,rm,r,c,t = boundary_val(v0,accs["v"], 1, T_part[0][0], X_part[j])
        lm = sps.csr_matrix(shifted(lm, ind, params))
        rm = sps.csr_matrix(shifted(rm, ind, params))
        m = lm - rm
#        lvals.append(lm)
#        rvals.append(rm)
        monos.append(m)
        rhs.append(r)
        cff.append(c)
        cnst_type.append([f"{q}-{j}x{0}-vel-on-left-pos" for q in t])



    conditions = []
    for i in range(treg):
        for j in range(xreg):
            if i < treg - 1 or j < xreg - 1:
                #pressure connect blocks
                lm,rm, r, c, t = betw_blocks(ppwrs, (i, j),(1,1), 0,
                        accs["p"], X_part, T_part, params)
                t = [f"{q}-{j}x{i}" for q in t]
                m1 = sps.csr_matrix(lm - rm)
#                lvals.append(lm)
#                rvals.append(rm)
                conditions.append((m1,r,c,t))
                #velocity connect blocks
                lm,rm, r, c, t = betw_blocks(ppwrs, (i, j),(1,1), 1,
                        accs["v"], X_part, T_part, params)
                t = [f"{q}-{j}x{i}" for q in t]
                m2 = sps.csr_matrix(lm - rm)
#                lvals.append(lm)
#                rvals.append(rm)
                conditions.append((m2,r,c,t))
    for m, r, c, t in conditions:
        monos.append(m)
        rhs.append(r)
        cff.append(c)
        cnst_type.append(t)

    lvals = sps.vstack(lvals)
    rvals = sps.vstack(rvals)

    monos = sps.vstack(monos)
    rhs = np.hstack(rhs)
    if cff0 is None:
        cff = np.hstack(cff)
    else:
        cff = cff0
    rhs /= cff
    monos /= cff.reshape(-1,1)

    cnst_type = np.hstack(cnst_type)

    return monos, rhs, cnst_type, sqp
