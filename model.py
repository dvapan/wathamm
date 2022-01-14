import numpy as np

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


def make_id(i,j):
    return i*xreg + j

def shifted(cffs,shift):
    pcount = len(cffs)
    psize = len(cffs[0])
    lzeros = np.zeros((pcount, psize * shift))
    rzeros = np.zeros((pcount, (max_reg - shift-1) * psize))
    cffs = np.hstack([lzeros,cffs,rzeros])
    return cffs

def eq1_left(pts):
    dpdx = mvmonoss(pts, ppwrs, 0, cff_cnt, [0, 1])
    return dpdx

def eq1_right(pts, cf=None, cfo=None):
    v = mvmonoss(pts, ppwrs, 1, cff_cnt, [0, 0])
    dvdt = mvmonoss(pts, ppwrs, 1, cff_cnt, [1, 0])
    if cf is None:
        v0 = np.zeros_like(v)
    elif cfo is None:
        v0 = np.zeros_like(v)
        v0[:,psize]=v.dot(cf)*a
    else:
        v0 = np.zeros_like(v)
        v_0 = v.dot(cf)
        v_0o = v.dot(cfo)
        tst = (v_0 - v_0o)*a + v_0o
        v0[:,psize] = tst

    return -rho*(dvdt + lmd*v0*np.abs(v0)/(2*d) + lmd*(v-v0)/d )

def eq2_left(pts):
    dpdt = mvmonoss(pts, ppwrs, 0, cff_cnt, [1, 0])
    return dpdt

def eq2_right(pts):
    dvdx = mvmonoss(pts, ppwrs, 1, cff_cnt, [0, 1])
    return -c2*rho*dvdx

def eq1(*grid_base, cf=None, cfo=None):
    in_pts = nodes(*grid_base)

    if cf is None:
        monos = eq1_left(in_pts) - eq1_right(in_pts,cf=cf,cfo=cfo)
        rhs = np.full(len(monos), 0)
        cff = np.full(len(monos), 100)
    else:
        left = eq1_left(in_pts)
        right = eq1_right(in_pts,cf=cf,cfo=cfo)
        monos = left - right
        rhs = np.full(len(monos), 0)
        lv = left.dot(cf)
        rv = right.dot(cf)
        lva = np.abs(lv)
        rva = np.abs(rv)
        svals = np.vstack([lva,rva])
        cff = np.amax(svals,axis=0)
        cff *= epsilon

    return monos, rhs, cff, ["eq1"]*len(monos)


def eq2(*grid_base, cf=None):
    in_pts = nodes(*grid_base)

    if cf is None:
        monos = eq2_left(in_pts) - eq2_right(in_pts)
        rhs = np.full(len(monos), 0)
        cff = np.full(len(monos), 100000)
    else:
        left = eq2_left(in_pts)
        right = eq2_right(in_pts)
        monos = left - right
        rhs = np.full(len(monos), 0)
        lv = left.dot(cf)
        rv = right.dot(cf)
        lva = np.abs(lv)
        rva = np.abs(rv)
        svals = np.vstack([lva,rva])
        cff = np.amax(svals,axis=0)
        print(cff)
        cff *= epsilon

    return monos, rhs, cff, ["eq2"]*len(monos)

def make_cnst_name(first, second=None):
    cnst_name = first
    if second != None:
        cnst_name += "_" + name
    return cnst_name


def boundary_val(val,eps, ind, *grid_base, name=None):
    """
    Boundary points
    """
    sb_pts_x0 = nodes(*grid_base)
    monos = mvmonoss(sb_pts_x0, ppwrs, ind, cff_cnt)
    rhs = np.full(len(monos), val)
    cff = np.full(len(monos), eps)
    return monos, rhs, cff, [make_cnst_name("bnd_val",name)]*len(monos)

def boundary_fnc(fnc,eps, ind,  *grid_base, name=None):
    """
    Boundary points
    """
    sb_pts_x0 = nodes(*grid_base)
    monos = mvmonoss(sb_pts_x0, ppwrs, ind, cff_cnt)
    rhs = np.apply_along_axis(fnc, 1, sb_pts_x0)
    cff = np.full(len(monos), eps)
    return monos, rhs, cff, [make_cnst_name("bnd_fnc",name)]*len(monos)


def betw_blocks(pws, gind,dind, pind, eps, X_part, T_part, name=None):
    i, j = gind
    di,dj = dind
    ind = make_id(i, j)
    monos = []

    if i < treg - 1:
        grid_base = T_part[i][-1], X_part[j]
        ptr_bnd = nodes(*grid_base)
        val = mvmonoss(ptr_bnd, pws, pind, cff_cnt)
        val = shifted(val, ind)

        ni, nj = i+di, j
        indn = make_id(ni, nj)
        grid_basen = T_part[ni][0], X_part[nj]
        ptr_bndn = nodes(*grid_basen)
        valn = mvmonoss(ptr_bndn, pws, pind, cff_cnt)
        valn = shifted(valn, indn)

        monos.append(valn - val)
    if j < xreg - 1:
        grid_base = T_part[i], X_part[j][-1]
        ptr_bnd = nodes(*grid_base)
        val = mvmonoss(ptr_bnd, pws, pind, cff_cnt)
        val = shifted(val, ind)

        ni, nj = i, j+dj
        indn = make_id(ni, nj)
        grid_basen = T_part[ni], X_part[nj][0]

        ptr_bndn = nodes(*grid_basen)
        valn = mvmonoss(ptr_bndn, pws, pind, cff_cnt)
        valn = shifted(valn, indn)

        monos.append(valn - val)
    monos = np.vstack(monos)
    rhs = np.full(len(monos), 0)
    cff = np.full(len(monos), eps)
    return monos, rhs, cff, [make_cnst_name("betw_blocks",name)]*len(monos)


def shifted(cffs,shift):
    pcount = len(cffs)
    psize = len(cffs[0])
    lzeros = np.zeros((pcount, psize * shift))
    rzeros = np.zeros((pcount, (max_reg - shift-1) * psize))
    cffs = np.hstack([lzeros,cffs,rzeros])
    return cffs


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


def count_points(pprx,pprt,poly_coeff=None,pco=None):
    monos = []
    rhs = []
    cff = []
    cnst_type = []
    totalx = xreg*pprx - xreg + 1
    totalt = treg*pprt - treg + 1
    X = np.linspace(0, length, totalx)
    T = np.linspace(0, total_time, totalt)
    X_part = list(mit.windowed(X,n=pprx,step=pprx - 1))
    T_part = list(mit.windowed(T,n=pprt,step=pprt - 1))
    bsize = sum(cff_cnt)
    for i in range(treg):
        for j in range(xreg):
            ind = make_id(i, j)
            if poly_coeff is None:
                cf = None
                cfo = None
            elif pco is None:
                cf = poly_coeff[ind*bsize:(ind+1)*bsize]
                cfo = np.zeros(bsize)
            else:
                cf = poly_coeff[ind*bsize:(ind+1)*bsize]
                cfo = pco[ind*bsize:(ind+1)*bsize]
            conditions = (eq1(T_part[i], X_part[j], cf=cf, cfo=cfo),
                          eq2(T_part[i], X_part[j], cf=cf))

            for m, r, c, t in conditions:
                m = shifted(m, ind)
                monos.append(m)
                rhs.append(r)
                cff.append(c)
                cnst_type.append([f"{q}-{j}x{i}" for q in t] )

    for i in range(treg):
        m,r,c,t = boundary_fnc(vs,0.1, 1, T_part[i],X_part[xreg - 1][-1])
        ind = make_id(i, xreg-1)
        m = shifted(m, ind)
        monos.append(m)
        rhs.append(r)
        cff.append(c)
        cnst_type.append([f"{q}-{xreg - 1}x{i}-vel-on-bound" for q in t])

    for j in range(xreg):
        m,r,c,t = boundary_fnc(ps,20000, 0, T_part[0][0], X_part[j])
#        m,r,c,t = boundary_val(p0,100000, 0, T_part[0][0], X_part[j])

        ind = make_id(0, j)
        m = shifted(m, ind)
        monos.append(m)
        rhs.append(r)
        cff.append(c)
        cnst_type.append([f"{q}-{j}x{0}-pressure-start-time" for q in t])

    for i in range(treg):
        m,r,c,t = boundary_val(p0,20000, 0, T_part[i], X_part[0][0])
        ind = make_id(i, 0)
        m = shifted(m, ind)
        monos.append(m)
        rhs.append(r)
        cff.append(c)
        cnst_type.append([f"{q}-{0}x{i}-pressure-start-pos" for q in t])


    for j in range(xreg):
        m,r,c,t = boundary_val(v0,0.1, 1, T_part[0][0], X_part[j])
        ind = make_id(0, j)
        m = shifted(m, ind)
        monos.append(m)
        rhs.append(r)
        cff.append(c)
        cnst_type.append([f"{q}-{j}x{0}-vel-on-left-pos" for q in t])



    conditions = []
    for i in range(treg):
        for j in range(xreg):
            if i < treg - 1 or j < xreg - 1:
                #pressure connect blocks
                m, r, c, t = betw_blocks(ppwrs, (i, j),(1,1), 0, 10000, X_part, T_part)
                t = [f"{q}-{j}x{i}" for q in t]
                conditions.append((m,r,c,t))
                #velocity connect blocks
                m, r, c, t = betw_blocks(ppwrs, (i, j),(1,1), 1, 0.01, X_part, T_part)
                t = [f"{q}-{j}x{i}" for q in t]
                conditions.append((m,r,c,t))
    for m, r, c,t in conditions:
        monos.append(m)
        rhs.append(r)
        cff.append(c)
        cnst_type.append(t)

    monos = np.vstack(monos)
    rhs = np.hstack(rhs)
    cff = np.hstack(cff)
    rhs /= cff
    monos /= cff.reshape(-1,1)

    cnst_type = np.hstack(cnst_type)

    return monos, rhs, cnst_type, cff
