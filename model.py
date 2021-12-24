import numpy as np
import scipy as sc

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
    lzeros = sc.zeros((pcount, psize * shift))
    rzeros = sc.zeros((pcount, (max_reg - shift-1) * psize))
    cffs = sc.hstack([lzeros,cffs,rzeros])
    return cffs

def eq1_left(pts, cf=None):
    dpdx = mvmonoss(pts, ppwrs, 0, cff_cnt, [0, 1])
    if cf is not None:
        dpdx = dpdx.dot(cf)
    return dpdx

def eq1_right(pts, cf=None):
    v = mvmonoss(pts, ppwrs, 1, cff_cnt, [0, 0])
    dvdt = mvmonoss(pts, ppwrs, 1, cff_cnt, [1, 0])
    if cf is not None:
        v = v.dot(cf)
        dvdt = dvdt.dot(cf)
    return -rho*(dvdt + lmd*v/(2*d))

def eq2_left(pts, cf=None):
    dpdt = mvmonoss(pts, ppwrs, 0, cff_cnt, [1, 0])
    if cf is not None:
        dpdt = dpdt.dot(cf)
    return dpdt

def eq2_right(pts, cf=None):
    dvdx = mvmonoss(pts, ppwrs, 1, cff_cnt, [0, 1])
    if cf is not None:
        dvdx = dvdx.dot(cf)

    return -c2*rho*dvdx

def eq1(*grid_base, cf=None):
    in_pts = nodes(*grid_base)

    monos = eq1_left(in_pts) - eq1_right(in_pts)

    rhs = np.full(len(monos), 0)
    if cf is None:
        cff = np.full(len(monos), 100)
    else:
        left_bal = np.abs(eq1_left(in_pts, cf))
        right_bal = np.abs(eq1_right(in_pts, cf))
        cff = 0.01*np.maximum(left_bal, right_bal)
        print("eq1")
        print(cff)
    return monos, rhs, cff, ["eq1"]*len(monos)


def eq2(*grid_base, cf=None):
    in_pts = nodes(*grid_base)
    monos = eq2_left(in_pts) - eq2_right(in_pts)

    rhs = np.full(len(monos), 0)
    if cf is None:
        cff = np.full(len(monos), 100000)
    else:
        left_bal = np.abs(eq2_left(in_pts, cf))
        right_bal = np.abs(eq2_right(in_pts, cf))
        cff = 0.01*np.maximum(left_bal, right_bal)
        print("eq2")
        print(cff)

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


def betw_blocks(pws, gind,dind, pind, eps, name=None):
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
        monos.append(val - valn)
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
        monos.append(val - valn)
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

i = 1


def count_points(poly_coeff=None):
    monos = []
    rhs = []
    cff = []
    cnst_type = []
    for i in range(treg):
        for j in range(xreg):
            conditions = (eq1(T_part[i], X_part[j], cf=poly_coeff),
                          eq2(T_part[i], X_part[j], cf=poly_coeff))

            ind = make_id(i, j)
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
                m, r, c, t = betw_blocks(ppwrs, (i, j),(1,1), 0, 10000)
                t = [f"{q}-{j}x{i}" for q in t]
                conditions.append((m,r,c,t))
                #velocity connect blocks
                m, r, c, t = betw_blocks(ppwrs, (i, j),(1,1), 1, 0.01)
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

    return monos, rhs, cnst_type
