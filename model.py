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

def mvvals(x, val, shift_ind, cff_cnt, diff=None):
    lzeros = sum((cff_cnt[i] for i in range(shift_ind)))
    rzeros = sum((cff_cnt[i] for i in range(shift_ind + 1, len(cff_cnt))))
    monos = np.zeros((len(x), cff_cnt[shift_ind]))
    monos[:,0] = val
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

def eq1(*grid_base, cf=None, cfo=None, cf_cff=None):
    monos = left - right

    if cf_cff is None:
        rhs = np.full(len(monos), 0)
        cff = np.full(len(monos), accs["eq1"])
    else:
        rhs = np.full(len(monos), 0)
        lv = left.dot(cf_cff)
        rv = right.dot(cf_cff)
        lva = np.abs(lv)
        rva = np.abs(rv)
        svals = np.vstack([lva,rva])
        cff = np.amax(svals,axis=0)
        cff *= epsilon

    return monos, rhs, cff, ["eq1"]*len(monos)


def eq2(*grid_base, cf=None, cf_cff=None):
    in_pts = nodes(*grid_base)

    left = eq2_left(in_pts)
    right = eq2_right(in_pts)
    monos = left - right
    if cf_cff is None:
        rhs = np.full(len(monos), 0)
        cff = np.full(len(monos), accs["eq2"])
    else:
        rhs = np.full(len(monos), 0)
        lv = left.dot(cf_cff)
        rv = right.dot(cf_cff)
        lva = np.abs(lv)
        rva = np.abs(rv)
        svals = np.vstack([lva,rva])
        cff = np.amax(svals,axis=0)
        cff *= epsilon

    return monos, rhs, cff, ["eq2"]*len(monos)

def make_cnst_name(first, second=None):
    cnst_name = first
    if second != None:
        cnst_name += "_" + name
    return cnst_name


def boundary_val(val,eps, ind, *grid_base, name=None, cf_cff=None):
    """
    Boundary points
    """
    sb_pts_x0 = nodes(*grid_base)
    monos = mvmonoss(sb_pts_x0, ppwrs, ind, cff_cnt)
    rhs = np.full(len(monos), val)
    if cf_cff is None:
        cff = np.full(len(monos), eps)
    else:
        svals = np.vstack([monos.dot(cf_cff),rhs])
        cff = np.amax(abs(svals),axis=0)
        cff *= epsilon
    return monos, rhs, cff, [make_cnst_name("bnd_val",name)]*len(monos)

def boundary_fnc(fnc,eps, ind,  *grid_base, name=None,cf_cff=None):
    """
    Boundary points
    """
    sb_pts_x0 = nodes(*grid_base)
    monos = mvmonoss(sb_pts_x0, ppwrs, ind, cff_cnt)
    rhs = np.apply_along_axis(fnc, 1, sb_pts_x0)
    if cf_cff is None:
        cff = np.full(len(monos), eps)
    else:
        svals = np.vstack([monos.dot(cf_cff),rhs])
        print(svals)
        cff = np.amax(abs(svals),axis=0)
        cff *= epsilon
    return monos, rhs, cff, [make_cnst_name("bnd_fnc",name)]*len(monos)


def betw_blocks(pws, gind,dind, pind, eps, X_part, T_part, name=None):
    i, j = gind
    di,dj = dind
    ind = make_id(i, j)
    monos = []
    lv = []
    rv = []

    if i < treg - 1:
        grid_base = T_part[i][-1], X_part[j]
        ptr_bnd = nodes(*grid_base)
        val = mvmonoss(ptr_bnd, pws, pind, cff_cnt)
        val = shifted(val, ind)
        lv.append(val)

        ni, nj = i+di, j
        indn = make_id(ni, nj)
        grid_basen = T_part[ni][0], X_part[nj]
        ptr_bndn = nodes(*grid_basen)
        valn = mvmonoss(ptr_bndn, pws, pind, cff_cnt)
        valn = shifted(valn, indn)
        rv.append(valn)
        monos.append(valn - val)
    if j < xreg - 1:
        grid_base = T_part[i], X_part[j][-1]
        ptr_bnd = nodes(*grid_base)
        val = mvmonoss(ptr_bnd, pws, pind, cff_cnt)
        val = shifted(val, ind)
        lv.append(val)

        ni, nj = i, j+dj
        indn = make_id(ni, nj)
        grid_basen = T_part[ni], X_part[nj][0]

        ptr_bndn = nodes(*grid_basen)
        valn = mvmonoss(ptr_bndn, pws, pind, cff_cnt)
        valn = shifted(valn, indn)
        rv.append(valn)
        
        monos.append(valn - val)
    monos = np.vstack(monos)
    rhs = np.full(len(monos), 0)
    if pc_cff is None:
        cff = np.full(len(monos), eps)
    else:
        lv = np.vstack(lv)
        rv = np.vstack(rv)
        lvv = lv.dot(pc_cff[:-1])
        rvv = rv.dot(pc_cff[:-1])
        svals = np.vstack([lvv,rvv])
        cff = np.amax(abs(svals),axis=0)
        cff *= epsilon
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


def count_points(pprx,pprt,pc=None,pco=None,cff0=None):
    lblns = []
    rblns = []
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
    refine_vals = True
    for i in range(treg):
        for j in range(xreg):
            ind = make_id(i, j)
            # TODO: append velocities val selection for block
            grid_base = T_part[i], X_part[j]
            in_pts = nodes(*grid_base)
            # equation 1
            left = eq1_left(in_pts)
            shlf = shifted(left, ind)
            right = eq1_right(in_pts)
            shrf = shifted(right, ind)
            lblns.append(shlf)
            rblns.append(shrf)
            rhs.append(np.zeros(len(in_pts)))
            if cff0 is None:
                cff0 = np.full(len(in_pts), accs["eq1"])
            cff.append(cff0)
            cnst_type.append(np.full(len(in_pts),f"eq1-{j}x{i}")

            # equation 2
            left = eq2_left(in_pts)
            shlf = shifted(left, ind)
            right = eq2_right(in_pts)
            shrf = shifted(right, ind)
            lblns.append(shlf)
            rblns.append(shrf)
            rhs.append(np.zeros(len(in_pts)))
            if cff0 is None:
                cff0 = np.full(len(in_pts), accs["eq2"])
            cff.append(cff0)
            cnst_type.append(np.full(len(in_pts),f"eq2-{j}x{i}")

    for i in range(treg):
        ind = make_id(i, xreg - 1)
        grid_base = T_part[i], X_part[xreg - 1][-1]
        sb_pts_x0 = nodes(*grid_base)
        left = mvmonoss(sb_pts_x0, ppwrs, 1, cff_cnt)
        shlf = shifted(left, ind)
        vals = np.apply_along_axis(vs, 1, sb_pts_x0)
        right = mvvals(sb_pts_x0, vals, 1, cff_cnt)
        shrf = shifted(right, ind)
        lblns.append(shlf)
        rblns.append(shrf)
        rhs.append(np.zeros(len(sb_pts_x0)))
        if cff0 is None:
            cff0 = np.full(len(sb_pts_x0), accs["v"])
        cff.append(cff0)
        cnst_type.append(np.full(len(sb_pts_x0),f"bnd-{j}x{i}-vel-rght")

    for j in range(xreg):
        ind = make_id(0, j)
        grid_base = T_part[0][0], X_part[j]
        sb_pts_x0 = nodes(*grid_base)
        left = mvmonoss(sb_pts_x0, ppwrs, 0, cff_cnt)
        shlf = shifted(left, ind)
        vals = np.apply_along_axis(ps, 1, sb_pts_x0)
        right = mvvals(sb_pts_x0, vals, 0, cff_cnt)
        shrf = shifted(right, ind)
        lblns.append(shlf)
        rblns.append(shrf)
        rhs.append(np.zeros(len(sb_pts_x0)))
        if cff0 is None:
            cff0 = np.full(len(sb_pts_x0), accs["p"])
        cff.append(cff0)
        cnst_type.append(np.full(len(sb_pts_x0),f"bnd-{j}x{i}-prs-start")

    for i in range(treg):
        ind = make_id(i, 0)
        grid_base = T_part[i], X_part[0][0]
        sb_pts_x0 = nodes(*grid_base)
        left = mvmonoss(sb_pts_x0, ppwrs, 0, cff_cnt)
        shlf = shifted(left, ind)
        right = mvvals(sb_pts_x0, p0, 0, cff_cnt)
        shrf = shifted(right, ind)
        lblns.append(shlf)
        rblns.append(shrf)
        rhs.append(np.zeros(len(sb_pts_x0)))
        if cff0 is None:
            cff0 = np.full(len(sb_pts_x0), accs["p"])
        cff.append(cff0)
        cnst_type.append(np.full(len(sb_pts_x0),f"bnd-{j}x{i}-prs-left")


    for j in range(xreg):
        ind = make_id(0, j)
        grid_base = T_part[0][0], X_part[j]
        sb_pts_x0 = nodes(*grid_base)
        left = mvmonoss(sb_pts_x0, ppwrs, 1, cff_cnt)
        shlf = shifted(left, ind)
        vals = np.full(len(sb_pts_x0), v0)
        right = mvvals(sb_pts_x0, v0, 1, cff_cnt)
        shrf = shifted(right, ind)
        lblns.append(shlf)
        rblns.append(shrf)
        rhs.append(np.zeros(len(sb_pts_x0)))
        if cff0 is None:
            cff0 = np.full(len(sb_pts_x0), accs["v"])
        cff.append(cff0)
        cnst_type.append(np.full(len(sb_pts_x0),f"bnd-{j}x{i}-vel-left")


    conditions = []
    for i in range(treg):
        for j in range(xreg):
            if i < treg - 1 or j < xreg - 1:
                #pressure connect blocks
                m, r, c, t = betw_blocks(ppwrs, (i, j),(1,1), 0, accs["p"], X_part, T_part)
                t = [f"{q}-{j}x{i}" for q in t]
                conditions.append((m,r,c,t))
                #velocity connect blocks
                m, r, c, t = betw_blocks(ppwrs, (i, j),(1,1), 1, accs["v"], X_part, T_part)
                t = [f"{q}-{j}x{i}" for q in t]
                conditions.append((m,r,c,t))
    for m, r, c, t in conditions:
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
