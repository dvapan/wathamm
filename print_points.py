import numpy as np
import scipy as sc
from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray

from poly import mvmonos, powers

import pandas as pd

from constants import *

import matplotlib.pyplot as plt

from wathamm import boundary_fnc

from wathamm import eq1_left, eq1_right
from wathamm import eq2_left, eq2_right
from wathamm import vs

pc = np.loadtxt("test_cff")

ppwrs = powers(max_poly_degree, 2)
psize = len(ppwrs)


cff_cnt = [psize,psize]

s,f = 0,cff_cnt[0]
p_cf = pc[s:f]
s,f = s+cff_cnt[0],f+cff_cnt[1]
v_cf = pc[s:f]

X = sc.linspace(0, length, totalx)
T = sc.linspace(0, time, totalt)

tt,xx = np.meshgrid(T,X)
in_pts = np.vstack([tt.flatten(),xx.flatten()]).T
tt,xx = np.meshgrid(T,X)

p = mvmonos(in_pts, ppwrs, [0, 0])
up = p.dot(p_cf)

dpdx = mvmonos(in_pts, ppwrs, [0, 1])
udpdx = dpdx.dot(p_cf)
dpdt = mvmonos(in_pts, ppwrs, [1, 0])
udpdt = dpdt.dot(p_cf)


v = mvmonos(in_pts, ppwrs, [0, 0])
uv = v.dot(v_cf)

dvdt = mvmonos(in_pts, ppwrs, [1, 0])
udvdt = dvdt.dot(v_cf)

dvdx = mvmonos(in_pts, ppwrs, [0, 1])
udvdx = dvdx.dot(v_cf)

# data = np.hstack([in_pts,up.reshape(-1,1)])


lbal_eq1 = eq1_left(in_pts, pc)
rbal_eq1 = eq1_right(in_pts, pc)
lbal_eq2 = eq2_left(in_pts, pc)
rbal_eq2 = eq2_right(in_pts, pc)



data = {
    "T": in_pts[:, 0],
    "X": in_pts[:, 1],
    "p": up,
    "v": uv,
    # "dpdx":udpdx,
    # "dpdt":udpdt,
    # "dvdx":udvdx,
    # "dvdt":udvdt,
    "lbal_eq1":lbal_eq1,
    "rbal_eq1":rbal_eq1,
    "resid1": rbal_eq1 - lbal_eq1,
    "cff1": np.maximum(lbal_eq1,rbal_eq1)*0.01,
    "lbal_eq2":lbal_eq2,
    "rbal_eq2":rbal_eq2,
    "resid2": rbal_eq2 - lbal_eq2,
    "cff2": np.maximum(lbal_eq2,rbal_eq2)*0.01,
}


df = pd.DataFrame(data)
df.set_index("T")

df.to_string(buf='out', float_format=lambda x: "{:.3f}".format(x),header = True, index = True)


tt,xx = np.meshgrid(T[0],X) 
p_pts = np.vstack([tt.flatten(),xx.flatten()]).T
tt,xx = np.meshgrid(T[0],X)

p = mvmonos(p_pts, ppwrs, [0, 0])
up = p.dot(p_cf)


data_p_init = {
    "T": p_pts[:, 0],
    "X": p_pts[:, 1],
    "p": up,
    "resid_p": up - p0
}


df = pd.DataFrame(data_p_init)
df.set_index("T")

df.to_string(buf='out_p_init', float_format=lambda x: "{:.3f}".format(x),header = True, index = True)


# def vs(pts):
#     t,x = pts
#     if 1 - 1/timeclose*t > 0:
#         return 1 - 1/timeclose*t
#     else:
#         return 0


tt,xx = np.meshgrid(T[0],X) 
v_pts = np.vstack([tt.flatten(),xx.flatten()]).T
tt,xx = np.meshgrid(T[0],X)

v = mvmonos(v_pts, ppwrs, [0, 0])
uv = p.dot(v_cf)


data_v_init = {
    "T": v_pts[:, 0],
    "X": v_pts[:, 1],
    "v": uv,
    "resid_v": uv - v0
}


df = pd.DataFrame(data_v_init)
df.set_index("T")

df.to_string(buf='out_v_init', float_format=lambda x: "{:.3f}".format(x),header = True, index = True)

tt,xx = np.meshgrid(T,X[-1]) 
v_pts = np.vstack([tt.flatten(),xx.flatten()]).T
tt,xx = np.meshgrid(T,X[-1])

v = mvmonos(v_pts, ppwrs, [0, 0])
uv = p.dot(v_cf)

rhs = np.apply_along_axis(vs, 1, v_pts)

data_v_init2 = {
    "T": v_pts[:, 0],
    "X": v_pts[:, 1],
    "v": uv,
    "resid_v": uv - rhs
}


df = pd.DataFrame(data_v_init2)
df.set_index("T")

df.to_string(buf='out_v_init2', float_format=lambda x: "{:.3f}".format(x),header = True, index = True)

