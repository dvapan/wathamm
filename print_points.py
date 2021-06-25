import numpy as np
import scipy as sc
from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray

from poly import mvmonos, powers

import pandas as pd

from constants import *

import matplotlib.pyplot as plt

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


lbal_eq1 = udpdx
rbal_eq1 = rho*(udvdt + lmd*uv/(2*d))
lbal_eq2 = udpdt 
rbal_eq2 = c2*rho*udvdx
 

data = {
    "T": in_pts[:, 0],
    "X": in_pts[:, 1],
    "p": up,
    "v": uv,
    "dpdx":udpdx,
    "dpdt":udpdt,
    "dvdx":udvdx,
    "dvdt":udvdt,
    "lbal_eq1":lbal_eq1,
    "rbal_eq1":rbal_eq1,
    "lbal_eq2":lbal_eq2,
    "rbal_eq2":rbal_eq2,
}

df = pd.DataFrame(data)
df.set_index("T")

pdata= {
        "T": T,
        "p1":df[df.X==X[0]]["p"].to_numpy(),
        "p2":df[df.X==X[65]]["p"].to_numpy(),
        "p3":df[df.X==X[70]]["p"].to_numpy(),
        "p4":df[df.X==X[-1]]["p"].to_numpy(),
        }

pdf = pd.DataFrame(pdata)           
pdf.plot(x="T",y=["p1","p2","p3","p4"])


pdata= {
        "T": T,
        "v1":df[df.X==X[0]]["v"].to_numpy(),
        "v2":df[df.X==X[65]]["v"].to_numpy(),
        "v3":df[df.X==X[70]]["v"].to_numpy(),
        "v4":df[df.X==X[-1]]["v"].to_numpy(),
        }

odf = pd.DataFrame(pdata)           
odf.plot(x="T",y=["v1","v2","v3","v4"])

     
plt.show()
odf.to_string(buf='out', float_format=lambda x: "{:.3f}".format(x),header = True, index = True)
#df = df[abs(df.X-254.5)<0.1]

#df = df.pivot(index='X', columns='T').stack()
# df = df.set_index('p')
#df.to_string(buf='out', float_format=lambda x: "{:.3f}".format(x),header = True, index = True)

