import numpy as np
import scipy as sc
from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray
import matplotlib.pyplot as plt

from poly import mvmonos, powers


from constants import *

pc = np.loadtxt("test_cff")

cff_cnt = [10,10]

s,f = 0,cff_cnt[0]
p_cf = pc[s:f]
s,f = s+cff_cnt[0],f+cff_cnt[1]
v_cf = pc[s:f]

X = sc.linspace(0, length, totalx*3)
T = sc.linspace(0, time, totalt*3)

# #pressure
# tt,xx = np.meshgrid(T,X)
# in_pts_cr = np.vstack([tt.flatten(),xx.flatten()]).T
# pp = mvmonos(in_pts_cr,powers(3,2))

# tt,xx = np.meshgrid(T,X)
# u = pp.dot(p_cf)
# uu = u.reshape((len(T), len(X)))

# print(uu[0,:])
# plt.plot(tt[0,:],uu[-1,:])
# velocity

tt,xx = np.meshgrid(T,X)
in_pts_cr = np.vstack([tt.flatten(),xx.flatten()]).T
pp = mvmonos(in_pts_cr,powers(3,2))
u = pp.dot(v_cf)
uu = u.reshape((len(T), len(X)))

plt.plot(tt[0,:],uu[-1,:])

# fig, ax = plt.subplots()
# p = ax.contourf(tt, xx, uu, np.linspace(700, 1900, 100), cmap='inferno')

# fig.colorbar(p, ax=ax)
# fig.tight_layout()
plt.show()
