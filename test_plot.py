import numpy as np
import scipy as sc
from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray
from poly import mvmonos, powers

from mpl_toolkits.mplot3d import axes3d

import matplotlib.pyplot as plt



from constants import *

pc = np.loadtxt("test_cff")

cff_cnt = [10,10]

s,f = 0,cff_cnt[0]
p_cf = pc[s:f]
s,f = s+cff_cnt[0],f+cff_cnt[1]
v_cf = pc[s:f]

X = sc.linspace(0, length, totalx)
T = sc.linspace(0, time, totalt)


#pressure
tt,xx = np.meshgrid(T,X)
in_pts_cr = np.vstack([tt.flatten(),xx.flatten()]).T

tt,xx = np.meshgrid(T,X)


p = mvmonos(in_pts_cr, powers(max_poly_degree, 2), [0, 0])
up = p.dot(p_cf)
pp = up.reshape((len(T), len(X)))

dpdx = mvmonos(in_pts_cr, powers(max_poly_degree, 2), [0, 1])
udpdx = dpdx.dot(p_cf)
dpdt = mvmonos(in_pts_cr, powers(max_poly_degree, 2), [1, 0])
udpdt = dpdt.dot(p_cf)


v = mvmonos(in_pts_cr, powers(max_poly_degree, 2), [0, 0])
uv = v.dot(v_cf)
vv = uv.reshape((len(T),len(X)))

dvdt = mvmonos(in_pts_cr, powers(max_poly_degree, 2), [1, 0])
udvdt = dvdt.dot(v_cf)

dvdx = mvmonos(in_pts_cr, powers(max_poly_degree, 2), [0, 1])
udvdx = dvdx.dot(v_cf)


# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# ax.plot_wireframe(T,X,pp)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Grab some test data.
X, Y, Z = axes3d.get_test_data(0.05)

# Plot a basic wireframe.
ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

plt.show()

plt.show()
