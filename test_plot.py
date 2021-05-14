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


# plt.plot(xx[:,0],pp[0,:])

# plt.plot(xx[:,0],pp[-1,:])

plt.plot(xx[:,0],vv[0,:])

plt.plot(xx[:,0],vv[-1,:])

print (pp)
print (vv)

# plt.plot(tt[0,:],vv[:, 0])

# plt.plot(tt[0,:],vv[:,-1])

# plt.plot(tt[0,:],pp[:,100])

# print(min(pp[:,-1]), max(pp[:,-1]))

# plt.plot(tt[0,:],pp[:,-1])


print (vv[0,0])

print(max(udpdx), min(udpdx))
rbnd = rho*(udvdt - 0.01*uv/(2*d))
print(max(rbnd), min(rbnd))

print(max(udpdt), min(udpdt))
rbnd = c2*rho*udvdx
print(max(rbnd), min(rbnd))

# np.savetxt("out",np.vstack([udpdx,rho*(udvdt + 1*uv/(2*d))]).reshape(-1,2))


# print(uu)
# plt.plot(xx[:,0],uu[0,:])

# fig, ax = plt.subplots()
# p = ax.contourf(tt, xx, uu, np.linspace(700, 1900, 100), cmap='inferno')

# fig.colorbar(p, ax=ax)
# fig.tight_layout()
plt.show()
