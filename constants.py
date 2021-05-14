import numpy as np
import more_itertools as mit

length = 100                      # Длина теплообменника         [м]
time = 10            # Время работы теплообменника  [с]
timeclose = 1
d = 0.5
delta = 0.2
rho = 1000
K = 2030*10**6 
E = 200000*10**6

c2 = 1/(rho/K + (rho*d)/(delta*E))

v0 = 1.0
p0 = 10*10**6


xreg,treg = 1,1
max_reg = xreg*treg
max_poly_degree = 3
pprx = 200                        # Точек на регион
pprt = 500
totalx = xreg*pprx - xreg + 1
totalt = treg*pprt - treg + 1

dx = length/xreg
dt = time/treg

X = np.linspace(0, length, totalx)
T = np.linspace(0, time, totalt)

X_part = list(mit.windowed(X,n=pprx,step=pprx - 1))
T_part = list(mit.windowed(T,n=pprt,step=pprt - 1))

index_info = 0
cnt_var = 0

epsilon = 1

