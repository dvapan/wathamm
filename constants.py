import numpy as np
import more_itertools as mit

length = 4                      # Длина теплообменника         [м]
time = 300                      # Время работы теплообменника  [с]
d = 0.5
delta = 0.2
rho = 1000
K = 2030
E = 200000
c2 = 1/(rho/K + (rho*d)/(delta*E))

v0 = 3
p0 = 1*10**6


xreg,treg = 1,1
max_reg = xreg*treg
max_poly_degree = 3
ppr = 10                        # Точек на регион

totalx = xreg*ppr - xreg + 1
totalt = treg*ppr - treg + 1

dx = length/xreg
dt = time/treg
timeclose = 10

X = np.linspace(0, length, totalx)
T = np.linspace(0, time, totalt)

X_part = list(mit.windowed(X,n=ppr,step=ppr - 1))
T_part = list(mit.windowed(T,n=ppr,step=ppr - 1))

index_info = 0
cnt_var = 0

epsilon = 1

