import numpy as np
import more_itertools as mit

length = 1400                     # Длина трубы          [м]
total_time = 4                         # Время расчета      [с]
timeclose = 0.125                     # Время закрытия от общего времени
d = 0.5
delta = 0.2
rho = 1000
K = 2030*10**6
E = 200000*10**6

lmd = 0.6  # 1*0.0558

c2 = 1/(rho/K + (rho*d)/(delta*E))

v0 = 1.0
p0 = 10*10**6


xreg,treg = 6,12
max_reg = xreg*treg
max_poly_degree = 3

#xreg,treg = 1,1
#max_reg = xreg*treg
#max_poly_degree = 5

dx = length/xreg
dt = total_time/treg


index_info = 0
cnt_var = 0

epsilon = 1
a = 0.3
