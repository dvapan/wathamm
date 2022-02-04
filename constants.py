import numpy as np
import more_itertools as mit

length = 500 #1400                     # Длина трубы          [м]
total_time = 1.5                         # Время расчета      [с]
timeclose = 0.125                     # Время закрытия от общего времени
d = 0.5
delta = 0.2
rho = 1000
K = 2030*10**6
E = 200000*10**6

lmd = 0.6  # 1*0.0558

c2 = 1/(rho/K + (rho*d)/(delta*E))

v0 = 1.0
p0 = 30*10**5


#xreg,treg = 10,20
#max_reg = xreg*treg
max_poly_degree = 3

index_info = 0
cnt_var = 0

epsilon = 0.001
a_0 = 0.1
a_cff = 1

accs = {
        "eq1": 5,   #5,   #1e+1,
        "eq2": 1500,#1500,#3e+3,
        "p":1e+4,
        "v":1e-2,
        }
