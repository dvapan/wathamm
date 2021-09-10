import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons


import sys

from poly import mvmonos, powers

import pandas as pd

from constants import *

import matplotlib.pyplot as plt

from wathamm import boundary_fnc

from wathamm import eq1_left, eq1_right
from wathamm import eq2_left, eq2_right

filename = sys.argv[1]

pc = np.loadtxt(filename)
print("Polynom approximate with: {}".format(pc[-1]))
pc = pc[:-1]

ppwrs = powers(max_poly_degree, 2)
psize = len(ppwrs)


cff_cnt = [psize,psize]

s,f = 0,cff_cnt[0]
p_cf = pc[s:f]
s,f = s+cff_cnt[0],f+cff_cnt[1]
v_cf = pc[s:f]

X = np.linspace(0, length, totalx)
T = np.array([0])

tt,xx = np.meshgrid(T,X)
in_pts = np.vstack([tt.flatten(),xx.flatten()]).T
tt,xx = np.meshgrid(T,X)

p = mvmonos(in_pts, ppwrs, [0, 0])
up = p.dot(p_cf)


# v = mvmonos(in_pts, ppwrs, [0, 0])
# uv = v.dot(v_cf)

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, bottom=0.25)

l, = plt.plot(X, up, lw=2, color='blue')
plt.axis([0, length, 0, p0*1.5])

axcolor = 'lightgoldenrodyellow'
axtime = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)

stime = Slider(axtime, 'Time', 0, total_time, valinit=0)



def update(val):
    t = stime.val
    T = np.array([t])

    tt,xx = np.meshgrid(T,X)
    in_pts = np.vstack([tt.flatten(),xx.flatten()]).T
    # tt,xx = np.meshgrid(T,X)

    p = mvmonos(in_pts, ppwrs, [0, 0])
    up = p.dot(p_cf)

    # v = mvmonos(in_pts, ppwrs, [0, 0])
    # uv = v.dot(v_cf)
    
    l.set_ydata(up)
    fig.canvas.draw_idle()
stime.on_changed(update)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

def reset(event):
    stime.reset()
button.on_clicked(reset)

plt.show()
