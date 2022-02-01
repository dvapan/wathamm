import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons


import sys

from poly import mvmonos, powers

import pandas as pd

from constants import *

import matplotlib.pyplot as plt

from model import boundary_fnc

from model import eq1_left, eq1_right
from model import eq2_left, eq2_right
from model import make_id, psize

def get_pv(pc, in_pts,p):
    ids = in_pts // np.array([ltreg,lxreg])
    pids = np.apply_along_axis(lambda x: int(make_id(*x,p)),1,ids)
    cf = pc[pids]
    p_cf,v_cf = np.hsplit(cf, 2)
    p = mvmonos(in_pts, ppwrs, [0, 0])
    up = np.sum(p*p_cf, axis=1)
    uv = np.sum(p*v_cf, axis=1)
    return up,uv


parser = argparse.ArgumentParser()
parser.add_argument("filenames", nargs='*')
parser.add_argument("--xreg", default=1,type=int)
parser.add_argument("--treg", default=1,type=int)
parser.add_argument("--pprx", default=7,type=int)
parser.add_argument("--pprt", default=7,type=int)
args = parser.parse_args(sys.argv[1:])
p = vars(args)
xreg = args.xreg
treg = args.treg
pprx = args.pprx
pprt = args.pprt

totalx = xreg*pprx - xreg + 1
totalt = treg*pprt - treg + 1
X = np.linspace(0, length, totalx)
T = np.linspace(0, total_time, totalt)
X_part = list(mit.windowed(X,n=pprx,step=pprx - 1))
T_part = list(mit.windowed(T,n=pprt,step=pprt - 1))
lxreg = X_part[0][-1] - X_part[0][0]
ltreg = T_part[0][-1] - T_part[0][0]

pcs = []
for filename in args.filenames:
    pc = np.loadtxt(filename)
    print("Polynom approximate with: {}".format(pc[-1]))
    pc = pc[:-1]
    pc = pc.reshape(-1,psize*2)
    pcs.append(pc)

ppwrs = powers(max_poly_degree, 2)
psize = len(ppwrs)

cff_cnt = [psize,psize]

X = np.array([0])
T = np.arange(0, total_time, 0.01)

fig, axs = plt.subplots(2)
plt.subplots_adjust(left=0.1, bottom=0.25)

tt,xx = np.meshgrid(T,X)
in_pts = np.vstack([tt.flatten(),xx.flatten()]).T
lp = []
lv = []
for pc in pcs:
    up,uv = get_pv(pc,in_pts,p)
    l1, = axs[0].plot(T, up, lw=2)
    l2, = axs[1].plot(T, uv, lw=2)
    lp.append(l1)
    lv.append(l2)
axs[0].axis([0, total_time, p0*0.8, p0*1.5])
axs[1].axis([0, total_time, -2*v0, 2*v0])

axcolor = 'lightgoldenrodyellow'
axtime = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)

spos = Slider(axtime, 'position', 0, length-0.001, valinit=0)

def update(val):
    x = spos.val
    X = np.array([x])

    tt,xx = np.meshgrid(T,X)
    in_pts = np.vstack([tt.flatten(),xx.flatten()]).T
    for i,pc in enumerate(pcs):
        up,uv = get_pv(pc,in_pts,p) 
        lp[i].set_ydata(up)
        lv[i].set_ydata(uv)
    fig.canvas.draw_idle()
spos.on_changed(update)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

def reset(event):
    spos.reset()
button.on_clicked(reset)

plt.show()
