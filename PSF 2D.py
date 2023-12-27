#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 12:52:03 2023

@author: ckettmayer
"""
# import addcopyfighandler


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmasher as cmr



def I_gauss(x,y,xs,ys,I0,w0):
    return ((I0) * np.exp(- 2*((x-xs)**2+(y-ys)**2)/(w0**2))) 


def I_donut(x,y,xs,ys,I0,w0):
    return ((I0) * np.e * 2*((x-xs)**2+(y-ys)**2)/(w0**2)  * np.exp(- 2*((x-xs)**2+(y-ys)**2)/(w0**2))) 






xs, ys = 0, 0  #scanner centrado
I0 = 1
w0 = 300
 
#plot del volumen de excitación 3D
N = 100
s = 2*w0

# Genera datos para x, y, z
x = np.linspace(-s, s, N)
y = np.linspace(-s, s, N)


X, Y = np.meshgrid(x, y)
# I = I_gauss(X,Y,xs,ys,I0,w0)
I = I_donut(X,Y,xs,ys,I0,w0)

fig = plt.figure(figsize=(8, 6))
ax3d = plt.axes(projection="3d")


cm = 'plasma'
norm = mpl.colors.Normalize(vmin=0, vmax=np.amax(I))  #modifico la normalización de los colores para que vaya entre dos valores fijos
sm = mpl.cm.ScalarMappable(cmap=cm, norm=norm)
sm.set_array([])
# plt.colorbar(sm, pad=0.2, label='I(a.u.)')


ax3d.plot_surface(X,Y,I, cmap=cm, norm=norm)

ax3d.set_xlabel('x (um)') 
ax3d.set_ylabel('y (um)') 
ax3d.set_zlabel('I (a.u.)') 
ax3d.xaxis.pane.fill = False
ax3d.yaxis.pane.fill = False
ax3d.zaxis.pane.fill = False