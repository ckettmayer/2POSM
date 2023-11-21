#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 21:21:58 2023

@author: ckettmayer
"""

### PSF calibration (known pixel size) ###

import addcopyfighandler
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
# import cmasher as cmr
from scipy.optimize import curve_fit


im = Image.open('part_300nm.tif')   
plt.imshow(im)
plt.title('Original Image')


##COMPLETE##
pixel_size = 50    #(nm)   



im_arr = np.asarray(im)
im_arr_nor = im_arr/np.max(im_arr)  #normalizo la intensidad


N = len(im_arr_nor[:,1])
x = y = np.arange(0, N) * pixel_size
X, Y = np.meshgrid(x, y)
Z = im_arr_nor

xdata = np.vstack((X.ravel(), Y.ravel()))
ydata = Z.ravel()

def I_gauss(M,xs,ys,I0,w):
    x, y = M
    return ((2*I0) / (np.pi) * np.exp(- 2* ((x-xs)**2+(y-ys)**2)/(w**2)))

p0 = [0, 0, 1, 300]
popt, pcov = curve_fit(I_gauss, xdata, ydata, p0)

Z_fit = I_gauss(xdata, *popt).reshape(N, N)

w = np.abs(popt[3])
w_err = np.sqrt(pcov[3, 3])

fig = plt.figure(figsize=(10, 5))

ax1 = fig.add_subplot(121, projection="3d")
ax1.plot_surface(X, Y, Z, cmap='viridis')
ax1.set_title('Original Image')
ax1.set_xlabel('x (nm)')
ax1.set_ylabel('y (nm)')
ax1.set_zlabel('I (a.u.)') 
ax1.xaxis.pane.fill = False
ax1.yaxis.pane.fill = False
ax1.zaxis.pane.fill = False

ax2 = fig.add_subplot(122, projection="3d")
ax2.plot_surface(X, Y, Z_fit, cmap='viridis')
ax2.set_title(f'Fit: w = ({w:.2f} $\pm$ {w_err:.2f}) nm')
ax2.set_xlabel('x (nm)')
ax2.set_ylabel('y (nm)')
ax2.set_zlabel('I (a.u.)') 
ax2.xaxis.pane.fill = False
ax2.yaxis.pane.fill = False
ax2.zaxis.pane.fill = False

fig.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.2)
plt.show()



