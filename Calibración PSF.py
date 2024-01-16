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


#cargar imagen en escala de grises.

im = Image.open('orden1_continuo.tif')   
# im = Image.open('part_300nm.tif')  

plt.imshow(im)
plt.title('Original Image')


##COMPLETE##
pixel_size = 19    #(nm)   


#Convierto en un array la intensidad normalizada de la imagen
im_arr = np.asarray(im)
im_arr_nor = im_arr/np.max(im_arr)  #normalizo la intensidad


#Armo un mesh en xy para sobre eso graficar la intensidad como una superficie z=f(x,y)
N = len(im_arr_nor[:,1])
x = y = np.arange(0, N) * pixel_size
X, Y = np.meshgrid(x, y)
Z = im_arr_nor


#Para poder ajustar la superficie con curve fit hay que transformar a la forma:
xdata = np.vstack((X.ravel(), Y.ravel()))
ydata = Z.ravel()



def I_gauss(M,xs,ys,I0,w0):
    x, y = M
    return ((I0) * np.exp(- 2*((x-xs)**2+(y-ys)**2)/(w0**2))) 


def I_donut(M,xs,ys,I0,w0):
    x, y = M
    return ((I0) * np.e * 2*((x-xs)**2+(y-ys)**2)/(w0**2)  * np.exp(- 2*((x-xs)**2+(y-ys)**2)/(w0**2))) 



def I_fit(M,xs,ys,I0,w0):
    # return I_gauss(M,xs,ys,I0,w0)
    return I_donut(M,xs,ys,I0,w0)


#valores iniciales al ajuste suponiendo que la imagen está centrada en el máximo o mínimo
p0 = [int(N/2)*pixel_size, int(N/2)*pixel_size, 1, 900]
popt, pcov = curve_fit(I_fit, xdata, ydata, p0)

Z_fit = I_fit(xdata, *popt).reshape(N, N)

w0 = np.abs(popt[3])
w0_err = np.sqrt(pcov[3, 3])

fig = plt.figure(figsize=(10, 5))

ax1 = fig.add_subplot(121, projection="3d")
ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.9)
ax1.set_title('Original Image')
ax1.set_xlabel('x (mm)')
ax1.set_ylabel('y (mm)')
ax1.set_zlabel('I (a.u.)') 
ax1.xaxis.pane.fill = False
ax1.yaxis.pane.fill = False
ax1.zaxis.pane.fill = False

ax2 = fig.add_subplot(122, projection="3d")
ax2.plot_surface(X, Y, Z_fit, cmap='viridis', alpha=0.9)
ax2.set_title(f'Fit: w0 = ({w0:.2f} $\pm$ {w0_err:.2f}) mm')
ax2.set_xlabel('x (mm)')
ax2.set_ylabel('y (mm)')
ax2.set_zlabel('I (a.u.)') 
ax2.xaxis.pane.fill = False
ax2.yaxis.pane.fill = False
ax2.zaxis.pane.fill = False

fig.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.2)
plt.show()



