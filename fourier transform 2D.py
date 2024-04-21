#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 18:52:22 2024

@author: ckettmayer
"""


#Transformada de fourier

# import addcopyfighandler
from matplotlib.widgets import Slider
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors



def I_gauss(x,y,xs,ys,I0,w0):
    return ((I0) * np.exp(- 2*((x-xs)**2+(y-ys)**2)/(w0**2))) 


def I_donut(x,y,xs,ys,I0,w0):
    return ((I0) * np.e * 2*((x-xs)**2+(y-ys)**2)/(w0**2)  * np.exp(- 2*((x-xs)**2+(y-ys)**2)/(w0**2))) 




def Iorb(x,y,xs,ys,I0,w0):
    # return I_gauss(x,y,xs,ys,I0,w0)        #haz gaussiano con máximo central
    return I_donut(x,y,xs,ys,I0,w0)     #haz donut con mínimo central



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

I = Iorb(X,Y,xs,ys,I0,w0)

fourier_I = np.fft.fftshift(np.fft.fft2(I))

'''
fig, axes = plt.subplots(1, 4, figsize=(15, 4), gridspec_kw={'width_ratios': [1, 1, 1, 0.1], 'wspace': 0.5})
'''



fig = plt.figure(figsize=(8, 6))
ax1 = plt.axes()
ax2 = plt.axes()


cm = 'plasma'
norm = mpl.colors.Normalize(vmin=0, vmax=np.amax(I))  #modifico la normalización de los colores para que vaya entre dos valores fijos
sm = mpl.cm.ScalarMappable(cmap=cm, norm=norm)
sm.set_array([])
# plt.colorbar(sm, pad=0.2, label='I(a.u.)')

# ax1.imshow(I, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', cmap='plasma')

# ax1.set_xlabel('x (um)') 
# ax1.set_ylabel('y (um)') 
# ax1.set_title('Haz original')




# Grafica la intensidad como una imagen
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(I, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', cmap=cm)
plt.colorbar(label='Intensidad')
plt.xlabel('x (nm)')
plt.ylabel('y (nm)')
plt.title('Intensidad de la función I(x, y)')

# Grafica la magnitud de la transformada de Fourier como una imagen
plt.subplot(1, 2, 2)
plt.imshow(np.abs(fourier_I), cmap=cm, extent=[-np.pi, np.pi, -np.pi, np.pi])
plt.colorbar(label='Magnitud de la Transformada de Fourier')
plt.xlabel('kx')
plt.ylabel('ky')
plt.title('Transformada de Fourier de la función I(x, y)')

plt.tight_layout()
plt.show()

