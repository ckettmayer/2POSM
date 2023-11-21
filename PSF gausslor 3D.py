# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 12:55:45 2023

@author: ckettmayer
"""

#Gaussian lorentzian beam 3D

import numpy as np
import matplotlib.pyplot as plt
import addcopyfighandler
import matplotlib as mpl
import cmasher as cmr


#PSF para microscopio de absorción de dos fotones

#Según paper de orbital tracking de Katarina y Enrico 2004, que creo que está mal
def I_gausslor_kat(x,y,z,xs,ys,zs,I0,l,w):
    return ( (I0/(1+(((z-zs)**2*l**2)/(np.pi**2*w**2)))**2) * np.exp(-(4* ((x-xs)**2 + (y-ys)**2)) / w**2) )

#Según paper de microscopía de 2 fotones de Berland 1995
def I_gausslor_ber(x,y,z,xs,ys,zs,I0,l,w):
    return ( (2*I0) / (np.pi*(1+ ((z-zs)**2*l**2)/(np.pi**2 *w**4) )) * np.exp(- 2* ((x-xs)**2+(y-ys)**2)/(w**2*(1+((z-zs)*l/(np.pi**2 *w**2))**2)) ) )


xs, ys, zs = 0 , 0 , 0  #scanner centrado
I0 = 1
l = 780
w = 300
 
#plot del volumen de excitación 3D
N = 100
s = 4*w

# Genera datos para x, y, z
x = np.linspace(-s, s, N)
y = np.linspace(-s, s, N)
z = np.linspace(-s, s, N)

# Crea una malla tridimensional
X, Y, Z = np.meshgrid(x, y, z)

# I = I_gausslor_kat(X, Y, Z, xs, ys, zs, I0, l, w)
I = I_gausslor_ber(X, Y, Z, xs, ys, zs, I0, l, w)




# fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig, axes = plt.subplots(1, 4, figsize=(15, 4), gridspec_kw={'width_ratios': [1, 1, 1, 0.1], 'wspace': 0.5})

cm = cmr.ember
norm = mpl.colors.Normalize(vmin=0, vmax=np.amax(I))  #modifico la normalización de los colores para que vaya entre dos valores fijos
sm = mpl.cm.ScalarMappable(cmap=cm, norm=norm)
sm.set_array([])

# Plano XY
axes[0].imshow(I[:, :, 50], extent=[x.min(), x.max(), y.min(), y.max()], cmap=cm, norm=norm)  #uso el colormap cm con la norma norm
axes[0].set_title('Plano XY (Z=0)')
axes[0].set_xlabel('X (nm)')
axes[0].set_ylabel('Y (nm)')
axes[0].grid(False)

# Plano XZ
axes[1].imshow(I[50, :, :].T, extent=[z.min(), z.max(), x.min(), x.max()], cmap=cm, norm=norm)  
axes[1].set_title('Plano XZ (Y=0)')
axes[1].set_xlabel('X (nm)')
axes[1].set_ylabel('Z (nm)')
axes[1].grid(False)

# Plano YZ
axes[2].imshow(I[:, 50, :].T, extent=[z.min(), z.max(), y.min(), y.max()], cmap=cm, norm=norm)
axes[2].set_title('Plano YZ (X=0)')
axes[2].set_xlabel('Y (nm)')
axes[2].set_ylabel('Z (nm)')
axes[2].grid(False)

cax = axes[3]
cbar = plt.colorbar(sm, cax=cax, pad=0.1, label='I (a.u.)', orientation='vertical')

fig.suptitle(r'$\lambda$='+str(l)+'nm, w$_0$='+str(w)+'nm')

plt.tight_layout()
plt.show()






