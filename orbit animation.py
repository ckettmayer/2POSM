# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 14:39:02 2023

@author: Constanza
"""

#animación orbital tracking


import addcopyfighandler
from matplotlib.widgets import Slider
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors


#I con haz gaussiano
def Iorb_gauss(A,theta,r,phi,I0,w0):
    return(I0*np.exp(-(2/w0**2)*(A**2+r**2-2*A*r*np.cos(theta-phi))))

#I con haz donut
def Iorb_donut(A,theta,r,phi,I0,w0):
    return(I0*2*np.e*(A**2+r**2-2*A*r*np.cos(theta-phi))/(w0**2)*np.exp(-(2/w0**2)*(A**2+r**2-2*A*r*np.cos(theta-phi))))




def Iorb(A,theta,r,phi,I0,w0):
    # return Iorb_gauss(A,theta,r,phi,I0,w0)         #haz gaussiano con máximo central
    return Iorb_donut(A,theta,r,phi,I0,w0)      #haz donut con mínimo central


I0 = 1
w0 = 150


N = 100 #

#posición de la partícula
r = 100
phi = np.pi/2 *3 


N = 24 #Orbit points

A = 150
theta = np.linspace(0, 2*np.pi, N)

l = 300  

#Create a polar mesh for PSF plot
xplot = np.linspace(-l, l, 100)
yplot = np.linspace(-l, l, 150)
X, Y = np.meshgrid(xplot, yplot)

R, Phi = np.sqrt(X**2 + Y**2), np.arctan2(Y, X)


colors = Iorb(A,theta,r,phi,I0,w0)

cm = 'viridis'
norm = mpl.colors.Normalize(vmin=0, vmax=I0)  
sm = mpl.cm.ScalarMappable(cmap=cm, norm=norm)
sm.set_array([])



#Custom colormap white-red for PSF
color_max = 'red'  
color_min = 'white'
cmap_segments = {
    'red': [(0.0, 1.0, 1.0), (1.0, 1.0, 1.0)],    
    'green': [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)],  
    'blue': [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]    
}
custom_cmap = mcolors.LinearSegmentedColormap('custom_colormap', cmap_segments, 256)





from matplotlib.animation import FuncAnimation

# Función para actualizar la animación en cada paso
def update(frame):
    # deletes subplot content
    ax1.cla()
    # ax2.cla()


    theta_m = theta[frame%24] 
    ax1.plot(theta, theta*0+A, color='grey', linestyle= 'dashed')
    ax1.scatter(Phi, R, c=Iorb(R, Phi, A, theta_m, I0, w0), marker='o', s=10, cmap = custom_cmap.reversed(), zorder=1)    #PSF
    ax1.scatter(phi, r, color='y', marker='*', s=400, edgecolors='k', zorder=3)     #particle
    
    ax2.scatter(theta_m, A, c=Iorb(A, theta_m, r, phi, I0, w0), marker='o', s=800, cmap=cm, norm=norm, alpha=1)   #orbit
    
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax1.set_ylim(0, l)
    ax2.set_ylim(0, l)



fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121,projection='polar')
ax2 = fig.add_subplot(122, projection='polar')



ax1.set_xticks([])
ax1.set_yticks([])
ax2.set_xticks([])
ax2.set_yticks([])
ax1.set_ylim(0, l)
ax2.set_ylim(0, l)


# Crea la animación
animation = FuncAnimation(fig, update, frames=100, interval=500)
# animation.save('mi_animacion_gauss.gif', writer='pillow', fps=30)

# Muestra la animación
plt.show()
