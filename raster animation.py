# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 16:12:28 2023

@author: ckettmayer
"""


import addcopyfighandler
from matplotlib.widgets import Slider
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors


#I con haz gaussiano
def Iorb_gauss(xs,ys,x,y,I0,w0):
    return(I0*np.exp(-(2/w0**2)*((x-xs)**2+(y-ys)**2)))

#I con haz donut
def Iorb_donut(xs,ys,x,y,I0,w0):
    return(I0*2*np.e*((x-xs)**2+(y-ys)**2)/(w0**2)*np.exp(-(2/w0**2)*((x-xs)**2+(y-ys)**2)))



def Iorb(xs,ys,x,y,I0,w0):
    return Iorb_gauss(xs,ys,x,y,I0,w0)         #haz gaussiano con máximo central
    # return Iorb_donut(xs,ys,x,y,I0,w0)       #haz donut con mínimo central


I0 = 1
w0 = 150


N = 100 #

#posición de la partícula
x = 300
y = 300

xs = 250
ys = 250

A = 500



#Create a polar mesh for PSF plot
xplot = np.linspace(0,A,100)
yplot = np.linspace(0,A,100)
X, Y = np.meshgrid(xplot, yplot)



colors = Iorb(xs,ys,x,y,I0,w0)

cm = 'viridis'
norm = mpl.colors.Normalize(vmin=0, vmax=I0)  #modifico la normalización de los colores para que vaya entre dos valores fijos
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
    # Borra el contenido de los subplots
    ax1.cla()
    # ax2.cla()
    
    xp = 60* (frame % 10)
    yp =  A - (frame // 10)*10 *6

    ax1.scatter(X, Y, c=Iorb(xp,yp,X,Y,I0,w0), marker='o', s=10, cmap = custom_cmap.reversed(), zorder=1) 
    ax1.scatter(x, y, color='y', marker='*', s=500, edgecolors='k', zorder=3)   


    ax2.scatter(xp, yp, c=Iorb(xp,yp,x,y,I0,w0), cmap = cm, norm = norm, marker='s', s=1050)
    ax1.set_ylim(0,A)
    ax1.set_xlim(0,A)
    ax2.set_ylim(-30,A+30)
    ax2.set_xlim(-30,A+30)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax1.set_aspect('equal')
    ax2.set_aspect('equal')

# Configura la figura y los subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.set_ylim(0,A)
ax1.set_xlim(0,A)
ax2.set_ylim(-30,A+50)
ax2.set_xlim(-30,A+50)
ax1.set_xticks([])
ax1.set_yticks([])
ax2.set_xticks([])
ax2.set_yticks([])

ax1.set_aspect('equal')
ax2.set_aspect('equal')

# Configura el título de la figura
# fig.suptitle('Animación en Matplotlib')

# Crea la animación
animation = FuncAnimation(fig, update, frames=100, interval=500)
# animation.save('mi_animacion_gauss.gif', writer='pillow', fps=30)

# Muestra la animación
plt.show()










