# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 16:12:28 2023

@author: ckettmayer
"""


# import addcopyfighandler
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


I0 = 5
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

    ax1.scatter(X, Y, c=Iorb(xp,yp,X,Y,I0,w0), marker='o', s=10, cmap = custom_cmap.reversed(), zorder=1) #PSF
    ax1.scatter(x, y, color='y', marker='*', s=500, edgecolors='k', zorder=3)   #Particle

    ax2.scatter(xp, yp, c=Iorb(xp,yp,x,y,I0,w0), cmap = cm, norm = norm, marker='s', s=1050) #Image
    
    
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


#%%

def Iorb(xs,ys,x,y,I0,w0):
    return Iorb_gauss(xs,ys,x,y,I0,w0)         #haz gaussiano con máximo central
    # return Iorb_donut(xs,ys,x,y,I0,w0)       #haz donut con mínimo central

I0 = 0.7
w0 = 150


N = 100 #

#posición de la partícula
x = 300
y = 300

xs = 250
ys = 250

A = 500



image_size=24
px_size=175

#Create a polar mesh for image plot
xi = np.linspace(0,A,image_size)
yi = np.linspace(0,A,image_size)
Xi, Yi = np.meshgrid(xi, yi)


fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(8, 4))

norm = mpl.colors.Normalize(vmin=0, vmax=1)
ax3.scatter(X, Y, c=Iorb(x,y,X,Y,I0,w0), marker='o', s=10, cmap = custom_cmap.reversed(), norm=norm, zorder=1) #PSF
ax3.scatter(x, y, color='y', marker='*', s=500, edgecolors='k', zorder=3)   #Particle


ax4.scatter(Xi, Yi, c=Iorb(Xi,Yi,x,y,I0,w0), cmap = cm, norm = norm, marker='s', s=px_size) #Image

ax3.set_ylim(0,A)
ax3.set_xlim(0,A)
ax4.set_ylim(0,A)
ax4.set_xlim(0,A)
ax4.set_title(f'{image_size}x{image_size}')
ax3.set_xticks([])
ax3.set_yticks([])
# ax4.set_xticks([])
# ax4.set_yticks([])

ax3.set_aspect('equal')
ax4.set_aspect('equal')


#%%

#Plot perfiles de intensidad
fig = plt.figure(figsize=(6, 4))
rplot = np.linspace(0,2*w0,200)
# plt.plot(rplot, Iorb_gauss(rplot,0,0,0,1,w0), label='I0=1')
# plt.plot(rplot, Iorb_gauss(rplot,0,0,0,0.7,w0), label='I0=0.7')

I1 = 1
I2 = 0.5

from scipy.integrate import quad

rcolor = np.linspace(0,2*w0)
rint = np.zeros(len(rcolor))
N  = 20
limsup = None 

for i in range(len(rcolor)):
    rint[i], _ = quad(lambda r: Iorb_donut(r, 0, 0, 0, I1, w0), 0, rcolor[i])

    if rint[i]>= N:
        limsup1 = rcolor[i]
        break
    
dif1 = - Iorb_donut(0,0,0,0,I1,w0) + Iorb_donut(limsup1,0,0,0,I1,w0)
    
# Colorear el área bajo la curva hasta limsup
# plt.fill_betweenx(Iorb_donut(rplot, 0, 0, 0, 1, w0), 0, limsup, alpha=0.3, label=f'Área hasta {limsup}')
plt.fill_between(rcolor, 0, Iorb_donut(rcolor, 0, 0, 0, I1, w0), where=(rcolor <= limsup1), alpha=0.3, label=f'Área hasta N={N}, dif={dif1:.2f}')
    



limsup2 = None 
for i in range(len(rcolor)):
    # rint[i] = quad(Iorb_donut(rplot,0,0,0,1,w0), 0, rcolor[i])
    rint[i], _ = quad(lambda r: Iorb_donut(r, 0, 0, 0, I2, w0), 0, rcolor[i])

    if rint[i]>= N:
        limsup2 = rcolor[i]
        break

dif2 = - Iorb_donut(0,0,0,0,I2,w0) + Iorb_donut(limsup2,0,0,0,I2,w0)
# Colorear el área bajo la curva hasta limsup
plt.fill_between(rcolor, 0, Iorb_donut(rcolor, 0, 0, 0, I2, w0), where=(rcolor <= limsup2), alpha=0.3, label=f'Área hasta N={N}, dif={dif2:.2f}')

plt.plot(rplot, Iorb_donut(rplot,0,0,0,I1,w0), label='I0=1')
plt.hlines(Iorb_donut(0,0,0,0,I1,w0), 0 , 1.5*limsup1, color='k', alpha=0.4)
plt.hlines(Iorb_donut(limsup1,0,0,0,I1,w0), 0 , 1.5*limsup1, color='k', alpha=0.4)

plt.plot(rplot, Iorb_donut(rplot,0,0,0,I2,w0), label='I0=0.7')
plt.hlines(Iorb_donut(0,0,0,0,I2,w0), 0 , 1.5*limsup2, color='k', alpha=0.4)
plt.hlines(Iorb_donut(limsup2,0,0,0,I2,w0), 0 , 1.5*limsup2, color='k', alpha=0.4)


plt.title(f'w0={w0}nm')
plt.ylabel('I (a.u.)')
plt.xlabel('x (nm)')
plt.ylim(0,1.1)
plt.grid()
plt.tight_layout()
plt.legend()
    

#%%

#Plot perfiles de intensidad
fig = plt.figure(figsize=(6, 4))
rplot = np.linspace(0,2*w0,200)

I1 = 1
I2 = 0.5

from scipy.integrate import quad

rcolor = np.linspace(0,2*w0)
rint = np.zeros(len(rcolor))
N  = 20
limsup = None 

for i in range(len(rcolor)):
    # rint[i] = quad(Iorb_donut(rplot,0,0,0,1,w0), 0, rcolor[i])
    rint[i], _ = quad(lambda r: Iorb_gauss(r, 0, 0, 0, I1, w0), 0, rcolor[i])

    if rint[i]>= N:
        limsup1 = rcolor[i]
        break
    
dif1 = Iorb_gauss(0,0,0,0,I1,w0) - Iorb_gauss(limsup1,0,0,0,I1,w0)
    
# Colorear el área bajo la curva hasta limsup
plt.fill_between(rcolor, 0, Iorb_gauss(rcolor, 0, 0, 0, I1, w0), where=(rcolor <= limsup1), alpha=0.3, label=f'Área hasta N={N}, dif={dif1:.2f}')



limsup2 = None 
for i in range(len(rcolor)):
    # rint[i] = quad(Iorb_donut(rplot,0,0,0,1,w0), 0, rcolor[i])
    rint[i], _ = quad(lambda r: Iorb_gauss(r, 0, 0, 0, I2, w0), 0, rcolor[i])

    if rint[i]>= N:
        limsup2 = rcolor[i]
        break
    
dif2 = Iorb_gauss(0,0,0,0,I2,w0) - Iorb_gauss(limsup2,0,0,0,I2,w0)

# Colorear el área bajo la curva hasta limsup
plt.fill_between(rcolor, 0, Iorb_gauss(rcolor, 0, 0, 0, I2, w0), where=(rcolor <= limsup2), alpha=0.3, label=f'Área hasta N={N}, dif={dif2:.2f}')


plt.plot(rplot, Iorb_gauss(rplot,0,0,0,I1,w0), label='I0=1')
plt.hlines(Iorb_gauss(0,0,0,0,I1,w0), 0 , 1.5*limsup1, color='k', alpha=0.4)
plt.hlines(Iorb_gauss(limsup1,0,0,0,I1,w0), 0 , 1.5*limsup1, color='k', alpha=0.4)

plt.plot(rplot, Iorb_gauss(rplot,0,0,0,I2,w0), label='I0=0.7')
plt.hlines(Iorb_gauss(0,0,0,0,I2,w0), 0 , 1.5*limsup2, color='k', alpha=0.4)
plt.hlines(Iorb_gauss(limsup2,0,0,0,I2,w0), 0 , 1.5*limsup2, color='k', alpha=0.4)
plt.title(f'w0={w0}nm')
plt.ylabel('I (a.u.)')
plt.xlabel('x (nm)')
plt.ylim(0,1.1)
plt.grid()
plt.tight_layout()
plt.legend()
    

