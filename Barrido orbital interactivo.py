# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 11:40:22 2023

@author: ckettmayer
"""

from matplotlib.widgets import Slider
import addcopyfighandler
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

#I con haz gaussiano
def Iorb_gauss(A,theta,r,phi,I0,w0,B):
    return(I0*np.exp(-(2/w0**2)*(A**2+r**2-2*A*r*np.cos(theta-phi)))+B)

#I con haz donut
def Iorb_donut(A,theta,r,phi,I0,w0,B):
    return(I0*2*np.e*(A**2+r**2-2*A*r*np.cos(theta-phi))/(w0**2)*np.exp(-(2/w0**2)*(A**2+r**2-2*A*r*np.cos(theta-phi)))+B)


I0 = 1
w0 = 300
B = 0

A = 150
N = 100 #Cantidad de puntos en la órbita
theta = np.linspace(0, 2*np.pi, N)

r = 0.1
phi = 0


fig = plt.figure(figsize=(5, 5))
# plt.plot(theta,Iorb_gauss(A,theta,r,phi,I0,w0,B))

rplot = np.linspace(0,2*w0,100)
plt.plot(rplot, Iorb_gauss(rplot,0,0,0,I0,w0,B), label='gauss')
plt.plot(rplot, Iorb_donut(rplot,0,0,0,I0,w0,B), label='donut')
plt.title(f'I0={I0}, w0={w0}nm, (A,theta)=(0,0), (r,phi)=(0,0)')
plt.ylabel('I (a.u.)')
plt.xlabel('x (nm)')
plt.grid()
plt.legend()
    



#%%

##ELEGIR EL HAZ QUE SE VA A GRAFICAR##

def Iorb(A,theta,r,phi,I0,w0,B):
    return Iorb_gauss(A,theta,r,phi,I0,w0,B)          #haz gaussiano con máximo central
    # return Iorb_donut(A,theta,r,phi,I0,w0,B)        #haz donut con mínimo central


colors = Iorb(A,theta,r,phi,I0,w0,B)

cm = 'viridis'
norm = mpl.colors.Normalize(vmin=0, vmax=I0)  #modifico la normalización de los colores para que vaya entre dos valores fijos
sm = mpl.cm.ScalarMappable(cmap=cm, norm=norm)
sm.set_array([])



fig = plt.figure( figsize=(5, 6))
ax = fig.add_subplot(projection='polar')

#Sliders
ax_parameter_r = plt.axes([0.17, 0.15, 0.65, 0.03], facecolor='lightgoldenrodyellow')   #[left, bottom, width, height]
slider_r = Slider(ax_parameter_r, 'r', 0, A, valinit=r)
ax_parameter_phi = plt.axes([0.17, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider_phi = Slider(ax_parameter_phi, 'phi', 0, 2*np.pi, valinit=phi)

ax.scatter(theta, theta*0+A, c=colors, marker='o', s=100, cmap=cm, norm=norm, alpha=1)
ax.scatter(phi, r, color='y', marker='*', s=250, edgecolors='k', zorder=3)
ax.set_ylim(0, A+0.1*A)


# Función llamada al cambiar el valor del slider
def update(val):
    ax.clear() 
    r = slider_r.val
    phi = slider_phi.val
    colors = Iorb(A,theta,r,phi,I0,w0,B)
    ax.scatter(theta, theta*0+A, c=colors, marker='o', s=100, cmap=cm, norm=norm, alpha=1)
    ax.scatter(phi, r, color='y', marker='*', s=250, edgecolors='k', zorder=3)
    ax.set_ylim(0, A+0.1*A)
    fig.canvas.draw_idle()
    
    
cbar = fig.colorbar(sm, ax=ax, pad=0.1, shrink=0.5)
cbar.set_label('Intensidad')    

# Conectar el slider con la función de actualización
slider_r.on_changed(update)
slider_phi.on_changed(update)
fig.tight_layout()


if Iorb(0,0,0,0,I0,w0,B)>Iorb(0,0,20,0,I0,w0,B):   #para que se fije si estamos graficando gauss o donut
    haz = 'Haz gaussiano (max central)'
else:
    haz = 'Haz donut (min central)'
       
fig.suptitle(f'{haz} \n I0={I0}, w0={w0}nm, B={B}, N={N}, A={A}nm')



