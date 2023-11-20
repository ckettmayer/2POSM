# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 17:25:31 2023

@author: ckettmayer
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import addcopyfighandler


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
theta = np.linspace(0, 2*np.pi,N)

r = 100
phi = np.pi/2

#%%

#Plot perfiles de intensidad
fig = plt.figure(figsize=(5, 5))
rplot = np.linspace(0,2*w0,100)
plt.plot(rplot, Iorb_gauss(rplot,0,0,0,I0,w0,B), label='gauss')
plt.plot(rplot, Iorb_donut(rplot,0,0,0,I0,w0,B), label='donut')
plt.title(f'I0={I0}, w0={w0}nm, (A,theta)=(0,0), (r,phi)=(0,0)')
plt.ylabel('I (a.u.)')
plt.xlabel('x (nm)')
plt.grid()
plt.legend()
    


#%%

#Intensidad en la órbita en polares 

##ELEGIR EL HAZ QUE SE VA A GRAFICAR##
def Iorb(A,theta,r,phi,I0,w0,B):
    # return Iorb_gauss(A,theta,r,phi,I0,w0,B)          #haz gaussiano con máximo central
    return Iorb_donut(A,theta,r,phi,I0,w0,B)        #haz donut con mínimo central


colors = Iorb(A,theta,r,phi,I0,w0,B)

cm = 'viridis'
norm = mpl.colors.Normalize(vmin=0, vmax=I0)  #modifico la normalización de los colores para que vaya entre dos valores fijos
sm = mpl.cm.ScalarMappable(cmap=cm, norm=norm)
sm.set_array([])


fig = plt.figure(figsize=(7,7))
gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.3, wspace=0.2)


ax1 = fig.add_subplot(gs[0], projection='polar')
ax1.scatter(theta, theta*0+A, c=colors, marker='o', s=100, cmap=cm, norm=norm)
ax1.scatter(phi,r, color='y', marker='*', s=250, edgecolors='k', zorder=3)
ax1.set_ylim(0, 175)


cbar = fig.colorbar(sm, ax=ax1, pad=0.1)
cbar.set_label('Intensidad')

if Iorb(0,0,0,0,I0,w0,B)>Iorb(0,0,20,0,I0,w0,B):   #para que se fije si estamos graficando gauss o donut
    haz = 'Gauss'
else:
    haz = 'Donut'
       
phi_pi= phi/np.pi

fig.suptitle(f'{haz}, I0={I0}, w0={w0}nm, B={B}, N={N}, A={A}nm, r={r}nm, $\phi$={phi_pi:.2f}$\pi$rad')


#Intensidad

ax2 = fig.add_subplot(gs[1])
ax2.grid()
ax2.scatter(theta, Iorb(A,theta,r,phi,I0,w0,B), c=colors, marker='o', s=100, cmap=cm, norm=norm)
# ax2.scatter(phi,0, color='y', marker='*', s=250, edgecolors='k', zorder=3)
ax2.set_xlabel(r'$\theta$ (rad)')
ax2.set_ylabel('I (a.u.)')
ax2.set_ylim(0, I0+0.1*I0)
# ax2.set_ylim(0.7, 0.9)
