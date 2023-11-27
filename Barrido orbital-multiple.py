# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 17:25:31 2023

@author: ckettmayer
"""


### Plot intensity traces for multiple particle positions###

import addcopyfighandler
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

#I con haz gaussiano
def Iorb_gauss(A,theta,r,phi,I0,w0,B):
    return(I0*np.exp(-(2/w0**2)*(A**2+r**2-2*A*r*np.cos(theta-phi)))+B)

#I con haz donut
def Iorb_donut(A,theta,r,phi,I0,w0,B):
    return(I0*2*np.e*(A**2+r**2-2*A*r*np.cos(theta-phi))/(w0**2)*np.exp(-(2/w0**2)*(A**2+r**2-2*A*r*np.cos(theta-phi)))+B)

I0 = 1
w0 = 300
B = 0

A = 200
N = 100 #Cantidad de puntos en la órbita
theta = np.linspace(0, 2*np.pi,N)


l = 150   



#Particle positions
rm = [0, 25, 50, 75, 100]
phi = np.pi/2


values = np.linspace(0, 0.8, len(rm))
c = cm.plasma(values)




#%%

#Plot perfiles de intensidad
# fig = plt.figure(figsize=(6, 4))
# rplot = np.linspace(-2*w0,2*w0,200)
# plt.plot(rplot, Iorb_gauss(rplot,0,0,0,I0,w0,B), label='gauss')
# plt.plot(rplot, Iorb_donut(rplot,0,0,0,I0,w0,B), label='donut')
# plt.title(f'I0={I0}, w0={w0}nm')
# plt.ylabel('I (a.u.)')
# plt.xlabel('x (nm)')
# plt.grid()
# plt.tight_layout()
# plt.legend()
    


#%%


##ELEGIR EL HAZ QUE SE VA A GRAFICAR##
def Iorb(A,theta,r,phi,I0,w0,B):
    return Iorb_gauss(A,theta,r,phi,I0,w0,B)          #haz gaussiano con máximo central
    # return Iorb_donut(A,theta,r,phi,I0,w0,B)        #haz donut con mínimo central




fig = plt.figure(figsize=(10,4))
gs = fig.add_gridspec(1, 2, width_ratios=[3, 4], hspace=0.5, wspace=0.2)



# Polar plot
ax1 = fig.add_subplot(gs[0], projection='polar')
plt.subplots_adjust(top=0.8, left=0.05, hspace=0.9)

#Create a polar mesh for PSF plot
xplot = np.linspace(-1.5*l,1.5*l,150)
yplot = np.linspace(-1.5*l,1.5*l,200)
X, Y = np.meshgrid(xplot, yplot)
#Convert to polar coordinates
R, Phi = np.sqrt(X**2 + Y**2), np.arctan2(Y, X)

#Custom colormap white-red for PSF
color_max = 'red'  
color_min = 'white'
cmap_segments = {
    'red': [(0.0, 1.0, 1.0), (1.0, 1.0, 1.0)],  
    'green': [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)],  
    'blue': [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]  
}
custom_cmap = mcolors.LinearSegmentedColormap('custom_colormap', cmap_segments, 256)


ax1.scatter(Phi, R, c=Iorb(R, Phi, A, phi, I0, w0, B), marker='o', s=5, cmap = custom_cmap.reversed(), zorder=1)  #psf
ax1.scatter(theta[0:-1], theta[0:-1]*0+A, color='k', edgecolors='k', marker='o', s=50, alpha=0.3, zorder=2)       #orbit

for i in range(len(rm)):
    ax1.scatter(phi,rm[i], color=c[i], marker='*', s=250, edgecolors='k', zorder=3)           #particle
ax1.set_ylim(0, l+50)
ax1.set_yticks(np.arange(0, 200, 100))




# Intensity (theta)

ax2 = fig.add_subplot(gs[1])
plt.subplots_adjust(top=0.8, bottom=0.2,  right=0.95, hspace=0.9)

for i in range(len(rm)):
    colors = Iorb(A,theta,i,phi,I0,w0,B)
    ax2.scatter(theta*180/np.pi, Iorb(A,theta,rm[i],phi,I0,w0,B), color=c[i], marker='o', s=20, zorder=2)  #I(theta)

ax2.set_xlabel(r'$\theta$ ($^o$)')
ax2.set_ylabel('I (a.u.)')
ax2.set_ylim(0, I0+0.1*I0)
ax2.set_xticks(np.arange(0, 361, 30)) 
ax2.set_ylim(-0.1, I0+0.1)
ax2.grid()
 





if Iorb(0,0,0,0,I0,w0,B)>Iorb(0,0,20,0,I0,w0,B):   #para que se fije si estamos graficando gauss o donut
    haz = 'Gauss'
else:
    haz = 'Donut' 
     
# phi_pi= phi/np.pi

fig.suptitle(f'{haz}, w0={w0}nm, A={A}nm, r={rm}nm, N={N}')   # I0={I0},  B={B}, $\phi$={phi_pi:.2f}$\pi$rad





#%%
A=300
#Plot perfiles de intensidad
fig = plt.figure(figsize=(6, 4))
rplot = np.linspace(0,A,200)
plt.plot(rplot, Iorb_gauss(A,0,rplot,0,I0,w0,B), label='gauss', color='r')    #gauss
plt.plot(rplot, Iorb_donut(A,0,rplot,0,I0,w0,B), label='donut', color='r', linestyle='dashed')    #donut

plt.scatter(A, 0, color='k', edgecolors='k', marker='o', s=50, alpha=0.7, zorder=2)       #orbit
plt.scatter(0, 0, color='k', edgecolors='k', marker='o', s=50, alpha=0.7, zorder=2)       #orbit

for i in range(len(rm)):
    plt.scatter(rm[i], 0, color=c[i], marker='*', s=250, edgecolors='k', zorder=3)           #particle

plt.title(f'I0={I0}, w0={w0}nm, A={A}nm')
plt.ylabel('I (a.u.)')
plt.xlabel('r (nm)')
plt.grid()
plt.tight_layout()
plt.legend()


# Iorb(A,theta,r,phi,I0,w0,B):






