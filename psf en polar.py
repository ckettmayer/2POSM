# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 11:27:03 2023

@author: Constanza
"""


# import addcopyfighandler
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors


#I con haz gaussiano
def Iorb_gauss(A,theta,r,phi,I0,w0,B):
    return(I0*np.exp(-(2/w0**2)*(A**2+r**2-2*A*r*np.cos(theta-phi)))+B)

#I con haz donut
def Iorb_donut(A,theta,r,phi,I0,w0,B):
    return(I0*2*np.e*(A**2+r**2-2*A*r*np.cos(theta-phi))/(w0**2)*np.exp(-(2/w0**2)*(A**2+r**2-2*A*r*np.cos(theta-phi)))+B)



##ELEGIR EL HAZ QUE SE VA A GRAFICAR##

def Iorb(A,theta,r,phi,I0,w0,B):
    return Iorb_gauss(A,theta,r,phi,I0,w0,B)          #haz gaussiano con máximo central
    # return Iorb_donut(A,theta,r,phi,I0,w0,B)        #haz donut con mínimo central




I0 = 1
w0 = 150
B = 0

A = 150
N = 100 #Cantidad de puntos en la órbita
theta = np.linspace(0, 2*np.pi, N)

r = 0.1
phi = 90


fig = plt.figure(figsize=(7, 9))
ax = fig.add_subplot(projection='polar')



#colormap for orbit
cm = 'viridis'
norm = mpl.colors.Normalize(vmin=0, vmax=I0)  #modifico la normalización de los colores para que vaya entre dos valores fijos
sm = mpl.cm.ScalarMappable(cmap=cm, norm=norm)
sm.set_array([])


#Create a polar mesh for PSF plot
xplot = np.linspace(0,360,100)
yplot = np.linspace(0, 2*A, 100)
X, Y = np.meshgrid(xplot, yplot)

#Convert to polar coordinates
R, Phi = np.sqrt(X**2 + Y**2), np.arctan2(Y, X)
color2 = Iorb(A, np.pi/2, R, Phi, I0, w0, B)

#Custom colormap white-red for PSF
color_max = 'red'  # Puedes usar códigos hexadecimales o nombres de colores estándar
color_min = 'white'
cmap_segments = {
    'red': [(0.0, 1.0, 1.0), (1.0, 1.0, 1.0)],  # Rojo a Blanco en el canal red
    'green': [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)],  # Sin cambio en el canal green
    'blue': [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]  # Sin cambio en el canal blue
}
custom_cmap = mcolors.LinearSegmentedColormap('custom_colormap', cmap_segments, 256)

# contour = ax.contour(Phi, R, color2, levels=10, cmap='viridis')

#PLOT PSF
ax.scatter(Phi, R, c=color2, marker='o', s=5, cmap = custom_cmap.reversed(), zorder=1)                 

# ax.scatter(Phi, R, c=color2, marker='o', s=250, cmap = 'plasma', zorder=1)                 


# ax.scatter(theta, theta*0+A, c=Iorb(A,theta,r,phi,I0,w0,B), marker='o', s=100, cmap=cm, norm=norm, alpha=1, zorder=4)         #orbit
# ax.scatter(phi, r, color='y', marker='*', s=250, edgecolors='k', zorder=3)                                                    #particle


ax.set_ylim(0, 2*A)




fig2 = plt.figure(figsize=(5, 5))
ax = fig2.add_subplot()
ax.scatter(X, Y, c=color2, marker='o', s=5, cmap = custom_cmap.reversed(), zorder=1)

# ax.scatter(X, Y, c=color2, marker='o', s=250, cmap = 'plasma', zorder=1)    

# contour = ax.contour(X, Y, color2, levels=10, cmap='viridis')             
plt.grid()
plt.xlim(-200,200)
plt.ylim(-100,300)



    
    
    
    
    
