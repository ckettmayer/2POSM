# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 11:23:06 2023

@author: ckettmayer
"""

import addcopyfighandler
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.fft import fft



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

#POSICION DE LA PARTÍCULA
r = 130
phi = np.pi/2


##ELEGIR EL HAZ QUE SE VA A GRAFICAR##
def Iorb(A,theta,r,phi,I0,w0,B):
    return Iorb_gauss(A,theta,r,phi,I0,w0,B)          #haz gaussiano con máximo central
    # return Iorb_donut(A,theta,r,phi,I0,w0,B)        #haz donut con mínimo central

x = theta
y = Iorb(A,theta,r,phi,I0,w0,B)



fft_values = fft(y)

n = 4
a = np.zeros(n)
b = np.zeros(n)


fourier_approx = np.zeros(len(x)) 
colors = np.linspace(0.2,0.8, num=n)

f = 1 / (2*np.pi)

plt.figure(figsize=(5, 4))
plt.plot(x*180/np.pi, y, label='original', color= 'grey', linestyle = 'none', marker='.', alpha=1)  
for i in range(0,n):
    if i==0:
        a[i] = 2 * fft_values[i].real / len(x)
        fourier_approx = fourier_approx + a[i]/2
    else:    
        a[i] = 2 * fft_values[i].real / len(x)
        b[i] = - 2 * fft_values[i].imag / len(x)
        fourier_approx = fourier_approx + a[i] * np.cos(2*np.pi*i*f *x) + b[i] * np.sin(2*np.pi*i*f*x) 
    
    plt.plot(x*180/np.pi, fourier_approx, color=plt.cm.plasma(colors[i]), alpha=.70)
    
    





if Iorb(0,0,0,0,I0,w0,B)>Iorb(0,0,20,0,I0,w0,B):   #para que se fije si estamos graficando gauss o donut
    haz = 'Gauss'
else:
    haz = 'Donut' 

plt.title(f'{haz}, I0={I0}, w0={w0}nm, N={N}, A={A}nm, \n r={r}nm, phi={phi*180/np.pi:.2f}, n={n}')
plt.legend()
plt.xlim(-10,370)
plt.ylim(0,1.2)
plt.xlabel(r'$\theta$ ($^\circ$)')
plt.ylabel(r'I (a.u.)')
plt.tight_layout()
plt.show()

