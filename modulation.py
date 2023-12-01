# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 12:50:01 2023

@author: Constanza
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
r = 100
phi = np.pi/2


##ELEGIR EL HAZ QUE SE VA A GRAFICAR##
def Iorb(A,theta,r,phi,I0,w0,B):
    # return Iorb_gauss(A,theta,r,phi,I0,w0,B)          #haz gaussiano con máximo central
    return Iorb_donut(A,theta,r,phi,I0,w0,B)        #haz donut con mínimo central

x = theta
y = Iorb(A,theta,r,phi,I0,w0,B)

# ej de funcion facil
# x = np.linspace(0,2*np.pi,100)
# y = 3 + 1 * np.cos(x) + 3 * np.sin(x) + 5 * np.cos(2*x)


fft_values = fft(y)


a0 = fft_values[0].real / len(x)

a1 = 2 * fft_values[1].real / len(x)
b1 = - 2 * fft_values[1].imag / len(x)


f = 1 / (2*np.pi)
fourier_0 = a0
fourier_1 = a0 + a1 * np.cos(2*np.pi*f *x) + b1 * np.sin(2*np.pi*f*x) 

plt.figure(figsize=(5, 4))
plt.plot(x*180/np.pi, y, label='original', color= 'grey', linestyle = 'none', marker='.', alpha=1)  
plt.plot(x*180/np.pi, x*0+fourier_0, color='r', linestyle='dashed', alpha=.70, label = 'fourier 0 order')
plt.plot(x*180/np.pi, fourier_1, color='r', alpha=.70, label = 'fourier 1st order')
plt.xlabel(r'$\theta$ (rad)')
plt.ylabel(r'I (a.u.)')
plt.legend()
plt.tight_layout()
plt.show()




if Iorb(0,0,0,0,I0,w0,B)>Iorb(0,0,20,0,I0,w0,B):   #para que se fije si estamos graficando gauss o donut
    haz = 'Gauss'
else:
    haz = 'Donut' 

plt.title(f'{haz}, I0={I0}, w0={w0}nm, N={N}, A={A}nm, \n r={r}nm, phi={phi*180/np.pi}')

plt.xlim(-10,370)
plt.xlabel(r'$\theta$ ($^\circ$)')
plt.ylabel(r'I (a.u.)')
plt.tight_layout()
plt.show()







#PENDIENTE
#Ver como varían los primeros coeficientes en función de r y phi para cada haz (y en función de la relación entre A y w0)
    
