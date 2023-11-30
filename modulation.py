# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 12:50:01 2023

@author: Constanza
"""

# import addcopyfighandler
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
r = 140
phi = np.pi


##ELEGIR EL HAZ QUE SE VA A GRAFICAR##
def Iorb(A,theta,r,phi,I0,w0,B):
    # return Iorb_gauss(A,theta,r,phi,I0,w0,B)          #haz gaussiano con máximo central
    return Iorb_donut(A,theta,r,phi,I0,w0,B)        #haz donut con mínimo central

x = theta
y = Iorb(A,theta,r,phi,I0,w0,B)

# ej de funcion facil
# x = np.linspace(0,1,100)
# y = 3 + 1 * np.cos(x) + 3 * np.sin(x) + 5 * np.cos(2*x)


fft_values = fft(y)

n = 10
a = np.zeros(n)
b = np.zeros(n)

# a[0] = fft_values[0].real / len(x)

fourier_approx = np.zeros(len(x)) 
colors = np.linspace(start=50, stop=200, num=n)

f = 1 / (2*np.pi)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='original', color= 'grey', linestyle = 'none', marker='.', alpha=1)  
for i in range(0,n):
    if i==0:
        a[i] = fft_values[i].real / len(x)
        fourier_approx = fourier_approx + a[i]
    else:    
        a[i] = 2 * fft_values[i].real / len(x)
        b[i] = - 2 * fft_values[i].imag / len(x)
        fourier_approx = fourier_approx + a[i] * np.cos(2*np.pi*i*f *x) + b[i] * np.sin(2*np.pi*i*f*x) 
    
    plt.plot(x, fourier_approx, color=plt.cm.plasma(int(colors[i])), alpha=.70)
    
    
plt.legend()


plt.tight_layout()
plt.show()


#PENDIENTE
#Ver como varían los primeros coeficientes en función de r y phi para cada haz (y en función de la relación entre A y w0)
    
