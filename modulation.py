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
r = 50
phi = np.pi/2


##ELEGIR EL HAZ QUE SE VA A GRAFICAR##
def Iorb(A,theta,r,phi,I0,w0,B):
    return Iorb_gauss(A,theta,r,phi,I0,w0,B)          #haz gaussiano con máximo central
    # return Iorb_donut(A,theta,r,phi,I0,w0,B)        #haz donut con mínimo central

x = theta
y = Iorb(A,theta,r,phi,I0,w0,B)
fft_values = fft(y)

# Obtener los primeros dos términos de la serie de Fourier
a0 = np.abs(fft_values[0]) / len(x)
a1 = np.abs(fft_values[1]) / len(x)
b1 = np.angle(fft_values[1])

# Construir la aproximación de la serie de Fourier con los primeros dos términos
fourier_approximation = a0  + a1 * np.cos(x) + b1 * np.sin(x)

# Visualizar la función original y la aproximación de la serie de Fourier
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(x, y, label='Función original')
plt.title('Función original')

plt.subplot(2, 1, 2)
plt.plot(x, x*0+fourier_approximation, label='Aproximación de Fourier')
plt.title('Aproximación de la serie de Fourier (2 términos)')

plt.tight_layout()
plt.show()


    
plt.plot(theta, Iorb(A,theta,r,phi,I0,w0,B))    