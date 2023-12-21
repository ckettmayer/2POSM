# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 12:50:01 2023

@author: ckettmayer
"""

import addcopyfighandler
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.fft import fft



#I con haz gaussiano
def Iorb_gauss(A,theta,r,phi,I0,w0):
    return(I0*np.exp(-(2/w0**2)*(A**2+r**2-2*A*r*np.cos(theta-phi))))

#I con haz donut
def Iorb_donut(A,theta,r,phi,I0,w0):
    return(I0*2*np.e*(A**2+r**2-2*A*r*np.cos(theta-phi))/(w0**2)*np.exp(-(2/w0**2)*(A**2+r**2-2*A*r*np.cos(theta-phi))))

I0 = 1
w0 = 300
B = 0

A = 150
N = 1000 #Cantidad de puntos en la órbita
theta = np.linspace(0, 2*np.pi,N)

#POSICION DE LA PARTÍCULA
r = 300
phi = np.pi * 1/2



##ELEGIR EL HAZ QUE SE VA A GRAFICAR##
# psf = input(prompt="PSF gauss (g) o dount (d)?: ")

psf = 'd'

if psf == 'g':
    def Iorb(A,theta,r,phi,I0,w0):
        return Iorb_gauss(A,theta,r,phi,I0,w0)          #haz gaussiano con máximo central
if psf == 'd':
    def Iorb(A,theta,r,phi,I0,w0):
        return Iorb_donut(A,theta,r,phi,I0,w0)        #haz donut con mínimo central      

if Iorb(0,0,0,0,I0,w0)>Iorb(0,0,20,0,I0,w0):   #para que se fije si estamos graficando gauss o donut
    haz = 'Gauss'
else:
    haz = 'Donut' 
    
    

def fourier_coef(x,y): 
    fft_values = fft(y)
    a0 = 2 * fft_values[0].real / len(x)
    a1 = 2 * fft_values[1].real / len(x)
    b1 = - 2 * fft_values[1].imag / len(x)
    a2 = 2 * fft_values[2].real / len(x)
    b2 = - 2 * fft_values[2].imag / len(x)
    return (a0,a1,b1,a2,b2)


x = theta
y = Iorb(A,theta,r,phi,I0,w0)



a0, a1, b1, a2, b2 = fourier_coef(x,y)[0], fourier_coef(x,y)[1], fourier_coef(x,y)[2], fourier_coef(x,y)[3], fourier_coef(x,y)[4]
A0 = (a0/2)
A1 = (a1**2+b1**2)**(1/2)
A2 = (a2**2+b2**2)**(1/2)

F1 = np.mod(np.arctan2(b1, a1), 2 * np.pi)
F2 = np.mod(np.arctan2(b2, a2), 2 * np.pi)


t0 = A0
t1 = A1 * np.cos(x - F1)  
t2 = A2 * np.cos(2 * x - F2)


#%%

#Fourier

plt.figure(figsize=(4, 3))
plt.plot(x,y, label= 'original', color='k')
plt.plot(x,x*0+t0, label = 't0')
plt.plot(x,t0+t1, label = 't0+t1', color='b')
plt.plot(x,t0+t2, label = 't0+t2', color = 'r')
plt.plot(x,t0+t1+t2, label = 't0+t1+t2', color = 'g')
plt.legend()
plt.ylim(-0.1,1.3)
plt.title(f'r={r}nm')






#%%



phi = 1 * np.pi / 2  

# R variation
rplot = np.linspace(0.1,6*A,100)
a0_r = np.zeros(len(rplot))
a1_r = np.zeros(len(rplot))
b1_r = np.zeros(len(rplot))
a2_r = np.zeros(len(rplot))
b2_r = np.zeros(len(rplot))




for i in range(len(rplot)):
    x = theta
    y = Iorb(A,theta,rplot[i],phi,I0,w0) 
    a0_r[i], a1_r[i], b1_r[i], a2_r[i], b2_r[i] = fourier_coef(x,y)[0], fourier_coef(x,y)[1], fourier_coef(x,y)[2], fourier_coef(x,y)[3], fourier_coef(x,y)[4]

A0 = (a0_r/2)
A1 = (a1_r**2+b1_r**2)**(1/2)
A2 = (a2_r**2+b2_r**2)**(1/2)


F1_gauss = np.degrees(np.mod(np.arctan2(b1_r, a1_r), 2 * np.pi))
F1_donut = np.degrees(np.mod(np.arctan2(-b1_r, -a1_r), 2 * np.pi))
F2 = np.degrees(np.mod(np.arctan2(-b2_r, -a2_r), 2 * np.pi))


    
plt.figure(figsize=(4, 3))

plt.plot(rplot, A*A1/A0, label='A*A1/A0', color = 'k', marker='.')
plt.plot(rplot, 100*A2/A0, label='100 * A2/A0', color = 'r', marker='.')
plt.plot(rplot, 100*A2/A1, label='100 * A2/A1', color = 'm', marker='.')
plt.plot(rplot, 100*(A2+A1)/A0, label='100 * (A1+A2)/A0', color = 'orange', marker='.')

# plt.plot(rplot, F1_gauss, label='F1', color = 'grey', marker='s')

# plt.plot(rplot, F1_donut, label='F1', color = 'grey', marker='s')
# plt.plot(rplot, F2, label='F2', color = 'tomato', marker='s')

plt.plot(rplot,rplot, color='r', label = 'y=x')

# plt.fill_between(rplot, -0.2, 2, where=rplot < A, alpha=0.5, label = 'inside of orbit')

plt.legend()
plt.xlabel('r (nm)')
plt.ylabel('Fourier coefficient')
# plt.ylim(-10,150)
# plt.ylim(-0.2,2)
plt.title(f'{haz}, phi = {phi*180/np.pi}'+r'$^\circ$'+f', w0={w0}, A={A}')
plt.tight_layout()


#%%

r = 50 * 3

# phi variation
phiplot = np.linspace(0.1,2*np.pi,100)
a0_phi = np.zeros(len(rplot))
a1_phi = np.zeros(len(rplot))
b1_phi = np.zeros(len(rplot))
a2_phi = np.zeros(len(rplot))
b2_phi = np.zeros(len(rplot))

for i in range(len(rplot)):
    x = theta
    y = Iorb(A,theta,r,phiplot[i],I0,w0) 
    a0_phi[i], a1_phi[i], b1_phi[i], a2_phi[i], b2_phi[i] = fourier_coef(x,y)[0], fourier_coef(x,y)[1], fourier_coef(x,y)[2], fourier_coef(x,y)[3], fourier_coef(x,y)[4]
    
plt.figure(figsize=(4, 3))

A0 = (a0_phi/2)
A1 = (a1_phi**2+b1_phi**2)**(1/2)
A2 = (a2_phi**2+b2_phi**2)**(1/2)

F1_gauss = np.degrees(np.mod(np.arctan2(b1_phi, a1_phi), 2 * np.pi))
F1_donut = np.degrees(np.mod(np.arctan2(-b1_phi, -a1_phi), 2 * np.pi))
F2 = np.degrees(np.mod(np.arctan2(-b2_phi, -a2_phi), 2 * np.pi))


plt.plot(phiplot*180/np.pi, A*A1/A0, label='A*A1/A0', color = 'k', marker='o')

plt.plot(phiplot*180/np.pi, F1_gauss, label='F1', color = 'grey', marker='s')
# plt.plot(phiplot*180/np.pi, F1_donut, label='F1', color = 'grey', marker='s')
# plt.plot(phiplot*180/np.pi, F2, label='F2', color = 'tomato', marker='s')


plt.plot(phiplot*180/np.pi,phiplot*180/np.pi, color='r', label = 'y=x')

plt.legend()
plt.xlabel(r'phi ($^\circ$)')
plt.ylabel('Fourier coefficient')
# plt.ylim(-1,1)
plt.title(f'{haz}, r = {r} nm')
plt.tight_layout()

plt.show(block=False)




