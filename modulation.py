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
phi = np.pi * 3/2



##ELEGIR EL HAZ QUE SE VA A GRAFICAR##
# psf = input(prompt="PSF gauss (g) o dount (d)?: ")

psf = 'g'

if psf == 'g':
    def Iorb(A,theta,r,phi,I0,w0,B):
        return Iorb_gauss(A,theta,r,phi,I0,w0,B)          #haz gaussiano con máximo central
if psf == 'd':
    def Iorb(A,theta,r,phi,I0,w0,B):
        return Iorb_donut(A,theta,r,phi,I0,w0,B)        #haz donut con mínimo central      

if Iorb(0,0,0,0,I0,w0,B)>Iorb(0,0,20,0,I0,w0,B):   #para que se fije si estamos graficando gauss o donut
    haz = 'Gauss'
else:
    haz = 'Donut' 
    
    

def fourier_coef(x,y): 
    fft_values = fft(y)
    a0 = 2 * fft_values[0].real / len(x)
    a1 = 2 * fft_values[1].real / len(x)
    b1 = - 2 * fft_values[1].imag / len(x)
    return (a0,a1,b1)


x = theta
y = Iorb(A,theta,r,phi,I0,w0,B)


#%%

phi = np.pi * (1/2)  

# R variation
rplot = np.linspace(0.1,A,10)
a0_r = np.zeros(len(rplot))
a1_r = np.zeros(len(rplot))
b1_r = np.zeros(len(rplot))

for i in range(len(rplot)):
    x = theta
    y = Iorb(A,theta,rplot[i],phi,I0,w0,B) 
    a0_r[i], a1_r[i], b1_r[i] = fourier_coef(x,y)[0], fourier_coef(x,y)[1], fourier_coef(x,y)[2]


A1 = (a1_r**2+b1_r**2)**(1/2)
A0 = (a0_r/2)

F1 = np.degrees(np.mod(np.arctan2(b1_r, a1_r), 2 * np.pi))


    
plt.figure(figsize=(5, 4))
# plt.plot(rplot, a0_r, label='a0', color = 'g')
# plt.plot(rplot, a1_r, label='a1', color = 'r')
# plt.plot(rplot, a1_r/a0_r, label='a1/a0', color = 'orange')
# plt.plot(rplot, b1_r, label='b1', color = 'b')
# plt.plot(rplot, b1_r/a0_r, label='b1/a0', color = 'dodgerblue')
# plt.plot(rplot, (a1_r+b1_r)/2, label='(a1+b1)/2', color = 'm')
plt.plot(rplot, 100*A1/A0, label='100 * A1/A0', color = 'k', marker='o')
plt.plot(rplot, F1, label='F1', color = 'grey', marker='s')

print('100*A1/A0_r')
print(100*A1/A0)
print('F1_r')
print(F1)

# plt.plot(rplot, 160*(a1_r**2+b1_r**2)**(1/2)/(a0_r/2), label='A1/A0', color = 'k')
# plt.plot(rplot, np.degrees(np.mod(np.arctan2(b1_r, a1_r), 2 * np.pi)) , label='theta1', color = 'grey')

# plt.plot(rplot, rplot*0+phi*180/np.pi, color= 'royalblue', label = 'y=phi')


plt.legend()
plt.xlabel('r (nm)')
plt.ylabel('Fourier coefficient')
# plt.ylim(-1,1)
# plt.ylim(-0.2,1.8)
plt.title(f'{haz}, phi = {phi*180/np.pi}'+r'$^\circ$')
plt.tight_layout()


#%%

r = 50 * 2

# phi variation
phiplot = np.linspace(0.1,2*np.pi,10)
a0_phi = np.zeros(len(rplot))
a1_phi = np.zeros(len(rplot))
b1_phi = np.zeros(len(rplot))

for i in range(len(rplot)):
    x = theta
    y = Iorb(A,theta,r,phiplot[i],I0,w0,B) 
    a0_phi[i], a1_phi[i], b1_phi[i] = fourier_coef(x,y)[0], fourier_coef(x,y)[1], fourier_coef(x,y)[2]
    
plt.figure(figsize=(5, 4))
# plt.plot(phiplot*180/np.pi, a0_phi, label='a0', color = 'g')
# plt.plot(phiplot*180/np.pi, a1_phi, label='a1', color = 'r')
# plt.plot(phiplot*180/np.pi, a1_phi/a0_phi, label='a1/a0', color = 'orange')
# plt.plot(phiplot*180/np.pi, b1_phi, label='b1', color = 'b')
# plt.plot(phiplot*180/np.pi, b1_phi/a0_phi, label='b1/a0', color = 'dodgerblue')
# plt.plot(phiplot*180/np.pi, (a1_phi+b1_phi)/2, label='(a1+b1)/2', color = 'm')


A1 = (a1_phi**2+b1_phi**2)**(1/2)
A0 = (a0_phi/2)

F1 = np.degrees(np.mod(np.arctan2(b1_phi, a1_phi), 2 * np.pi))


plt.plot(phiplot*180/np.pi, 100*A1/A0, label='100*A1/A0', color = 'k', marker='o')
plt.plot(phiplot*180/np.pi, F1, label='F1', color = 'grey', marker='s')

# plt.plot(phiplot*180/np.pi, phiplot*0+r , label = 'y=r', color= 'tomato', zorder = 4)
# plt.plot(phiplot*180/np.pi, phiplot*180/np.pi , label = 'y=x', color= 'royalblue', zorder = 4)


print('100*A1/A0_phi')
print(100*A1/A0)
print('F1_phi')
print(F1)


plt.legend()
plt.xlabel(r'phi ($^\circ$)')
plt.ylabel('Fourier coefficient')
# plt.ylim(-1,1)
plt.title(f'{haz}, r = {r} nm')
plt.tight_layout()

plt.show(block=False)






#PENDIENTE
#Ver como varían los primeros coeficientes en función de r y phi para cada haz (y en función de la relación entre A y w0)
    
