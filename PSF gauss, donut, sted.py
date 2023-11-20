# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 14:29:06 2023

@author: ckettmayer
"""

#Intensidad con haz gaussiano, donut y sted

import numpy as np
import matplotlib.pyplot as plt



def I_gauss(r,A0,FWHM):
    return (A0*np.e**(-4*np.log(2)*(r**2/FWHM**2)))


def I_donut(r,A0,FWHM):
    return (A0*4*np.e*np.log(2)*(r**2/FWHM**2)*np.e**(-4*np.log(2)*(r**2/FWHM**2)))



# plt.close('all')

rplot = np.linspace(-600,600,500)

A = 1

F = 300

plt.plot(rplot,I_gauss(rplot,A,F), label='gauss') 
plt.plot(rplot,I_donut(rplot,A,F), label='donut') 
plt.plot(rplot, I_gauss(rplot,A,F)- I_donut(rplot,2*A,F), label= 'STED (gauss-2*donut)')
plt.legend()
plt.xlabel('r(nm)')
plt.ylabel('I(a.u.)')
plt.ylim(0,A+0.5)
plt.title('A='+str(A)+', FWHM='+str(F))