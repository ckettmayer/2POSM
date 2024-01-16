# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 11:40:22 2023

@author: ckettmayer
"""

# import addcopyfighandler
from matplotlib.widgets import Slider
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
from scipy.fft import fft

#I con haz gaussiano
def Iorb_gauss(A,theta,r,phi,I0,w0):
    return(I0*np.exp(-(2/w0**2)*(A**2+r**2-2*A*r*np.cos(theta-phi))))

#I con haz donut
def Iorb_donut(A,theta,r,phi,I0,w0):
    return(I0*2*np.e*(A**2+r**2-2*A*r*np.cos(theta-phi))/(w0**2)*np.exp(-(2/w0**2)*(A**2+r**2-2*A*r*np.cos(theta-phi))))


I0 = 1
w0 = 300


N = 100 #points in orbit

A = 150 #orbit radius
theta = np.linspace(0, 2*np.pi, N)

r = 100
phi = np.pi/2   #Theta and phi variables are stored in radians, but are plotted in degrees

l = 400         #max particle distance from origin


#%%

#psf plot
fig = plt.figure(figsize=(5, 3))
rplot = np.linspace(-2*w0,2*w0,100)
plt.plot(rplot, Iorb_gauss(rplot,0,0,0,I0,w0), label='gauss')
plt.plot(rplot, Iorb_donut(rplot,0,0,0,I0,w0), label='donut')
plt.title(f'I0={I0}, w0={w0}nm, (A,theta)=(0,0), (r,phi)=(0,0)')
plt.ylabel('I (a.u.)')
plt.xlabel('x (nm)')
plt.grid()
plt.legend()
plt.tight_layout()
    



#%%

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
    haz = 'Haz gaussiano (max central)'
else:
    haz = 'Haz donut (min central)' 
    
    


colors = Iorb(A,theta,r,phi,I0,w0)

cm = 'viridis'
norm = mpl.colors.Normalize(vmin=0, vmax=I0)  #modifico la normalización de los colores para que vaya entre dos valores fijos
sm = mpl.cm.ScalarMappable(cmap=cm, norm=norm)
sm.set_array([])



fig = plt.figure(figsize=(6, 8))

gs = fig.add_gridspec(3, 1, height_ratios=[2,0.7,0.4], hspace=0.3, wspace=10)

ax = fig.add_subplot(gs[0], projection='polar')
# plt.subplots_adjust(top=1, hspace=10)

#Sliders for r, phi and A
ax_parameter_r = plt.axes([0.17, 0.15, 0.65, 0.03], facecolor='lightgoldenrodyellow')   #[left, bottom, width, height]
slider_r = Slider(ax_parameter_r, 'r', 0, l, valinit=r)

ax_parameter_phi = plt.axes([0.17, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider_phi = Slider(ax_parameter_phi, 'phi', 0, 360, valinit= 90)    #el slider de phi está en grados, pero al evaluar el valor hay que pasar a radianes

ax_parameter_A = plt.axes([0.17, 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider_A = Slider(ax_parameter_A, 'A', 50, l, valinit=A)    #el slider de phi está en grados, pero al evaluar el valor hay que pasar a radianes



#Create a polar mesh for PSF plot
xplot = np.linspace(-1.5*l,1.5*l,100)
yplot = np.linspace(-1.5*l,1.5*l, 150)
X, Y = np.meshgrid(xplot, yplot)
#Convert to polar coordinates
R, Phi = np.sqrt(X**2 + Y**2), np.arctan2(Y, X)





#Custom colormap white-red for PSF
color_max = 'red'  
color_min = 'white'
cmap_segments = {
    'red': [(0.0, 1.0, 1.0), (1.0, 1.0, 1.0)],  # Rojo a Blanco en el canal red
    'green': [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)],  # Sin cambio en el canal green
    'blue': [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]  # Sin cambio en el canal blue
}
custom_cmap = mcolors.LinearSegmentedColormap('custom_colormap', cmap_segments, 256)

#PLOT 
ax.scatter(Phi, R, c=Iorb(R, Phi, A, phi, I0, w0), marker='o', s=5, cmap = custom_cmap.reversed(), zorder=1)   #PSF   
ax.scatter(theta, theta*0+A, c=colors, marker='o', s=100, cmap=cm, norm=norm, alpha=1)                            #orbit
ax.scatter(phi, r, color='y', marker='*', s=250, edgecolors='k', zorder=3)                                        #particle

ax.set_ylim(0, l+50)





       
fig.suptitle(f'{haz} \n I0={I0}, w0={w0}nm, N={N}, A={A}nm')



#Intensidad

ax2 = fig.add_subplot(gs[1])
ax2.grid()

ax2.scatter(theta*180/np.pi, Iorb(A,theta,r,phi,I0,w0), c=colors, marker='o', s=100, cmap=cm, norm=norm)

# ax2.scatter(phi,0, color='y', marker='*', s=250, edgecolors='k', zorder=3)
ax2.set_xlabel(r'$\theta$ ($^{\circ}$)')
ax2.set_ylabel('I (a.u.)')
ax2.set_ylim(-0.1*I0, I0+0.1*I0)


def fourier(x,y): 
    fft_values = fft(y)
    a0 = 2 * fft_values[0].real / len(x)
    a1 = 2 * fft_values[1].real / len(x)
    b1 = - 2 * fft_values[1].imag / len(x)
    a2 = 2 * fft_values[2].real / len(x)
    b2 = - 2 * fft_values[2].imag / len(x)
    
    A0 = (a0/2)
    A1 = (a1**2+b1**2)**(1/2)
    A2 = (a2**2+b2**2)**(1/2)
    F1 = np.mod(np.arctan2(b1, a1), 2 * np.pi)
    F2 = np.mod(np.arctan2(b2, a2), 2 * np.pi)
    
    t0 = A0
    t1 = A1 * np.cos(x*np.pi/180 - F1)  
    t2 = A2 * np.cos(2 * x*np.pi/180 - F2)
    return (t0,t1,t2)


x = theta*180/np.pi
y = Iorb(A,theta,r,phi,I0,w0)
t0,t1,t2 = fourier(x,y)[0], fourier(x,y)[1], fourier(x,y)[2]
ax2.plot(x,x*0+t0, label = 't0', color='k')
ax2.plot(x,t0+t1, label = 't0+t1', color='b')
ax2.plot(x,t0+t2, label = 't0+t2', color = 'r')


# Función llamada al cambiar el valor del slider
def update(val):
    ax.clear()
    ax2.clear()
    r = slider_r.val
    phi = slider_phi.val
    A = slider_A.val
    colors = Iorb(A,theta,r,phi*np.pi/180,I0,w0)
    ax.scatter(Phi, R, c=Iorb(R, Phi, A, phi*np.pi/180, I0, w0), marker='o', s=10, cmap = custom_cmap.reversed(), zorder=1)    #PSF
    ax.scatter(theta, theta*0+A, c=colors, marker='o', s=100, cmap=cm, norm=norm, alpha=1)   #orbit
    ax.scatter(phi*np.pi/180, r, color='y', marker='*', s=250, edgecolors='k', zorder=3)  
    ax2.scatter(theta*180/np.pi, colors, c=colors, marker='o', s=100, cmap=cm, norm=norm)
    
    x = theta*180/np.pi
    y = Iorb(A,theta,r,phi*np.pi/180,I0,w0)
    t0,t1,t2 = fourier(x,y)[0], fourier(x,y)[1], fourier(x,y)[2]
    ax2.plot(x,x*0+t0, label = 't0', color='k')
    ax2.plot(x,t0+t1, label = 't0+t1', color='b')
    ax2.plot(x,t0+t2, label = 't0+t2', color = 'r')
    
    ax2.grid()
    #particle
    ax.set_ylim(0, l+50)
    ax2.set_ylim(-0.1*I0, I0+0.1*I0)
    fig.canvas.draw_idle()
    
       
cbar = fig.colorbar(sm, ax=ax, pad=0.1, shrink=0.5)
cbar.set_label('Intensidad')    

# Conectar el slider con la función de actualización
slider_r.on_changed(update)
slider_phi.on_changed(update)
slider_A.on_changed(update)

