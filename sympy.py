# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 14:41:45 2023

@author: ckettmayer
"""
import numpy as np
from sympy import symbols, integrate, exp, sin, cos, pi, latex, init_printing
from IPython.display import display


# Inicializar el sistema de impresión para obtener una salida más bonita
init_printing(use_latex=True)

# Variables simbólicas
theta, I0, w0, A, r, phi = symbols('theta I0 w0 A r phi')

# Función
f = I0 * exp(-(2/w0**2)*(A**2 + r**2 - 2*A*r*cos(theta-phi)))

# Calcular la integral indefinida
integral_result = integrate(f, theta)

# Imprimir el resultado de la integral indefinida directamente en Spyder
print(integral_result)
display(f)
display(integral_result)
