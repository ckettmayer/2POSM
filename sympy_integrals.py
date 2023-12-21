# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 14:41:45 2023

@author: ckettmayer
"""

from sympy import symbols, integrate, exp, sin, cos, pi, latex, init_printing, simplify
from IPython.display import display


# Inicializar el sistema de impresión para obtener una salida más bonita
init_printing(use_latex=True)

# Variables simbólicas
theta, I0, w0, A, r, phi = symbols('theta I0 w0 A r phi')

# Función
f = I0 * exp(-(2/w0**2)*(A**2 + r**2 - 2*A*r*cos(theta-phi))) * cos(theta)

# Calcular la integral indefinida
# integral_result = integrate(f, theta)

definite_integral_result = integrate(f, (theta, 0, 2*pi))


# Imprimir el resultado de la integral indefinida directamente en Spyder
# print(integral_result)

simple = simplify(definite_integral_result)
display(simple)
print(latex(simple))

