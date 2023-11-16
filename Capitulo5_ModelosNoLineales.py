##==============================================================================
## Non-Linear models
## Book: Aprendizaje Automatico y Profundo en Python, Capitulo 5
## Description: Section 5.8 - non linear models
## Author: Carlos M. Pineda Pertuz
##==============================================================================
## Programmer: Alberto Ramirez Bello
## Date: October 20th, 2023
##==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as pl

## Funcion cubica (y=x**3)
x = np.arange(-5.0, 5.0, 0.1)
y = 1*(x**3) + 1*(x**2) + 1*x + 1
y_noise = 20 * np.random.normal(size=x.size)
ydata = y + y_noise

pl.plot(x, ydata, 'bo')
pl.plot(x, y, 'r')
pl.xlabel('Variable independiente')
pl.ylabel('Variable dependiente')
pl.show()

## Funcion cuadratica y=x**2
x2 = np.arange(-5.0, 5.0, 0.1)
y2 = np.power(x2,2)
y2_noise = 2*np.random.normal(size=x.size)
y2data = y2 + y2_noise

pl.plot(x2, y2data, 'bo')
pl.plot(x2, y2, 'r')
pl.ylabel('Variable dependiente')
pl.xlabel('Variable independiente')
pl.show()

## Funcion exponencial (y = a + bc**x)
X = np.arange(-5.0, 5.0, 0.1)
Y = np.exp(X)

pl.plot(X,Y)
pl.ylabel('Variable dependiente')
pl.xlabel('Variable independiente')
pl.show()

## Funcion logaritmica ( y=log(x) )
Y2 = np.log(X)
pl.plot(X,Y2)
pl.ylabel('Variable dependiente')
pl.xlabel('Variable independiente')
pl.show()

## Funcion logistica o sigmoide y= alpha + (b/(1 + c**(x-d) ) )
Y3=1.0/(1.0+ np.exp(-X) )
pl.plot(X,Y3)
pl.ylabel('Variable dependiente')
pl.xlabel('Variable independiente')
pl.show()


#carolina rangel 
#55-59-66-84-94