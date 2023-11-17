##==============================================================================
## Non-Linear models
## Book: Aprendizaje Automatico y Profundo en Python, Capitulo 5
## Description: Section 5.8.2 - Telephone subscription
## Author: Carlos M. Pineda Pertuz
##==============================================================================
## Programmer: Alberto Ramirez Bello
## Date: October 20th, 2023
##==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df = pd.read_csv("datos_suscripciones.csv")
#print(df.sample(7))
pl.scatter( df['año'], df['cantidad'], color='r' )
pl.xlabel('Año')
pl.xlim(1960,2020)
pl.ylabel('Cantidad')
pl.show()

## Funcion sigmoide para visualizar junto a los datos, no esta ajustado
def sigmoide(x, beta1, beta2):
    y = 1/( 1 + np.exp(-beta1*(x-beta2)) )
    return y

beta_1 = 0.3
beta_2 = 2000.0
Y_pred = sigmoide( df['año'], beta_1, beta_2 )
pl.plot(df['año'], Y_pred*26800, 'g')
pl.plot(df['año'], df['cantidad'], 'ro')
pl.xlim(1960,2020)
pl.xlabel('Año')
pl.ylabel('Cantidad')
pl.show()

## Normalizando los datos
xdata = df['año']/max(df['año'])
ydata = df['cantidad']/max(df['cantidad'])
popt, pcov = curve_fit(sigmoide, xdata, ydata)
print('\nbeta1: {:.2f}\nbeta2: {:.2f}'.format(popt[0], popt[1]) )

x = np.linspace(1960,2020,50)
x = x/max(x)
pl.figure(figsize=(8,5))
y = sigmoide(x,*popt)
pl.plot(xdata, ydata, 'ro', label='datos')
pl.plot(x,y, linewidth=3.0, label='Ajuste')
pl.ylabel('Subscripciones')
pl.xlabel('Años')
pl.show()

X = xdata
y = ydata
x_train, x_test, y_train, y_test = train_test_split(xdata, ydata, test_size=0.2, random_state=1)
popt, pcov = curve_fit(sigmoide, x_train, y_train)

# Predicciones...
y_hat = sigmoide(x_test, *popt)

# Evaluacion
print( 'R2:{:.2f}\n'.format( r2_score(y_hat, y_test) ) )
