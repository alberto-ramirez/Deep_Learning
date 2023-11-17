##==============================================================================
## Lineal Regression 
## Book: Aprendizaje Automatico y Profundo en Python, Capitulo 5
## Description: Lineal Regression, simple regression
## Author: Carlos M. Pineda Pertuz
##==============================================================================
## Programmer: Alberto Ramirez Bello
## Date: October 19th, 2023
##==============================================================================

import time
inicio = time.time()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

reg = LinearRegression()

## Obtener las 10 primeras muestras
df    = pd.read_csv("precios_casas.csv")
X     = df[['sqft_living']].values[:10]
y     = df['price'].values[:10]
epoch = 100

## Estandarizar los valores de los vectores X, y
sc_x = StandardScaler()
sc_y = StandardScaler()
X_esc = sc_x.fit_transform(X)
y_esc = sc_y.fit_transform( y[:, np.newaxis] ).flatten()

reg.fit(X_esc, y_esc)
print('El valor de w0 es: {:.3f}'.format(reg.intercept_))
print('El valor de w1 es: {:.3f}'.format(reg.coef_[0]))

fin = time.time()
ejecucion = (fin - inicio)
print(f'Tiempo total de ejecucion del programa antes de mostrar las graficas: {ejecucion}')

sqtf_living = sc_x.transform(np.array([[10000]]) )
precio = reg.predict(sqtf_living)
print('Precio predicho: %.3f' % precio)
