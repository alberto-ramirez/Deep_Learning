##==============================================================================
## Lineal Regression 
## Book: Aprendizaje Automatico y Profundo en Python, Capitulo 5
## Description: Multiple Lineal Regression, Matriz notation
## Author: Carlos M. Pineda Pertuz
##==============================================================================
## Programmer: Alberto Ramirez Bello
## Date: October 20th, 2023
##==============================================================================

from cProfile import label
import time
inicio = time.time()

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

## Obtener las 10 primeras muestras
df = pd.read_csv("precios_casas.csv")
X  = df[ ['sqft_living'] ].values
y  = df['price'].values
w  = np.zeros(X.shape[1])

## Estandarizar los valores de los vectores X, y
sc_x = StandardScaler()
sc_y = StandardScaler()
X_esc = sc_x.fit_transform(X)
y_esc = sc_y.fit_transform( y[:, np.newaxis] ).flatten()

## Agregando columna de unos
X2 = np.hstack( (np.ones( (X_esc.shape[0], 1)), X_esc) )
t1 = np.linalg.inv(np.dot(X2.T,X2))
t2 = np.dot(X2.T, y)
w  = np.dot(t1, t2)
print('El valor de w0 es: {:.3f}'.format( w[0] ) )
print('El valor de w1 es: {:.3f}'.format( w[1] ) )

fin = time.time()
ejecucion = (fin - inicio)
print(f'Tiempo total de ejecucion del programa antes de mostrar las graficas: {ejecucion}')
