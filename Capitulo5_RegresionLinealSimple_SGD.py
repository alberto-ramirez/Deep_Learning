##==============================================================================
## Lineal Regression 
## Book: Aprendizaje Automatico y Profundo en Python, Capitulo 5
## Description: Lineal Regression, simple regression with SGD Regressor
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
from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, eta0=0.001, random_state=1)

## Obtener las 10 primeras muestras
df    = pd.read_csv("precios_casas.csv")
X     = df[['sqft_living']].values[:10]
y     = df['price'].values[:10]

sgd_reg.fit(X,y)
w0,w1 = sgd_reg.intercept_, sgd_reg.coef_

print('El valor de w0 es: {}'.format(w0))
print('El valor de w1 es: {}'.format(w1))

fin = time.time()
ejecucion = (fin - inicio)
print(f'Tiempo total de ejecucion del programa antes de mostrar las graficas: {ejecucion}')
