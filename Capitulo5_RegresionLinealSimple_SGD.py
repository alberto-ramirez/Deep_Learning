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
df    = pd.read_csv("C:\\Users\\alber\\Documents\\Progra\\Deep_Learning\\aprendizaje_automatico_y_profundo\\fuentes\\casas\\precios_casas.csv")
X     = df[['sqft_living']].values[:10]
y     = df['price'].values[:10]

sgd_reg.fit(X,y)
w0,w1 = sgd_reg.intercept_, sgd_reg.coef_

print('El valor de w0 es: {}'.format(w0))
print('El valor de w1 es: {}'.format(w1))

fin = time.time()
ejecucion = (fin - inicio)
print(f'Tiempo total de ejecucion del programa antes de mostrar las graficas: {ejecucion}')


## ======================================================= sin SDG
#Tiempo total de ejecucion del programa antes de mostrar las graficas: 2.2704732418060303, en la primera ejecucion
#El valor de w0 es: 0.000
#El valor de w1 es: 0.737
## ==================================================================================================================
#El valor de w0 es: [-2.54084659e+08]
#El valor de w1 es: [3.23536513e+11]
#Tiempo total de ejecucion del programa antes de mostrar las graficas: 1.5704107284545898 -> con 100 iteraciones

## ================================================================================================================== con mil iteraciones
## El valor de w0 es: [-2.54084659e+08]
## El valor de w1 es: [3.23536513e+11]
## Tiempo total de ejecucion del programa antes de mostrar las graficas: 1.7200253009796143