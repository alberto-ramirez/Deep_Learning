##==============================================================================
## Lineal Regression 
## Book: Aprendizaje Automatico y Profundo en Python, Capitulo 5
## Description: Lineal Regression, Random Sample Consensous (RAMSAC)
## Author: Carlos M. Pineda Pertuz
##==============================================================================
## Programmer: Alberto Ramirez Bello
## Date: October 19th, 2023
##==============================================================================

from cProfile import label
import time
inicio = time.time()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor

reg = LinearRegression()

## Obtener las 10 primeras muestras
df    = pd.read_csv("C:\\Users\\alber\\Documents\\Progra\\Deep_Learning\\aprendizaje_automatico_y_profundo\\fuentes\\casas\\precios_casas.csv")
X     = df[ ['sqft_living'] ].values[:10]
y     = df['price'].values[:10]


## Estandarizar los valores de los vectores X, y
sc_x = StandardScaler()
sc_y = StandardScaler()
X_esc = sc_x.fit_transform(X)
y_esc = sc_y.fit_transform( y[:, np.newaxis] ).flatten()

reg.fit(X_esc,y_esc)
## Ajustar el model con datos de entrenamiento
ransanc = RANSACRegressor(reg)
ransanc.fit(X_esc, y_esc)
print('El valor de w0 es: {:.3f}'.format(ransanc.estimator_.intercept_) )
print('El valor de w1 es: {}'.format(ransanc.estimator_.coef_) )

## Graficando los modelos generados
inlier_mask  = ransanc.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
line_X = np.arange( X_esc.min(), X_esc.max() )[:,np.newaxis]
line_y = reg.predict(line_X)
line_y_ransac = ransanc.predict(line_X)
lw = 2

plt.scatter(X_esc[inlier_mask], y_esc[inlier_mask], color='yellowgreen', marker='.', label='Inliers')
plt.scatter(X_esc[outlier_mask], y_esc[outlier_mask], color='gold', marker='.', label='Outliers')
plt.plot(line_X, line_y, color='navy', linewidth=lw, label='Regresion Lineal')
plt.plot(line_X, line_y_ransac, color='cornflowerblue', linewidth=lw, label='Regresor RANSAC')
plt.legend(loc='upper left')
plt.xlabel('Entrada')
plt.ylabel('Respuesta')

fin = time.time()
ejecucion = (fin - inicio)
print(f'Tiempo total de ejecucion del programa antes de mostrar las graficas: {ejecucion}')
plt.show()

#El valor de w0 es: -0.422
#El valor de w1 es: [0.15225054]
#Tiempo total de ejecucion del programa antes de mostrar las graficas: 2.020704507827759
