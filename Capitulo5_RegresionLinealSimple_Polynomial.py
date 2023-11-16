##==============================================================================
## Lineal Regression 
## Book: Aprendizaje Automatico y Profundo en Python, Capitulo 5
## Description: Lineal Regression, Polynomial regression
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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

reg = LinearRegression()

## Obtener las 10 primeras muestras
df    = pd.read_csv("C:\\Users\\alber\\Documents\\Progra\\Deep_Learning\\aprendizaje_automatico_y_profundo\\fuentes\\casas\\precios_casas.csv")
X     = df[ ['sqft_living'] ].values
y     = df['price'].values

## Estandarizar los valores de los vectores X, y
sc_x = StandardScaler()
sc_y = StandardScaler()
X_esc = sc_x.fit_transform(X)
y_esc = sc_y.fit_transform( y[:, np.newaxis] ).flatten()

X_ent, X_pru, y_ent, y_pru = train_test_split(X_esc, y_esc, test_size=0.3, random_state=50)
poly = PolynomialFeatures(degree=2)
X_ent_poly = poly.fit_transform(X_ent)
X_pru_poly = poly.fit_transform(X_pru)
#print('Mostrando el contenido del arreglo X_ent_poly:\n', X_ent_poly)

## Ajustar el model con datos de entrenamiento
reg.fit(X_ent_poly, y_ent)
print('Coeficientes: ', reg.coef_)
print('Interceptcion: ', reg.intercept_)
print('Precision del modelo: {}'.format(reg.score(X_ent_poly, y_ent)) )

x_fit = np.arange( X_ent.min(), X_ent.max(), 1 )[:, np.newaxis]
y_p = reg.predict(poly.fit_transform(x_fit) )
plt.scatter(X_ent, y_ent, label='Puntos de entrenamiento')
plt.plot(x_fit, y_p)
plt.legend(loc='upper left')

fin = time.time()
ejecucion = (fin - inicio)
print(f'Tiempo total de ejecucion del programa antes de mostrar las graficas: {ejecucion}')

plt.show()

#Coeficientes:  [0.         0.3447626  0.07234282]
#Interceptcion:  -0.05851055222583371
#Precision del modelo: 0.1736017131986194