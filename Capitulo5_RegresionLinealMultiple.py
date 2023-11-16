##==============================================================================
## Lineal Regression 
## Book: Aprendizaje Automatico y Profundo en Python, Capitulo 5
## Description: Lineal Regression, Multiple regression
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
from sklearn.model_selection import train_test_split

reg = LinearRegression()

## Obtener las 10 primeras muestras
df    = pd.read_csv("C:\\Users\\alber\\Documents\\Progra\\Deep_Learning\\aprendizaje_automatico_y_profundo\\fuentes\\casas\\precios_casas.csv")
X     = df[ ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot'] ].values
y     = df['price'].values

## Estandarizar los valores de los vectores X, y
sc_x = StandardScaler()
sc_y = StandardScaler()
X = sc_x.fit_transform(X)
y = sc_y.fit_transform( y[:, np.newaxis] ).flatten()

X_ent, X_pru, y_ent, y_pru = train_test_split(X, y, test_size=0.3, random_state=1)

## Ajustar el model con datos de entrenamiento
reg.fit(X_ent, y_ent)
print('Intercepcion: {:.3f}'.format(reg.intercept_) )
print('Coeficientes: ', reg.coef_ )

fin = time.time()
ejecucion = (fin - inicio)
print(f'Tiempo total de ejecucion del programa antes de mostrar las graficas: {ejecucion}')

## Calculando exactitud del modelo
print('Exactitud: {:.3f}'.format(reg.score(X_pru, y_pru)))

#Intercepcion: 0.009
#Coeficientes:  [-0.09001078  0.00709681  0.49100858 -0.05325143]
#Tiempo total de ejecucion del programa antes de mostrar las graficas: 1.9104938507080078
#Exactitud: 0.428
