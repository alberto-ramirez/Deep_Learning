##==============================================================================
## Gradient Descent
## Book: Aprendizaje Automatico y Profundo en Python, Capitulo 5
## Description: Regression models, Gradient Descent
## Author: Carlos M. Pineda Pertuz
##==============================================================================
## Programmer: Alberto Ramirez Bello
## Date: October 18th, 2023
##==============================================================================

import time
inicio = time.time()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

## Obtener las 10 primeras muestras
df    = pd.read_csv("precios_casas.csv")
X     = df[['sqft_living']].values[:10]
y     = df['price'].values[:10]
epoch = 100

## Estandarizar los valores de los vectores X, y
sc_x = StandardScaler()
sc_y = StandardScaler()
X = sc_x.fit_transform(X)
y = sc_y.fit_transform( y[:, np.newaxis] ).flatten()

## Funciones para la aplicacion del algoritmo
def calcular_prediccion(x, w):
    '''Esta funcion calcula un valor para el vector y, basado en los pesos w'''

    prediccion = np.dot(x, w[1:]) + w[0]
    return prediccion

def calcular_costo(X, y):
    '''Esta funcion calcula los valores de J'''

    m = len(X)
    error = 0.0
    prediccion = calcular_prediccion(X,w)
    error = (y - prediccion)
    costo = (error ** 2).sum() / m
    return costo

def entrenar(X, y, w, max_iter, tasa_aprendizaje):
    '''Esta funcion aplica el algoritmo de gradiente'''

    costos = []
    
    for i in range(max_iter):
        prediccion = calcular_prediccion(X,w)
        error = (y - prediccion)
        w[0] += 2*tasa_aprendizaje*error.sum()
        w[1:] += 2*tasa_aprendizaje*np.dot(X.T,error)
        costo = calcular_costo(X,y)
        costos.append(costo)
    return w, costos

w = [0.1, 0.5]
w, costos = entrenar(X, y, w, max_iter=epoch, tasa_aprendizaje=0.001)
print('El valor de w0 es : {:.3f}'.format( w[0] ))
print('El valor de w1 es : {:.3f}'.format( w[1] ))

fig,axs = plt.subplots(2, figsize=(15,4))
fig.suptitle('Gradiente Descendiente')

axs[0].scatter(X, y, c='blue', edgecolor='white', s=70)
axs[0].plot(X, calcular_prediccion(X,w), color='black', lw=2)
axs[0].set_title('sqtf_living vs price')

axs[1].plot(range(1, epoch + 1 ), costos)
axs[1].set_title('Funcion de costo (J) por iteracion')

fin = time.time()
ejecucion = (fin - inicio)
print(f'Tiempo total de ejecucion del programa antes de mostrar las graficas: {ejecucion}')
plt.show()
