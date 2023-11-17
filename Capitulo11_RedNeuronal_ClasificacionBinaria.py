##==============================================================================
## Introduction to Neural Networks
## Book: Aprendizaje Automatico y Profundo en Python, Capitulo 11
## Description: Neural Network Trainning
## Author: Carlos M. Pineda Pertuz
##==============================================================================
## Programmer: Alberto Ramirez Bello
## Date: October 26th, 2023
##==============================================================================

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn import datasets
from keras import optimizers
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

dataset = datasets.load_breast_cancer()
X = dataset.data
y = dataset.target
X = X[:, [1, 2, 3]] ## Se seleccionan 3 columnas del dataset: mean radius, mean texture, mean perimeter

# Estandarizacion de los datos
X[:,0] = ( X[:,0] - X[:,0].mean() ) / X[:, 0].std()
X[:,1] = ( X[:,1] - X[:,1].mean() ) / X[:, 1].std()
X[:,2] = ( X[:,2] - X[:,2].mean() ) / X[:, 2].std()

X_ent, X_pru, y_ent, y_pru = train_test_split(X, y, test_size=0.2, random_state=2)
m = X_ent.shape[0] ## Numero de muestras de entrenamiento
y_ent = y_ent[:, np.newaxis]
y_pru = y_pru[:, np.newaxis]

## Creacion del objeto de clase Sequential y las capas del modelo
modelo = Sequential()
modelo.add( Dense(10, activation='relu', input_shape=(X.shape[1],) ) )  # Capa de entrada del modelo, se activa con la funcion relu
modelo.add( Dense(5, activation='relu') ) # Capa escondida
modelo.add( Dense(1, activation='sigmoid') ) # Capa de salida

## Mostrar numero de parametros de la red neuronal
modelo.summary()

## Definicion del optimizador, en este caso gradiente de descenso estocastico SGD
sgd = optimizers.SGD(learning_rate=0.01)
modelo.compile( loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'] )  # Funcion de costo: binary cross entropy

## Graficando con plot_model de keras
keras.utils.plot_model(modelo, show_shapes=True)

## Entrenar la red
historial = modelo.fit( X_ent, y_ent, epochs=1000, batch_size=64, validation_split=0.2 )
perdida, exactitud = modelo.evaluate(X_pru, y_pru)
print('\nLa exactitud del modelo es: ', exactitud)

## grafico historico
pd.DataFrame(historial.history).plot(figsize=(7,7))
