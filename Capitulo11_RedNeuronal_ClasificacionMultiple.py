##==============================================================================
## Introduction to Neural Networks
## Book: Aprendizaje Automatico y Profundo en Python, Capitulo 11
## Description: Neural Network for multiple classification
## Author: Carlos M. Pineda Pertuz
##==============================================================================
## Programmer: Alberto Ramirez Bello
## Date: October 28th, 2023
##==============================================================================

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn import datasets
from keras import optimizers
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

(X_ent, y_ent),(X_pru,y_pru) = fashion_mnist.load_data()

## Mostrando contenido de los grupos
print('\nDimensiones del arreglo X de entrenamiento: ',str(X_ent.shape))
print('Dimensiones del arreglo y de entrenamiento: ',str(y_ent.shape))
print('Dimensiones del arreglo X de prueba: ',str(X_pru.shape))
print('Dimensiones del arreglo y de prueba: ',str(y_pru.shape),'\n')

## Mostrar muestras del dataset
columnas = ['Blusa', 'Pantalon', 'Chaqueta', 'Vestido', 'Saco', 'Sandalia', 'Sueter', 'Zapatilla', 'Bolso', 'Bota']
#plt.figure( figsize=(12,8) )

#for i in range(15):
#    plt.subplot(3,5,i+1)
#    plt.imshow(X_ent[i])
#    plt.title( columnas[ y_ent[i] ] )
#    plt.axis('off')

#plt.show()

## Preprocesado
X_ent = X_ent.astype('float32')
X_pru = X_pru.astype('float32')
X_ent = X_ent/255
X_pru = X_pru/255
y_ent = to_categorical(y_ent, num_classes=10)
y_pru = to_categorical(y_pru, num_classes=10)

print('Contenido de y_ent[0]: ', y_ent[0])
print('Contenido de y_ent[1]: ', y_ent[1])

## Crear modelo con dos capas ocultas, 10 neuronas cada capa
modelo_multiple = keras.Sequential()
modelo_multiple.add(keras.layers.Flatten(input_shape=(28,28)))
modelo_multiple.add(keras.layers.Dense(10, activation='sigmoid'))
modelo_multiple.add(keras.layers.Dense(10, activation='softmax'))
print(modelo_multiple.summary())

## Configuracion del modelo
modelo_multiple.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
modelo_multiple.fit(X_ent, y_ent, epochs=10, verbose=1)

## Evaluando
loss, exac = modelo_multiple.evaluate(X_pru, y_pru)
print('La exactitud es: {:.2f}'.format(exac) )
plt.figure(figsize=(15,5))
x = 0

for i in [1, 5, 10, 15, 20, 25, 30, 35, 40, 45]:
    plt.subplot(2, 5, x+1)
    img = X_pru[i]
    plt.imshow(img)
    img = ( np.expand_dims(img,0) )
    res = modelo_multiple.predict(img)
    plt.title(columnas[np.argmax(y_pru[i])] + '-Pred: '+ columnas[np.argmax(res) ])
    plt.axis('off')
    x = x+1

plt.show()
