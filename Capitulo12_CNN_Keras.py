##==============================================================================
## Introduction to Neural Networks
## Book: Aprendizaje Automatico y Profundo en Python, Capitulo 12
## Description: Convolutional Neural Network
## Author: Carlos M. Pineda Pertuz
##==============================================================================
## Programmer: Alberto Ramirez Bello
## Date: October 30th, 2023
##==============================================================================

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
from keras.datasets import fashion_mnist
from keras import regularizers
from scipy.signal import convolve2d

X = np.array([ [1, 1, 1, 0, 0], [0, 1, 1, 1, 0], [0, 0, 1, 1, 1], [0, 0, 1, 1, 0], [0, 1, 1, 0, 0] ])
K = np.array([ [1, 0, 1], [0, 1, 0], [1, 0, 1] ])
print('Matrix X:\n', X)
print('\nMatrix K:\n', K)
## Usando el convolve
print('\nUsando el convolve', convolve2d(X, K, mode='valid'))

(X_ent, y_ent), (X_pru, y_pru) = fashion_mnist.load_data()

## Re-Acomodando
X_ent = X_ent.reshape((60000, 28, 28, 1))
X_pru = X_pru.reshape((10000, 28, 28, 1))
X_ent = X_ent.astype('float32')
X_pru = X_pru.astype('float32')

## Normalizando
X_ent, X_pru = X_ent/255.0, X_pru/255.0
y_ent = tf.keras.utils.to_categorical(y_ent, 10)
y_pru = tf.keras.utils.to_categorical(y_pru, 10)

## Crear la CNN con dos capas convolucionales, filtros de 3x3, funcion de activacion RELU
modelo = models.Sequential()
modelo_reg = models.Sequential()
modelo_drop = models.Sequential()

modelo.add( layers.Conv2D( 32, (3, 3), activation='relu', input_shape=(28,28,1) ) )
modelo.add( layers.MaxPooling2D( pool_size=(2,2), strides=(2,2) ) )
modelo.add( layers.Conv2D( 64, (3,3), activation='relu' ) )
modelo.add( layers.MaxPooling2D( pool_size=(2,2), strides=(2,2) ))
modelo.add( layers.Flatten() )
modelo.add( layers.Dense(10, activation='softmax') )
print(modelo.summary())

modelo_reg.add( layers.Conv2D( 32, (3, 3), activation='relu', input_shape=(28,28,1), kernel_regularizer=regularizers.l2(0.001) ) )
modelo_reg.add( layers.MaxPooling2D( pool_size=(2,2), strides=(2,2) ) )
modelo_reg.add( layers.Conv2D( 64, (3,3), activation='relu' , kernel_regularizer=regularizers.l2(0.001) ))
modelo_reg.add( layers.MaxPooling2D( pool_size=(2,2), strides=(2,2) ))
modelo_reg.add( layers.Flatten() )
modelo_reg.add( layers.Dense(10, activation='softmax') )

modelo_drop.add( layers.Conv2D( 32, (3, 3), activation='relu', input_shape=(28,28,1) ) )
modelo_drop.add( layers.BatchNormalization() )
modelo_drop.add( layers.MaxPooling2D( pool_size=(2,2), strides=(2,2) ) )
modelo_drop.add( layers.Dropout(0.2) )
modelo_drop.add( layers.Conv2D( 64, (3,3), activation='relu' ) )
modelo_drop.add( layers.BatchNormalization() )
modelo_drop.add( layers.MaxPooling2D( pool_size=(2,2), strides=(2,2) ))
modelo_drop.add( layers.Dropout(0.2) )
modelo_drop.add( layers.Flatten() )
modelo_drop.add( layers.Dense(10, activation='softmax') )

## Compilando los modelos
modelo.compile( loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = modelo.fit(X_ent, y_ent, batch_size=64, epochs=10, verbose=1, validation_split=0.1)
modelo_reg.compile( loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history_reg = modelo_reg.fit(X_ent, y_ent, batch_size=64, epochs=10, verbose=1, validation_split=0.1)
modelo_drop.compile( loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history_drop = modelo_drop.fit(X_ent, y_ent, batch_size=128, epochs=10, verbose=1, validation_split=0.1)

## Puntuaciones de perdida y exactitud
puntuaciones = modelo.evaluate(X_pru, y_pru, verbose=1)
puntuaciones_reg = modelo_reg.evaluate(X_pru, y_pru, verbose=1)
puntuaciones_drop = modelo_drop.evaluate(X_pru, y_pru, verbose=1)
print('\nPerdida = {:.3f}'.format(puntuaciones[0]) )
print('Exactitud = {:.3f}'.format(puntuaciones[1]) )
print('Perdida con regularizacion = {:.3f}'.format(puntuaciones_reg[0]) )
print('Exactitud con regularizacion = {:.3f}'.format(puntuaciones_reg[1]) )
print('Perdida con dropout = {:.3f}'.format(puntuaciones_drop[0]) )
print('Exactitud con dropout = {:.3f}'.format(puntuaciones_drop[1]),'\n' )

## Graficando la perdida de entrenamiento y validacion
history_dict = history.history
loss_values  = history_dict['loss']
acc = history_dict['accuracy']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(acc)+1 )
plt.plot(epochs, loss_values, 'b', label='Perdida de entrenamiento')
plt.plot(epochs, val_loss_values, 'r', label='Perdida de Validacion')
plt.title('Perdida de entrenamiento y validacion')
plt.xlabel('Epochs')
plt.ylabel('Perdida')
plt.legend()
plt.show()

## Exactitud en el entrnamiento y la validacion
plt.clf()
acc2 = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
plt.plot(epochs, acc2, 'b', label='Exactitud de entrenamiento')
plt.plot(epochs, val_acc, 'r', label='Exactitud de validacion')
plt.title('Exactitud de entrenamiento y validacion')
plt.xlabel('Epochs')
plt.ylabel('Exactitud')
plt.legend()
plt.show()
