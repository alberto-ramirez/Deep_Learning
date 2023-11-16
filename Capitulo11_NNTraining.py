##==============================================================================
## Introduction to Neural Networks
## Book: Aprendizaje Automatico y Profundo en Python, Capitulo 11
## Description: Neural Network Trainning
## Author: Carlos M. Pineda Pertuz
##==============================================================================
## Programmer: Alberto Ramirez Bello
## Date: October 21st, 2023
##==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import datasets

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

## Funciones utilitarias, para la red neuronal (funciones de activacion)
def calcular_costo(a2, y):
    costo= -1/m * (np.sum( np.multiply( np.log(a2), y ) + np.multiply( (1-y), np.log(1-a2) ) ) )
    return costo

def sigmoide(t):
    return 1/(1 + np.exp(-t) )

def derivada_sigmoide(p):
    return sigmoide(p) * sigmoide(1-p)

def relu(Z):
    return np.maximum(0,Z)

def derivada_Relu(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x

def exactitud(y,yhat):
    acc = int( sum(y==yhat / len(y)*100 ) )
    return acc

def predecir(x):
    w1 = modelo['w1']
    b1 = modelo['b1']
    w2 = modelo['w2']
    b2 = modelo['b2']
    z1 = np.dot(x, w1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = sigmoide(z2)
    p  = np.round(a2)
    return p

def forward(X, w1, w2, b1, b2):
    '''Realiza la propagacion hacia adelante, para la primera capa aplica la funcion de activacion Relu,
       Para la segunda capa se utiliza la funcion de activacion sigmoide'''
    
    z1 = np.dot( X, w1 ) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = sigmoide(z2)
    cache = { 'z1':z1, 'a1':a1, 'z2': z2, 'a2':a2 } ## Guarda los datos que seran usados en ña funcion backpropagation
    return a2, cache

def backward(w1, b1, w2, b2, cache, alfa):
    '''Propagacion hacia atras, iniciando de la capa de salida'''

    a1  = cache['a1']
    a2  = cache['a2']
    z1  = cache['z1']
    z2  = cache['z2']
    dz2 = ( a2 - y_ent ) * derivada_sigmoide(z2)
    dw2 = np.dot(a2.T, dz2)
    db2 = np.mean( dz2, axis=0 )
    dz1 = np.dot( dz2, w2.T ) * derivada_Relu(z1)
    dw1 = np.dot( X_ent.T, dz1 )
    db1 = np.mean(dz1, axis=0)
    w2  = w2 - alfa * dw2 / m
    b2  = b2 - alfa * db2 / m
    w1  = w1 - alfa * dw1 / m 
    db1 = np.mean(dz1, axis=0)
    w2  = w2 - alfa * dw2 / m
    b2  = b2 - alfa * db2 / m
    w1  = w1 - alfa * dw1 / m
    b1  = b1 - alfa * db1 / m

    return w1, w2, b1, b2

def entrenar(X, y, alfa, num_iter):
    global modelo
    global errores 
    errores = []
    unid_entrada = X.shape[1]
    unid_oculta = 10
    unid_salida = 1
    np.random.seed(1)
    w1 = np.random.randn(unid_entrada, unid_oculta) # Pesos de la capa 1
    b1 = np.random.randn(1, unid_oculta) # Sesgo capa 1
    w2 = np.random.randn(unid_oculta, unid_salida) # Pesos capa 2
    b2 = np.random.randn(1, unid_salida) # Sesgo capa 2

    for i in range(num_iter):
        a2, cache = forward(X, w1, w2, b1, b2)
        costo = calcular_costo(a2, y)
        errores.append(costo)
        w1, w2, b1, b2 = backward(w1, b1, w2, b2, cache, alfa)

        if i % 100 == 0:
            print(f'Costo despues de la iteracion i = {i} : {costo}')

    modelo = {'w1':w1, 'b1':b1, 'w2': w2, 'b2': b2}

entrenar(X_ent, y_ent, alfa=0.01, num_iter=10000)

pl.plot(errores)
pl.xlabel('Epoca')
pl.ylabel('Error')
pl.title('Curva de entrenamiento')
pl.show()

pred_ent = predecir(X_ent)
pred_pru = predecir(X_pru)
print('\nExactitud de entrenamiento: {}'.format( exactitud(y_ent, pred_ent) ) )
print('Exactitud de prueba: {}'.format( exactitud(y_pru, pred_pru) ) )

### Utilizando scikit learn
clf = MLPClassifier(max_iter=10000, learning_rate_init=0.01, hidden_layer_sizes=10, activation='relu', solver='sgd')
clf.fit( X_ent, y_ent.ravel() )
clf.predict_proba(X_pru[:1])
exac = clf.score(X_pru, y_pru)
print('\n----------------------- Usando la libreria MLPClassifier de Scikit-learn\nExactitud = {:.2f}'.format(exac))