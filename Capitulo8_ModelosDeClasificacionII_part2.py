##==================================================================================
## Modelos de clasificacion II
## Book: Aprendizaje Automatico y Profundo en Python, Capitulo 8
## Description: Esta segunda parte contiene un ejemplo de bosque aleatorio
##    Adaboost (Adaptative Boosting)
##    Dataset: Iris.
## Author: Carlos M. Pineda Pertuz
##==================================================================================
## Programmer: Alberto Ramirez Bello
## Date: January 15th, 2024
##==================================================================================

import numpy as np
import pydot
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

print('\n-------------- Iniciando el bosque aleatorio -------------------------------------------')

iris = datasets.load_iris()
X = iris.data
y = iris.target
X = X[:, [0,1]]
X_ent, X_pru, y_ent, y_pru = train_test_split(X, y, test_size=0.2, random_state=1)
arbolito = DecisionTreeClassifier(criterion='gini', max_depth=3)
arbolito.fit(X_ent, y_ent)
clases = iris.target_names
indices = [0, 1, 2]
colores = ['red', 'green', 'blue']

print('\nMostrando resultados obtenidos del arbolito...')
print(f'Exactitud del conjunto de entrenamiento: {arbolito.score(X_ent, y_ent):.5f}')
print(f'Exactitud del conjunto de prueba: {arbolito.score(X_pru, y_pru):.5f}')


### Puntos de datos y limites de decision
plt.figure()

for color, i, target in zip(colores, indices, clases):
    ax = plt.scatter( X_ent[y_ent == i, 0], X_ent[y_ent == i, 1], color=color, label=target )

xlim = plt.xlim()
ylim = plt.ylim()
xx, yy = np.meshgrid( np.linspace(*xlim, num=200), np.linspace(*ylim, num=200) )
Z = arbolito.predict( np.c_[xx.ravel(), yy.ravel()] ).reshape(xx.shape)
contornos = plt.contourf(xx, yy, Z, alpha=0.3, cmap='Set2')

plt.legend( ['Setosa', 'Versicolor', 'Virginica'], loc='best' )
plt.xlabel( 'Longitud del Sépalo', fontsize=12 )
plt.ylabel( 'Ancho del Pépalo', fontsize=12 )
plt.title( 'Arbol de Decision con datos IRIS')
plt.show()

### Para entrenar al bosque
bosque = RandomForestClassifier(random_state=0, n_jobs=-1, criterion='gini', n_estimators=15)
bosque.fit(X_ent, y_ent)

plt.figure()

for color, i, target in zip(colores, indices, clases):
    axrf = plt.scatter( X_ent[y_ent == i, 0], X_ent[y_ent == i, 1], color=color, label=target )

xlimrf = plt.xlim()
ylimrf = plt.ylim()
xxrf, yyrf = np.meshgrid( np.linspace(*xlimrf, num=200), np.linspace(*ylimrf, num=200) )
Zrf = bosque.predict( np.c_[xxrf.ravel(), yyrf.ravel()] ).reshape(xxrf.shape)
contornosrf = plt.contourf(xxrf, yyrf, Zrf, alpha=0.3, cmap='Set2')

plt.legend( ['Setosa', 'Versicolor', 'Virginica'], loc='best' )
plt.xlabel( 'Longitud del Sépalo', fontsize=12 )
plt.ylabel( 'Ancho del Pépalo', fontsize=12 )
plt.title( 'Bosque Aleatorio con datos IRIS')
plt.show()


print('\nMostrando resultados obtenidos del bosque aleatorio...')
print(f'Exactitud del conjunto de entrenamiento: {bosque.score(X_ent, y_ent):.5f}')
print(f'Exactitud del conjunto de prueba: {bosque.score(X_pru, y_pru):.5f}')
print('\nComenzando el ADAboost')

### Adaptative boosting
adaboost = AdaBoostClassifier( DecisionTreeClassifier(max_depth=2), n_estimators=25, learning_rate=0.1, random_state=0 )
ada_model = adaboost.fit( X_ent, y_ent )
print('Mostrando resultados obtenidos del AdaBoost...')
print(f'Exactitud del conjunto de entrenamiento: {ada_model.score(X_ent, y_ent):.5f}')
print(f'Exactitud del conjunto de prueba: {ada_model.score(X_pru, y_pru):.5f}')

plt.figure()

for color, i, target in zip(colores, indices, clases):
    axada = plt.scatter( X_ent[y_ent == i, 0], X_ent[y_ent == i, 1], color=color, label=target )

xlimada = plt.xlim()
ylimada = plt.ylim()
xxada, yyada = np.meshgrid( np.linspace(*xlimada, num=200), np.linspace(*ylimada, num=200) )
Zada = ada_model.predict( np.c_[xxada.ravel(), yyada.ravel()] ).reshape(xxada.shape)
contornosada = plt.contourf(xxada, yyada, Zada, alpha=0.3, cmap='Set2')

plt.legend( ['Setosa', 'Versicolor', 'Virginica'], loc='best' )
plt.xlabel( 'Longitud del Sépalo', fontsize=12 )
plt.ylabel( 'Ancho del Pépalo', fontsize=12 )
plt.title( 'Adaptative Boosting con datos IRIS')
plt.show()

### Gradient Boosting
gradboost = GradientBoostingClassifier( n_estimators=10, learning_rate=1.0, max_depth=1, random_state=0 )
gradboost_model = gradboost.fit( X_ent, y_ent )
puntajesvc = []
max_estimadores = 25

for i in range(1, max_estimadores):
    punt = cross_val_score( GradientBoostingClassifier( n_estimators=i, learning_rate=10.0/float(i) ), X_ent, y_ent, cv=10, scoring='accuracy' ).mean()
    puntajesvc.append(punt)

plt.plot( range(1, max_estimadores), puntajesvc )
plt.xlabel('Número de estimadores')
plt.ylabel('Promedio de validacion cruzada')
plt.show()

plt.figure()

for color, i, target in zip(colores, indices, clases):
    axgb = plt.scatter( X_ent[y_ent == i, 0], X_ent[y_ent == i, 1], color=color, label=target )

xlimgb = plt.xlim()
ylimgb = plt.ylim()
xxgb, yygb = np.meshgrid( np.linspace(*xlimgb, num=200), np.linspace(*ylimgb, num=200) )
Zgb = gradboost_model.predict( np.c_[xxgb.ravel(), yygb.ravel()] ).reshape(xxgb.shape)
contornosgb = plt.contourf(xxgb, yygb, Zgb, alpha=0.3, cmap='Set2')

plt.legend( ['Setosa', 'Versicolor', 'Virginica'], loc='best' )
plt.xlabel( 'Longitud del Sépalo', fontsize=12 )
plt.ylabel( 'Ancho del Pépalo', fontsize=12 )
plt.title( 'Gradient Boosting con datos IRIS')
plt.show()