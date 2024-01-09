##==================================================================================
## Modelos de clasificacion I
## Book: Aprendizaje Automatico y Profundo en Python, Capitulo 7
## Description: Esta quinta parte contiene el SVM lineal para clasificacion
##   Multiclase, de igual manera se utilizara el dataset iris
##   Se modificara el kernel utilizado ya que seran datos no lineales.
## Author: Carlos M. Pineda Pertuz
##==================================================================================
## Programmer: Alberto Ramirez Bello
## Date: January 8th, 2024
##==================================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_circles
from matplotlib.colors import ListedColormap

print('\n+++++++ Iniciando el script con SVM y variacion de kernels ++++++++++++')

iris = datasets.load_iris()
neims = iris.target_names
X,y = make_circles(n_samples=175, noise=0.05)
rgb = np.array(['r','b'])

for i in range(2):
    plt.scatter(X[y==i,0], X[y==i,1], label=f'Clase {i}', color=rgb[i])

plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

X_ent, X_pru, y_ent, y_pru = train_test_split(X, y, test_size=0.1, random_state=1)
svc = SVC(kernel='linear', C=1, random_state=1)
svc.fit(X_ent, y_ent)
y_pred = svc.predict(X_pru)
svc_rbf = SVC(kernel='rbf', C=2, random_state=1, degree=3, gamma='auto')
svc_rbf.fit(X_ent, y_ent)
y_pred_rbf = svc_rbf.predict(X_pru)

print(f"\nExactitud del SVC con kernel lineal: {accuracy_score(y_pru, y_pred)*100}")
print(f"Exactitud del SVC con kernel rbf:   {accuracy_score(y_pru, y_pred_rbf)*100}")

for i in range(2):        
    plt.scatter(X_ent[y_ent==i,0], X_ent[y_ent==i,1], label=f'clase {i}', color=rgb[i])
    
w = svc.coef_[0]
pendiente = -w[0] / w[1]
b = svc.intercept_[0]    
xx = np.linspace(-1, 1)
yy = pendiente * xx - (b / w[1]) 
  
plt.xlabel('x1', fontsize=15)
plt.ylabel('x2', fontsize=15)
plt.title("SVC con kernel lineal")
plt.plot(xx, yy, linewidth=2, color='red')
plt.show()

fig, subejes = plt.subplots(1, 4, figsize=(18, 4) )

for subeje, c, g in zip(subejes, [0.1, 0.5, 1, 10], [1, 5, 10, 100]):
    clf = SVC(kernel='rbf', C=c, gamma=g, random_state=67).fit(X_ent, y_ent)
    titulo = f'SVC Lineal, C = {c:.4f}'
    colores = ['red', 'green', 'blue']

    for color, j, target in zip(colores, [0,1,2], neims):
        subeje.scatter(X_ent[y_ent == j, 0], X_ent[y_ent == j, 1], color=color, label=target, alpha=0.9)
        subeje.set_xlabel('Longitud del Sépalo', fontsize=12)
        subeje.set_ylabel('Ancho del Pétalo', fontsize=12)
        subeje.set_title(titulo)
        colores2 = ['red', 'blue', 'green']
        cmaping = ListedColormap(colores2[:len( np.unique(y) ) ])
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx1, xx2 = np.meshgrid( np.arange(x1_min, x1_max, 0.02), np.arange(x2_min, x2_max, 0.02) )
        Z = clf.predict( np.array( [xx1.ravel(), xx2.ravel()] ).T )
        Z = Z.reshape(xx1.shape)
        subeje.contourf( xx1, xx2, Z, alpha=0.4, cmap=cmaping )
        subeje.set_xlim(xx1.min(), xx1.max())
        subeje.set_ylim(xx2.min(), xx2.max())
plt.show()        
