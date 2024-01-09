##==================================================================================
## Modelos de clasificacion I
## Book: Aprendizaje Automatico y Profundo en Python, Capitulo 7
## Description: Esta cuarta parte contiene el SVM lineal para clasificacion
##   Multiclase, de igual manera se utilizara el dataset iris
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
from matplotlib.colors import ListedColormap

## Cargando los datasets
print('\n++++++++++++++++ Inicializando el script del modelo: SVM +++++')
iris = datasets.load_iris()

## Separando los datos
Xiris= iris.data[:, [0,1]] 
yiris= iris.target

## Creando los grupos de pruebas y entrenamiento
Xiris_ent, Xiris_pru, yiris_ent, yiris_pru = train_test_split(Xiris, yiris, test_size=0.1, random_state=1)
colores = ['red', 'green', 'blue']

for color, i, target in zip(colores, [0, 1, 2], iris.target_names):
    plt.scatter( Xiris_ent[yiris_ent == i, 0], Xiris_ent[yiris_ent == i, 1], color=color, label=target )

plt.xlabel('Longitud del Sépalo')
plt.ylabel('Ancho del Pétalo')
plt.legend(loc='best')
plt.show()

### Entrenar el modelo SVM
ciris = 1
clf_iris = SVC(kernel='linear',C=ciris).fit(Xiris,yiris)
titulo = 'SVM como clasificador multiclase'
xiris_min, xiris_max = Xiris_ent[:,0].min() - 1, Xiris_ent[:,0].max() + 1
yiris_min, yiris_max = Xiris_ent[:,1].min() - 1, Xiris_ent[:,1].max() + 1
hiris = ( xiris_max / xiris_min ) / 100
xxiris, yyiris = np.meshgrid( np.arange(xiris_min, xiris_max, hiris), np.arange(yiris_min, yiris_max, hiris) )
Ziris = clf_iris.predict( np.c_[xxiris.ravel(), yyiris.ravel()] )
Ziris = Ziris.reshape(xxiris.shape)
plt.contourf(xxiris, yyiris, Ziris, cmap=plt.cm.Accent, alpha=0.8)

for color, j, taryet in zip(colores, [0,1,2], iris.target_names):
    plt.scatter( Xiris[yiris == j, 0], Xiris[yiris == j, 1], color=color, label=taryet)

plt.xlabel('Longitud del Sépalo')
plt.ylabel('Ancho del Pétalo')
plt.title(titulo)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.show()  

### Realizacion de prueba para revalidar resultados del modelo
prueba1 = [[5, 4]]
pred = clf_iris.predict(prueba1)
print(f"\nPredicción del modelo: {pred[0]}, que es una Setosa\nExactitud del modelo: {clf_iris.score(Xiris_pru, yiris_pru)}\n")

### SubGraficas para cambiar el valor de C
fig, subejes = plt.subplots(1, 3, figsize=(18,4) )

for c, subeje in zip([0.1, 1, 100],subejes):
    clf = SVC( kernel='linear', C=c, random_state=67).fit(Xiris_ent, yiris_ent)
    titulo2 = f'SVC Lineal, C={c:.4f}'

    for color, g, taryet2 in zip(colores, [0,1,2], iris.target_names):
        subeje.scatter(Xiris_ent[yiris_ent == g, 0], Xiris_ent[yiris_ent == g, 1], color=color, label=taryet2, alpha=0.9)
        subeje.set_xlabel('Longitud del Sépalo', fontsize=12)
        subeje.set_ylabel('Ancho del Pétalo', fontsize=12)
        subeje.set_title(titulo2)

        for w, b, color in zip(clf.coef_, clf.intercept_, colores):
            x_min, x_max = Xiris_ent[:,0].min() - 1, Xiris_ent[:,0].max() + 1
            h = (x_max / x_min) / 100
            xx = np.arange(x_min, x_max, h)
            pendiente = -w[0] / w[1]
            yy = pendiente * xx - ( b / w[1] )
            subeje.plot(xx, yy, c=color, alpha=0.8)
            subeje.legend( ['Clase Setosa', 'Clase Versicolor', 'Clase Virginica', 'Linea Clase Setosa', 'Linea Clase Versicolor', 'Linea Clase Virginica'], loc='best')
            subeje.set_xlim(3, 8)
            subeje.set_ylim(1, 8)
plt.show()            