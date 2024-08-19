##==================================================================================
## Modelos de clasificacion II
## Book: Aprendizaje Automatico y Profundo en Python, Capitulo 8
## Description: Esta primera parte contiene un ejemplo de arbol de decisión
##    Utilizando la clase DecisionTreeClassifier de scikit-learn
##    Dataset: Aprobacion de prestamos y 3 niveles de profundidad.
## Author: Carlos M. Pineda Pertuz
##==================================================================================
## Programmer: Alberto Ramirez Bello
## Date: January 13th, 2024
##==================================================================================

import numpy as np
import pydot
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn import datasets
from sklearn.model_selection import train_test_split

print('\n-------------- Iniciando el arbol de decision -------------------------------------------')

iris = datasets.load_iris()
X = np.array( [[ 'No', 'Bajo'], ['Si', 'Medio'], ['Si', 'Alto'], ['No', 'Alto'], ['Si','Bajo'],
               [ 'No', 'Bajo'], ['No', 'Medio'], ['Si', 'Alto'], ['No', 'Alto'], ['No','Bajo']] )
y = [0, 1, 1, 1, 1, 0, 0, 1, 1, 0]
le = LabelEncoder()
X[:,0] = le.fit_transform( X[:,0] )
X[:,1] = le.fit_transform( X[:,1] )
arbol = DecisionTreeClassifier( criterion='gini', max_depth=3, random_state=1 )
arbol.fit(X,y)

### Realizando una prueba con individuo que tiene casa y un ingreso bajo
prueba = np.array( ['Si', 'Bajo'] )
prueba_cod = le.fit_transform(prueba)
prediccion = arbol.predict(prueba_cod.reshape(1, -1))
print(f'\nPreidiccion del arbol de decision para la prueba 1: [Si, Bajo] = {prediccion[0]}')

### Creando archivo .dot para la generacion de la imagen del arbol
#dot = tree.export_graphviz(arbol, feature_names=['Tiene_Casa', 'Ingreso_anual'], class_names=['Aprobado', 'No_aprobado'], out_file='arbol.dot')
#(graph,) = pydot.graph_from_dot_file('arbol.dot')
#graph.write_png('arbol1.png')

### Identificacion de las caracteristicas importantes
Xiris = iris.data
yiris = iris.target
Xiris_ent, Xiris_pru, yiris_ent, yiris_pru = train_test_split(Xiris, yiris, test_size=0.2, random_state=1)
arbol_iris = DecisionTreeClassifier(criterion='gini', max_depth=3)
arbol_iris.fit(Xiris_ent, yiris_ent)
print(f'Exactitud del conjunto de entrenamiento: {arbol_iris.score(Xiris_ent, yiris_ent):.4f}')
print(f'Exactitud del conjunto de prueba: {arbol_iris.score(Xiris_pru, yiris_pru):.4f}')
print('Creando archivo .dot y .png para visualizar el arbol generado...')
dot_iris = tree.export_graphviz(arbol, feature_names=['Tiene_Casa', 'Ingreso_anual'], class_names=['Aprobado', 'No_aprobado'], out_file='arbol_iris.dot')
(graph,) = pydot.graph_from_dot_file('arbol_iris.dot')
graph.write_png('arbol_iris.png')

### Utilizando la propiedad features_importances_ del clasificador para determinar las caracteristicas más importantes del datset
caracteristicas_importantes = arbol_iris.feature_importances_
indices = np.argsort(caracteristicas_importantes)[::1]  # Ordenar las caracteristicas en orden descendente
nombres = [iris.feature_names[i] for i in indices]
plt.title('Caracteristicas importantes del dataset iris')
plt.bar(range(Xiris.shape[1]), caracteristicas_importantes[indices])
plt.xticks(range(Xiris.shape[1]), nombres, rotation=90)
plt.show()