##=======================================================================================
## Clustering
## Book: Aprendizaje Automatico y Profundo en Python, Capitulo 9
## Description: Algoritmos de agrupamiento: (K-medias, Clustering jerarquico y DBSCAN).
## Author: Carlos M. Pineda Pertuz
##=======================================================================================
## Programmer: Alberto Ramirez Bello
## Date: August 23rd, 2024
##=======================================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn import metrics
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from scipy.spatial import distance_matrix
from scipy.cluster import hierarchy
from sklearn.datasets import make_moons

datos = load_breast_cancer()

## Revisar el contenido
print(f'\n====== Cargando los datos para trabajar: ======\n {datos.feature_names}')

df = pd.DataFrame( datos.data, columns = datos.feature_names )
df['target'] = datos.target
print(f'\n------- Dataframe creado: -------\n {df}')

## Tomando solo 3 caracteristicas para trabajar
X = np.array( list( zip( df['mean radius'], df['mean perimeter'] )))
print(f'\n------- Arreglo Numpy: mean radius & mean perimeter -------\n {X}')

## Objeto clasificador
knn = KMeans( n_clusters = 2 )

## Entrenando el modelo
knn.fit(X)

## Centroides (arrays con 31 valores)
centroides = knn.cluster_centers_
etiquetas = knn.labels_
print(f'\n------- Centroides obtenidos: -------\n {centroides}\n\nCluster de cada centroide:\n{etiquetas}')

## Prediccion con una muestra: mean radius = 20, mean perimeter = 140
muestra = [[20,140]]
pred = knn.predict(muestra)
print(f'\nLa Muestra: {muestra} se encuentra en el cluster: {pred}')

## Visualizando datos
c = [ 'r' , 'y' ]
colors = [ c[i] for i in etiquetas ]
plt.scatter( df['mean radius'], df['mean perimeter'], c = colors, s = 20 )
plt.scatter( centroides[:,0], centroides[:,1], marker = '*', s = 100, c = 'green', edgecolors = 'black')
plt.xlabel( 'mean radius' )
plt.ylabel( 'mean perimeter' )
plt.show()

## Calcular el numero optimo de Clusteres

### Elbow
inercias = []

for i in range(1,11):
    kmeans = KMeans( n_clusters = i, init = 'k-means++', random_state = 42 )
    kmeans.fit( X )
    inercias.append( kmeans.inertia_ )

#### Suma de las distancias
plt.plot( range(1,11), inercias )
plt.title('Método del codo')
plt.xlabel('Número de clusters')
plt.show()

#### Silueta
exactitud = metrics.silhouette_score( X, etiquetas )
print(f'\nCoeficiente de la silueta: {exactitud:.3f}')

#### valor optimo de k entre 2 y 10
proms_silueta = []
k_ini = 2

for k in range(k_ini, 10):
    kmean = KMeans( n_clusters = k ).fit( X )
    exactitud = metrics.silhouette_score( X, kmean.labels_ )
    print(f"\nCoeficiente de silueta para k = {k} es {exactitud:.3f}")
    proms_silueta.append(exactitud)

#### el valor de k optimo es el que tiene el promedio más alto
k_optimo = proms_silueta.index( max( proms_silueta ) ) + k_ini
print(f'\nEl valor óptimo de K es: {k_optimo}')

#### Graficando la silueta
plt.plot( range( k_ini, 10 ), proms_silueta )
plt.title( "Coeficiente de silueta" )
plt.xlabel( "Numero de clusteres" )
plt.ylabel( "Coeficientes de silueta" )
plt.show()

### Clusteres aglomerativos:
agl = AgglomerativeClustering( n_clusters = 4, linkage = 'complete')
agl.fit( X )
matriz_distancias = distance_matrix( X, X )
print(f'\nMatriz de distancias para el dendograma:\n{matriz_distancias}')

Z = hierarchy.linkage( matriz_distancias, 'complete' )
dendro = hierarchy.dendrogram( Z )
plt.title( "Cluster Aglomerativo" )
plt.xlabel('Indice de muestra')
plt.ylabel('Distancia de cluster')
plt.show()

#### Density Based Spatial Clustering of Applications with Noise
num_muestras = 2000
H,I = make_moons(n_samples = num_muestras, noise = 0.05)
plt.scatter( H[:,0], H[:,1], c = I, cmap = 'viridis', s = 50 )
plt.title( "Muestras para DBSCA" )
plt.xlabel( 'Caracteristica 1' )
plt.ylabel( 'Caracteristica 2' )
plt.show()

##### K-Means
kmeans2 = KMeans( n_clusters = 2 )
kmeans2.fit(H)
y_pred = kmeans2.predict(H)
plt.scatter( H[:,0], H[:,1], c = y_pred, cmap = 'viridis', s = 50 )
plt.scatter( kmeans2.cluster_centers_[:,0], kmeans2.cluster_centers_[:,1], marker = '*', c = np.unique( y_pred ), 
            cmap = 'RdBu', s = 150, linewidth = 2 )
plt.title( "K-Means" )
plt.xlabel( 'Caracteristica 1' )
plt.ylabel( 'Caracteristica 2' )
plt.show()


##### DBScan
dbs = DBSCAN(eps = 0.1)
J = dbs.fit_predict(H)
plt.scatter( H[:,0], H[:,1], c = J, cmap = 'viridis', s = 50)
plt.title( "DBScan" )
plt.xlabel( 'Caracteristica 1' )
plt.ylabel( 'Caracteristica 2' )
plt.show()
