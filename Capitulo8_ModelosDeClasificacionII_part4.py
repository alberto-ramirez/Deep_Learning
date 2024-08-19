##==================================================================================
## Modelos de clasificacion II
## Book: Aprendizaje Automatico y Profundo en Python, Capitulo 8
## Description: Esta segunda tercera contiene un ejemplo del modelo KNN
##    K vecinos más cercanos (KNN)
##    Dataset: Spam  No Spam.
## Author: Carlos M. Pineda Pertuz
##==================================================================================
## Programmer: Alberto Ramirez Bello
## Date: August 11th, 2024
##==================================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

iris=datasets.load_iris()
X=iris.data
y=iris.target
X=X[0:100, [1,2]]
y=y[0:100]
X_ent, X_pru, y_ent, y_pru=train_test_split(X,y,test_size=0.2,random_state=1)

sc=MinMaxScaler()
X_ent=sc.fit_transform(X_ent)
X_pru=sc.transform(X_pru)
knn=KNeighborsClassifier(n_neighbors=5,metric='euclidean')
knn.fit(X_ent, y_ent)
print(f'Exactitud: {knn.score(X_pru,y_pru)}')

#### Test's sample prediction
prueba=np.array([1,1])
prediccion=knn.predict(prueba.reshape(1,-1))
print(f'Predicción: {prediccion}')

#### Nearest neighbors to our sample
distancias, indices = knn.kneighbors(prueba.reshape(1,-1))
print('5 vecinos mas cercanos...')

for i, j in enumerate(indices[0][:5]):
    print(X_ent[j])

X1, X2 = np.meshgrid( np.arange( start = X_ent[:,0].min() - 1,
                                 stop  = X_ent[:, 0].max() + 1,
                                 step  = 0.1 ),
                      np.arange( start = X_ent[:,1].min() - 1,
                                 stop  = X_ent[:,1].max() + 1,
                                 step  = 0.1 ) )
plt.contourf( X1, X2, knn.predict( np.array( [X1.ravel(), X2.ravel()] ).T ).reshape( X1.shape ), cmap = plt.cm.BuGn )
colors = ['red', 'yellow']

for color, i, target in zip(colors, [0,1], iris.target_names):
    ax = plt.scatter( X_ent[y_ent==i, 0], X_ent[y_ent==i, 1], color=color, label=target, s=150, edgecolor='black' )

for i in indices:
    plt.scatter( X_ent[i,0], X_ent[i,1], marker='o', s=150, edgecolor='black' )

plt.scatter(1,1, marker='*', s=150, color='white', edgecolor='black')

plt.title('Ejemplo KNN')
plt.xlabel('Ancho Sépalo')
plt.ylabel('Longitud Pétalo')
plt.legend()
plt.show()