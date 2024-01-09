##==================================================================================
## Modelos de clasificacion I
## Book: Aprendizaje Automatico y Profundo en Python, Capitulo 7
## Description: Esta tercera parte contiene las Maquinas de Soporte de Vectores
##   (SVM). El dataset iris es utilizado para clasificar solo las Setosas como 
##   clase 0 y Versicolor como clase 1.
##   Dado que solo seran 2 tipos de flores las clasificadsa se usaran solo los 
##   primeros 100 registros.  
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
iris = datasets.load_iris()

## Separando los datos
Xiris= iris.data
yiris= iris.target
Xiris= Xiris[0:100, [1,2]]
yiris= yiris[0:100]

## Creando los grupos de pruebas y entrenamiento
Xiris_ent, Xiris_pru, yiris_ent, yiris_pru = train_test_split(Xiris, yiris, test_size=0.2, random_state=1)

## Cargando el modelo y entrenandolo
svc_iris = SVC(kernel='linear', C=1.0)
svc_iris.fit(Xiris_ent, yiris_ent)

## Generando predicciones con el modelo ya entrenado
yiris_pred = svc_iris.predict(Xiris_pru)

## Extrayendo información
print('\n+++++++++++ Iniciando el script de las SVMs +++++++++++ \n')
print(f'Exactitud de la SVM en el dataset Iris:                        {accuracy_score(yiris_pru,yiris_pred)*100}')
print('Vector de pesos (w):                                          ', svc_iris.coef_[0])
print('Punto de intersección b:                                      ', svc_iris.intercept_[0])
print('Indice de los vectores de soporte:                            ',svc_iris.support_)
print('\nVectores de soporte:\n',svc_iris.support_vectors_)
print('\nNumeros de vectores de soporte por clase:                     ', svc_iris.n_support_)
print('Coeficientes del vector de soporte en la funcion de decision: ', np.abs(svc_iris.dual_coef_))

## Dibujando hiperplano y las 100 muestras de las clases Setosa y Versicolor
df = pd.DataFrame(Xiris_ent, columns=['Ancho del sepalo', 'Longitud del petalo'])
df['clase'] = yiris_ent
wiris = svc_iris.coef_[0]                           ### Pesos
pendiente = -wiris[0] / wiris[1]                    ### Pendiente del hiperplano
biris = svc_iris.intercept_[0]                      ### Punto de interseccion
xxiris = np.linspace(1, 5.1)                        ### Coordenadas del hiperplano
yyiris = pendiente * xxiris - (biris / wiris[1])    ### Coordenadas del hiperplano
siris = svc_iris.support_vectors_[0]                ### Primer vector de soporte, siris es una variable temporal
yyiris_bajo = pendiente * xxiris + ( siris[1] - pendiente * siris[0] )
siris = svc_iris.support_vectors_[2]                ### Ultimo vector de soporte
yyiris_arriba = pendiente * xxiris + ( siris[1] - pendiente * siris[0] )

print('Longitud del Xiris_ent: ', len(Xiris_ent))
print('Longitud del yiris_ent: ', len(yiris_ent))

sc=StandardScaler()
Xiris_setFit = sc.fit_transform(Xiris_ent)
Xiris_set =sc.inverse_transform(Xiris_setFit)

for y in range(2):
    plt.scatter(Xiris_ent[yiris_ent==y,0], Xiris_ent[yiris_ent==y,1], c=ListedColormap(['red','blue'])(y), label='clase '+ format(y) )

plt.plot(xxiris, yyiris, linewidth=2, color='green')### Dibujo de hiperplano
plt.plot(xxiris, yyiris_bajo, 'k--')                ### Dibujo de margenes
plt.plot(xxiris, yyiris_arriba, 'k--')              ### Dibujo de margenes
plt.xlabel('Ancho del sépalo',fontsize=15)
plt.ylabel('Longitud del pétalo',fontsize=15)
plt.title('SVM Lineal')
plt.show()