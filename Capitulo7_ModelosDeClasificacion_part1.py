##==============================================================================
## Modelos de clasificacion I
## Book: Aprendizaje Automatico y Profundo en Python, Capitulo 7
## Description: Perceptron, Adaline, Logistic regression, Evaluation Metrics, SVM
##              Por simplicidad, el modelo de regresion logistica y SVM estaran 
##              En otros scripts, esta primera parte solo contiene al perceptron
##              y a ADELINE
## Author: Carlos M. Pineda Pertuz
##==============================================================================
## Programmer: Alberto Ramirez Bello
## Date: January 5th, 2024
##==============================================================================

import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

### Funciones para el perceptron

def obtener_entrada_red(x,pesos):
    """Esta funcion calcula un valor para la variable 'y' basado en los pesos"""
    prediccion = x.dot(pesos[1:]) + pesos[0]
    return prediccion

def calcular_prediccion(X,pesos):
    """Esta funcion permite predecir la clase de una instancia"""
    z = obtener_entrada_red(X,pesos)

    if (z >= 0.0):
        activacion = 1
    else:
        activacion = 0

    return activacion        

def entrenar(X,y,pesos,max_iter,tasa_aprendizaje,errors):
    """Aplicando el descenso del gradiente"""
    for i in range(max_iter):
        error = 0

        for xi, yi, in zip(X,y):
            prediccion = calcular_prediccion(xi,pesos)
            error = yi - prediccion
            actualizacion = tasa_aprendizaje*error
            pesos[1:] += actualizacion*xi
            pesos[0] += actualizacion
        errors.append(error)
    return pesos        

### Fin de las funciones para el perceptron

### Funciones para ADELINE
def obtener_entrada_ade(x,pesos):
    """Esta funcion calcula un valor para la variable 'y' basado en los pesos"""
    z=x.dot(pesos[1:])+pesos[0]
    return z

def activacion_ade(X,pesos):
    return obtener_entrada_ade(X,pesos)

def calcular_prediccion_ade(X,pesos):
    """Esta funcion permite predecir la clase de una instancia"""
    a=activacion_ade(X,pesos)

    if ( a >= 0.0 ):
        res=1
    else:
        res=0

    return res        

def entrenar_ade(X,y,pesos,max_iter,tasa_aprendizaje,costos):
    """Utilizando el descenso del gradiente"""
    for i in range(max_iter):
        z=obtener_entrada_ade(X,pesos)
        error=y-z
        pesos[1:] += tasa_aprendizaje*X.T.dot(error)
        pesos[0]  += tasa_aprendizaje*error.sum()
        costo = (error**2).sum()/2
        costos.append(costo)

    return pesos    
### Fin de las Funciones para ADELINE

## Percerptron para aprender funcion logica OR
errors = []
max_iter=10
tasa_aprendizaje=0.01
X = np.array( [ [1,1], [0,1], [1,0], [0,0] ] )
y = np.array( [1, 1, 1, 0] )
pesos = np.zeros(X.shape[1]+1)

print('\nIniciando el perceptron...')

pesos = entrenar(X,y,pesos,max_iter,tasa_aprendizaje,errors)
print(f'Valores obtenidos de los pesos: {pesos}')

### realizando pruebas con 3 muestras
muestra1 = np.array([1,1])
muestra2 = np.array([0,0])
muestra3 = np.array([1,0])
muestra4 = np.array([[0,0]])

p1 = calcular_prediccion(muestra1,pesos)
p2 = calcular_prediccion(muestra2,pesos)
p3 = calcular_prediccion(muestra3,pesos)

print('\nPrediccion de la muestra 1 con valores [1,1]: {:.2f}'.format(p1))
print('Prediccion de la muestra 2 con valores [0,0]: {:.2f}'.format(p2))
print('Prediccion de la muestra 3 con valores [1,0]: {:.2f}'.format(p3))

### Graficando el modelo para comprobar el funcionamiento
plt.plot(range(1,len(errors)+1), errors, marker='o')
plt.xlabel('Iteraciones')
plt.ylabel('Error')
plt.show()

### Utilizando el perceptron de la libreria scikit-learn
clf = Perceptron(random_state=1, alpha=0.01, max_iter=10)
clf.fit(X,y)
print('\nExactitud del modelo de scikit-learn: {:.2f}'.format(clf.score(X,y)))
print('Contenido de la muestra 4: ',muestra4)
print('Prediccion del modelo de scikit-learn con la muestra 4 que tiene el mismo valor que la muestra 2 pero es multidimensional: {:.2f}'.format(clf.predict(muestra4)[0]))

### Neurona Lineal Adaptativa (ADALINE), se utilizara el dataset iris
iris = datasets.load_iris()
Xiris = iris.data
yiris = iris.target
#print('\nMostrando Xiris antes del reacomodo...\n',Xiris)
Xiris = Xiris[0:100,[1,2]] # Los primeros 100 elementos de solo 2 de las 3 clases de flores que hay
yiris = yiris[0:100]
#print('\nMostrando Xiris despues del reacomodo...\n',Xiris)
### Estandarizando los datos
Xiris[:,0] = ( Xiris[:,0] - Xiris[:,0].mean() ) / Xiris[:,0].std()
Xiris[:,1] = ( Xiris[:,1] - Xiris[:,1].mean() ) / Xiris[:,1].std()

### Graficando el dataset
plt.scatter( Xiris[:50,0], Xiris[:50,1], color='red', marker='o', label='Setosa' )
plt.scatter( Xiris[50:100,0], Xiris[50:100,1], color='blue', marker='x', label='Versicolor' )
plt.xlabel('Ancho Sépalo')
plt.ylabel('Longitud Pépalo')
plt.legend(loc='upper left')
plt.show()

### Inicializando los pesos
rgen = np.random.RandomState(1)
pesosAd = rgen.normal( loc=0.0, scale=0.01, size=1+Xiris.shape[1] )
errorsAd = []
costos = []

### Llamando a la funcion
pesosAd = entrenar_ade(Xiris, yiris, pesosAd, max_iter, tasa_aprendizaje, costos)
### Grafica de convergencia
plt.plot(range(1, len(costos)+1 ), costos, marker='o')
plt.xlabel('Iteraciones')
plt.ylabel('Error')
plt.show()

### Regresion logistica con el dataset breast cancer, con la regularizacion se pueden determinar valores no necesarios del dataset
brsC = datasets.load_breast_cancer()
X_brsC  = brsC['data']
y_brsC  = brsC['target']
nombres = brsC['feature_names']
sgd = SGDClassifier(loss='log_loss', penalty='l1', alpha=0.01, max_iter=1000, tol=0.001)
puntajes = []

for i in range(X_brsC.shape[1]):
    punt = cross_val_score(sgd, X_brsC[:, i:i+1], y_brsC, scoring='r2', cv=ShuffleSplit(n_splits=len(X_brsC), test_size=3, train_size=0.3))
    puntajes.append( (round(np.mean(punt), 3), nombres[i]) )

caracteristicas = sorted(puntajes, reverse=True)
print('Caracteristicas del breast_cancer dataset con regularizacion: \n',caracteristicas)


### Grupos de entrenamiento y prueba de ambos datasets
X_ent_iris, X_pru_iris, y_ent_iris, y_pru_iris = train_test_split(Xiris,yiris,random_state=1)
X_ent_brsc, X_pru_brsc, y_ent_brsc, y_pru_brsc = train_test_split(X_brsC,y_brsC,random_state=1)


### Perceptron con iris dataset
clf_iris = Perceptron(random_state=1, alpha=0.01, max_iter=250)
clf_iris.fit(X_ent_iris,y_ent_iris)

### Perceptron con brs dataset
clf_brst = Perceptron(random_state=1, alpha=0.01, max_iter=250)
clf_brst.fit(X_ent_brsc,y_ent_brsc)

### Clasificador SGDC para iris dataset
sgd_iris = SGDClassifier(loss='log_loss', penalty='l1', alpha=0.01, max_iter=250, tol=0.001)
sgd_iris.fit(X_ent_iris, y_ent_iris)

### Clasificador SGDC para brsc dataset
sgd_brsc = SGDClassifier(loss='log_loss', penalty='l1', alpha=0.01, max_iter=250, tol=0.001)
sgd_brsc.fit(X_ent_brsc, y_ent_brsc)

#### Calculando el TPR o recall manualito junto con el FNR
FP_iris = metrics.confusion_matrix(y_true=y_pru_iris, y_pred=clf_iris.predict(X_pru_iris)).sum(axis=0) - np.diag(metrics.confusion_matrix(y_true=y_pru_iris, y_pred=clf_iris.predict(X_pru_iris)))
FN_iris = metrics.confusion_matrix(y_true=y_pru_iris, y_pred=clf_iris.predict(X_pru_iris)).sum(axis=1) - np.diag(metrics.confusion_matrix(y_true=y_pru_iris, y_pred=clf_iris.predict(X_pru_iris)))
TP_iris = np.diag(metrics.confusion_matrix(y_true=y_pru_iris, y_pred=clf_iris.predict(X_pru_iris)))
TPR_iris = TP_iris / (TP_iris + FN_iris)

print(f'\nMatriz de confusion del Perceptron para el dataset iris:\n{metrics.confusion_matrix(y_true=y_pru_iris, y_pred=clf_iris.predict(X_pru_iris))}')
print(f'La exactitud del perceptron para el dataset de iris es: {metrics.accuracy_score(y_true=y_pru_iris,y_pred=clf_iris.predict(X_pru_iris)):.4f}')
print(f'La precision del perceptron para el dataset de iris es: {metrics.precision_score(y_true=y_pru_iris,y_pred=clf_iris.predict(X_pru_iris)):.4f}')
print(f'La tasa de verdadero positivo del perceptron para el dataset de iris es: {metrics.recall_score(y_true=y_pru_iris,y_pred=clf_iris.predict(X_pru_iris)):.4f}')
print(f'El TPR del perceptron para el dataset de iris calculado a manopla es: {TPR_iris}')
print(f'F1 (precisión y sensibilidad) del perceptron para el dataset de iris es: {metrics.f1_score(y_true=y_pru_iris,y_pred=clf_iris.predict(X_pru_iris)):.4f}')
print(f'Reporte de clasificacion de la matriz de confusion:\n {metrics.classification_report(y_true=y_pru_iris,y_pred=clf_iris.predict(X_pru_iris))}')

print(f'\nMatriz de confusion del Clasificador SGDC para el dataset iris:\n{metrics.confusion_matrix(y_true=y_pru_iris, y_pred=sgd_iris.predict(X_pru_iris))}')
print(f'La exactitud del SGDC para el dataset de iris es: {metrics.accuracy_score(y_true=y_pru_iris,y_pred=sgd_iris.predict(X_pru_iris)):.4f}')
print(f'La precision del SGDC para el dataset de iris es: {metrics.precision_score(y_true=y_pru_iris,y_pred=sgd_iris.predict(X_pru_iris)):.4f}')
print(f'La tasa de verdadero positivo del SGDC para el dataset de iris es: {metrics.recall_score(y_true=y_pru_iris,y_pred=sgd_iris.predict(X_pru_iris)):.4f}')
print(f'F1 (precisión y sensibilidad) del SGDC para el dataset de iris es: {metrics.f1_score(y_true=y_pru_iris,y_pred=sgd_iris.predict(X_pru_iris)):.4f}')
print(f'Reporte de clasificacion de la matriz de confusion:\n {metrics.classification_report(y_true=y_pru_iris,y_pred=sgd_iris.predict(X_pru_iris))}')

print(f'\nMatriz de confusion del Perceptron para el dataset brsc:\n{metrics.confusion_matrix(y_true=y_pru_brsc, y_pred=clf_brst.predict(X_pru_brsc))}')
print(f'La exactitud del perceptron para el dataset de brsc es: {metrics.accuracy_score(y_true=y_pru_brsc,y_pred=clf_brst.predict(X_pru_brsc)):.4f}')
print(f'La precision del perceptron para el dataset de brsc es: {metrics.precision_score(y_true=y_pru_brsc,y_pred=clf_brst.predict(X_pru_brsc)):.4f}')
print(f'La tasa de verdadero positivo del perceptron para el dataset de brsc es: {metrics.recall_score(y_true=y_pru_brsc,y_pred=clf_brst.predict(X_pru_brsc)):.4f}')
print(f'F1 (precisión y sensibilidad) del perceptron para el dataset de brsc es: {metrics.f1_score(y_true=y_pru_brsc,y_pred=clf_brst.predict(X_pru_brsc)):.4f}')
print(f'Reporte de clasificacion de la matriz de confusion:\n {metrics.classification_report(y_true=y_pru_brsc,y_pred=clf_brst.predict(X_pru_brsc))}')

print(f'\nMatriz de confusion del Clasificador SGDC para el dataset brsc:\n{metrics.confusion_matrix(y_true=y_pru_brsc, y_pred=sgd_brsc.predict(X_pru_brsc))}')
print(f'La exactitud del SGDC para el dataset de brsc es: {metrics.accuracy_score(y_true=y_pru_brsc,y_pred=sgd_brsc.predict(X_pru_brsc)):.4f}')
print(f'La precision del SGDC para el dataset de brsc es: {metrics.precision_score(y_true=y_pru_brsc,y_pred=sgd_brsc.predict(X_pru_brsc)):.4f}')
print(f'La tasa de verdadero positivo del SGDC para el dataset de brsc es: {metrics.recall_score(y_true=y_pru_brsc,y_pred=sgd_brsc.predict(X_pru_brsc)):.4f}')
print(f'F1 (precisión y sensibilidad) del SGDC para el dataset de brsc es: {metrics.f1_score(y_true=y_pru_brsc,y_pred=sgd_brsc.predict(X_pru_brsc)):.4f}')
print(f'Reporte de clasificacion de la matriz de confusion:\n {metrics.classification_report(y_true=y_pru_brsc,y_pred=sgd_brsc.predict(X_pru_brsc))}')

print('\nEsta es una prueba del Display para la confusion matrix: \n')
metrics.ConfusionMatrixDisplay.from_estimator( clf_iris, X_pru_iris, y_pru_iris, cmap=plt.cm.Blues )
plt.show()

### Curvas ROC (Receiver Operating Characteristics ) de los modelos.
#### Primero hallar la prediccion de probabilidad usando el conjunto de prueba
perceptron_probs_iris = clf_iris._predict_proba_lr(X_pru_iris)
pred_perceptron_iris = perceptron_probs_iris[:,1]
sgdc_probs_iris = sgd_iris.predict_proba(X_pru_iris)
pred_sgdc_iris = sgdc_probs_iris[:,1]
perceptron_probs_brsc = clf_brst._predict_proba_lr(X_pru_brsc)
pred_perceptron_brsc = perceptron_probs_brsc[:,1]
sgdc_probs_brsc = sgd_brsc.predict_proba(X_pru_brsc)
pred_sgdc_brsc = sgdc_probs_brsc[:,1]

#### Encontrar el FPR, TPR y Umbrales
fpr_iris_percep, tpr_iris_percep, perceptron_iris_threshold = roc_curve(y_pru_iris, pred_perceptron_iris)
fpr_iris_sgdc, tpr_iris_sgdc, sgdc_iris_threshold = roc_curve(y_pru_iris, pred_sgdc_iris)
fpr_brsc_percep, tpr_brsc_percep, perceptron_brsc_threshold = roc_curve(y_pru_brsc, pred_perceptron_brsc)
fpr_brsc_sgdc, tpr_brsc_sgdc, perceptron_iris_threshold = roc_curve(y_pru_brsc, pred_sgdc_brsc)

#### Area bajo la curva
roc_auc_per_iris = auc(fpr_iris_percep, tpr_iris_percep)
roc_auc_sgdc_iris = auc(fpr_iris_sgdc, tpr_iris_sgdc)
roc_auc_per_brsc = auc(fpr_brsc_percep, tpr_brsc_percep)
roc_auc_sgdc_brsc = auc(fpr_brsc_sgdc, tpr_brsc_sgdc)

#### ROC del perceptron e iris
plt.plot(fpr_iris_percep, tpr_iris_percep, 'b', label=f'AUC = {roc_auc_per_iris:.4f}')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('Tasa Verdadero Positivo (TPR)')
plt.xlabel('Tasa Falso Positivo (FPR)')
plt.title('Curva ROC Perceptron Iris')
plt.legend(loc='lower right')
plt.show()

#### ROC del sgdc e iris
plt.plot(fpr_iris_sgdc, tpr_iris_sgdc, 'b', label=f'AUC = {roc_auc_sgdc_iris:.4f}')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('Tasa Verdadero Positivo (TPR)')
plt.xlabel('Tasa Falso Positivo (FPR)')
plt.title('Curva ROC SGDC Iris')
plt.legend(loc='lower right')
plt.show()

#### ROC del perceptron y brsc
plt.plot(fpr_brsc_percep, tpr_brsc_percep, 'b', label=f'AUC = {roc_auc_per_brsc:.4f}')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('Tasa Verdadero Positivo (TPR)')
plt.xlabel('Tasa Falso Positivo (FPR)')
plt.title('Curva ROC Perceptron BRSC')
plt.legend(loc='lower right')
plt.show()

#### ROC del sgdc y el brsc
plt.plot(fpr_brsc_sgdc, tpr_brsc_sgdc, 'b', label=f'AUC = {roc_auc_sgdc_brsc:.4f}')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('Tasa Verdadero Positivo (TPR)')
plt.xlabel('Tasa Falso Positivo (FPR)')
plt.title('Curva ROC SGDC for BRSC')
plt.legend(loc='lower right')
plt.show()