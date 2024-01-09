##==============================================================================
## Modelos de clasificacion I
## Book: Aprendizaje Automatico y Profundo en Python, Capitulo 7
## Description: Esta segunda parte contiene la Regresion logistica comun
## Author: Carlos M. Pineda Pertuz
##==============================================================================
## Programmer: Alberto Ramirez Bello
## Date: January 5th, 2024
##==============================================================================

import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn import metrics

### Funciones para la regresion logistica
def sigmoide(x):
    """Funcion sigmoide"""
    return 1./(1.+np.exp(-x))

def calcular_prediccion(x,pesos):
    """Esta funcion calcula el valor para la variable 'y' basado en los pesos"""
    z = x.dot(pesos[1:]) + pesos[0]
    prediccion = sigmoide(z)
    return prediccion

def calcular_costo(X,y,pesos):
    """Esta funcion permite calcular los valores de J"""
    prediccion=calcular_prediccion(X,pesos)
    costo=np.mean(-y*np.log(prediccion) - (1-y)*np.log(1-prediccion) )
    return costo

def entrenar(X,y,pesos,max_iter,tasa_aprendizaje):
    """Utilizando el gradiente descendiente"""
    pesos = rgen.normal(loc=0.0,scale=0.01,size=1+X.shape[1])

    for i in range(max_iter):
        prediccion=calcular_prediccion(X,pesos)
        error=y-prediccion
        pesos[1:] += tasa_aprendizaje*X.T.dot(error)
        pesos[0]  += tasa_aprendizaje*error.sum()

        ### Imprimir el costo cada 10 iteraciones...
        if ( i % 10 ) == 0:
            print(f"Costo: {calcular_costo(X,y,pesos)}")

    return pesos
### Fin de las funciones para la regresion logistica

### Funciones para la regresion logistica con scikit-learn
def limite_decision(clf,X,subplot):
    min1, max1 = X[:,0].min() - 1, X[:,0].max() + 1
    min2, max2 = X[:,0].min() - 1, X[:,0].max() + 1
    x1grid = np.arange(min1, max1, 0.1)
    x2grid = np.arange(min2, max2, 0.1)
    xx, yy = np.meshgrid(x1grid,x2grid)
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
    grid = np.hstack((r1,r2))
    yhat = clf.predict(grid)
    zz = yhat.reshape(xx.shape)
    subplot.contourf(xx, yy, zz, cmap=cmap_light)
### Fin de las funciones para la regresion logistica con scikit-learn
    
### Funciones para la regresion logistica con gradiente descendiente estocastico
def entrenar_estoc(X,y,pesos,max_iter,tasa_aprendizaje):
    """Esta funcion aplica el descenso de gradiente estocastico"""
    for i in range(max_iter):
        error = 0.0

        for xi, yi in zip(X,y):
            prediccion =  calcular_prediccion(xi,pesos)
            error = yi - prediccion
            pesos[1:] += tasa_aprendizaje*np.dot(xi.T,error)
            pesos[0]  += tasa_aprendizaje*error.sum()
            ### Imprimiendo el costo cada 10 iteraciones

        if ( i%10 == 0 ):
            print(f"Costo estocastico: {calcular_costo(X,y,pesos):.4f} ")

    return pesos            
### Fin de las funciones para la regresion logistica con gradiente descendiente estocastico  

dataset=datasets.load_breast_cancer()
print("\nIniciando la regresion logistica...\n")

### Graficando las variables mean radius y mean texture
X = dataset.data
y = dataset.target
X = X[:, [1,2]]
y = y[:]
max_iter=200
tasa_aprendizaje=0.01

### Estandarizando datos
X[:,0] = ( X[:,0] - X[:,0].mean() ) / X[:,0].std()
X[:,1] = ( X[:,1] - X[:,1].mean() ) / X[:,1].std()

BENIGNO = 0
MALIGNO = 1

for target,color in zip( range(2),['r','b'] ):
    plt.scatter( X[y==target, MALIGNO], X[y==target, BENIGNO], color=color, label=dataset.target_names[target] )
plt.xlabel( dataset.feature_names[MALIGNO],fontsize=15 )
plt.ylabel( dataset.feature_names[BENIGNO],fontsize=15 )
plt.legend( prop={'size':10} )
plt.show()

### Inicializando los pesos
rgen = np.random.RandomState(1)
pesos = []
pesos = entrenar(X,y,pesos,max_iter,tasa_aprendizaje)
pesos_estoc = entrenar_estoc(X,y,pesos,max_iter,tasa_aprendizaje)
print("\nPesos de la funcion: ",pesos)
prueba  = np.array([1,1])
prueba2 = np.array([[1,1]])
prueba3 = np.array([0.5,0.5])
prueba4 = np.array([[0.5,0.5]])
X_ent, X_pru, y_ent, y_pru = train_test_split(X,y,random_state=1)
rloc = LogisticRegression()
rloc.fit(X_ent,y_ent)
clf_Est = SGDClassifier(loss='log_loss')
clf_Est.fit(X_ent,y_ent)
prediccion = calcular_prediccion(prueba,pesos)
prediccion_scl = rloc.predict(prueba2)
prediccion_estoc = calcular_prediccion(prueba3,pesos_estoc)
prediccion_scl_est = clf_Est.predict(prueba4)
print(f'Prediccion del modelo: {prediccion}')

### Graficando los datos de entrenamiento y la muestra de prueba
for target, color in zip( range(2), ['r','b']):
    plt.scatter( X[y==target, MALIGNO], X[y==target, BENIGNO], color=color, label=dataset.target_names[target] )

plt.xlabel(dataset.feature_names[MALIGNO], fontsize=15)
plt.ylabel(dataset.feature_names[BENIGNO], fontsize=15)
plt.legend(loc='upper left')

if ( prediccion >= 0.5 ):
    c = 'green'
else:
    c = 'black'

plt.scatter( prueba[0], prueba[1], marker='*', color=c, s=150 )
plt.show()

print("\nIniciando la regresion logistica con scikit learn")
print('Exactitud del conjunto de entrenamiento: ',rloc.score(X_ent,y_ent) )
print('Exactitud del conjunto de pruebas: ',rloc.score(X_ent,y_ent) )
colores = ['#FFFFAA','#EFEFEF']
cmap_light = ListedColormap(colores[0:2])

# Grafico con el limite de decision
min_x = int(min(X_ent[:,0]))
max_x = int(max(X_ent[:,0]))
xx = range(min_x,max_x)
yy = -( xx*rloc.coef_[0][0] + rloc.intercept_[0] ) / rloc.coef_[0][1]
plt.plot(xx,yy,color='black')

for class_value in range(2):
    row_ix = np.where(y==class_value)
    plt.scatter(X[row_ix, 0],X[row_ix, 1])

plt.xlabel('Mean texture')
plt.ylabel('Mean radius')
plt.show()

print("Prediccion de la prueba2: ",prediccion_scl)
print("El valor de predict_proba: ", rloc.predict_proba(prueba2) )

### Creando modelos de regresion logistica con diferentes valores de C
fig, subaxes = plt.subplots( 1,3, figsize=(15,4) )
etiquetas = 2

for c,subplot in zip([0.01, 0.1, 10], subaxes):
    clf = LogisticRegression( C=c, solver='liblinear' ).fit( X_ent, y_ent )
    p_ent = clf.score(X_ent, y_ent)
    p_pru = clf.score(X_pru, y_pru)
    limite_decision(clf, X_ent, subplot)

    for y in range(etiquetas):
        title = f'Regresion logistica (Mean Texture vs Mean Radius), C={c}'
        subplot.scatter(X_ent[y_ent==y,0], X_ent[y_ent==y,1],label='class' + format(y) )
        subplot.set_xlabel('Mean texture', fontsize=15)
        subplot.set_ylabel('Mean radius', fontsize=15)
        subplot.set_title('C=' + str(c) + f' punto.ent: {p_ent:.4f}' + f' punto.prueba: {p_pru:.4f}')
        subplot.legend()
plt.show()

parametros = [ {'C':[0.5, 1.0, 1.5, 1.8, 2.0, 2.5],
                'solver':['newton-cg', 'lbfgs', 'sag', 'saga']} ]

gs = GridSearchCV( estimator=LogisticRegression(multi_class='auto'),
                   param_grid=parametros, scoring='accuracy', cv=10, n_jobs=-1 )
gs.fit(X_ent, y_ent)
print(f"\nExactitud con el cross val: {cross_val_score(gs.best_estimator_, X_ent, y_ent, scoring='accuracy', cv=10).mean():.5f}")
clf2 = LogisticRegression(C=1.0, solver='newton-cg').fit(X_ent, y_ent)
punto_pru = clf2.score(X_pru, y_pru)
print(f'Exactitud {punto_pru:.4f}')
print(f'Prediccion Estocastica: {prediccion_estoc:.4f}')
print(f'Prediccion Estocastica con scikit-learn: {prediccion_scl_est}')
print(f'Probabilidades Estocasticas con scikit-learn: {clf_Est.predict_proba(prueba4)}')

### Graficando el modelo estocastico
min_xEst = int(min(X_ent[:,0]))
max_xEst = int(max(X_ent[:,0]))
xxx = range(min_xEst,max_xEst)
yyy = -(xxx*pesos_estoc[1] + pesos_estoc[0]) / pesos_estoc[2]

plt.figure(figsize=(10,7))
plt.plot(xxx, yyy, color='black')
plt.scatter(X_ent[:,0], X_ent[:,1], c=y_ent.ravel(), alpha=1)
plt.scatter(prueba3[0], prueba3[1], marker='*', color='blue', edgecolors='black', s=250)
plt.xlabel('mean radius')
plt.ylabel('mean texture')
plt.title('Logistic Regression')
plt.show()

print(f'\nMatriz de confusion del modelo Logistic Regression con los valores y_pru y X_pru:\n{metrics.confusion_matrix(y_true=y_pru, y_pred=rloc.predict(X_pru))}')
print(f'\nMatriz de confusion del Clasificador SGDC con los valores y_pru y X_pru:\n{metrics.confusion_matrix(y_true=y_pru, y_pred=clf_Est.predict(X_pru))}')
