##=======================================================================================
## Reduccion de la dimensionalidad
## Book: Aprendizaje Automatico y Profundo en Python, Capitulo 10
## Description: Análisis discriminante lineal (ADL).
## Author: Carlos M. Pineda Pertuz
##=======================================================================================
## Programmer: Alberto Ramirez Bello
## Date: September 2nd, 2024
##=======================================================================================

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

digitos = load_digits()
X,y = digitos.data, digitos.target
print(f'\n++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('Iniciando el modelo...')
print(f'Valores, \nX = {X}\n\ny = {y}')
plt.figure( figsize = [5,5] )
plt.subplot(121)
plt.imshow( digitos.images[5], plt.cm.gray_r )
plt.title(f"Numero: {digitos.target[5]}")
plt.subplot(122)
plt.imshow( digitos.images[10], plt.cm.gray_r)
plt.title(f"Numero: {digitos.target[10]}")
plt.show()

X_ent, X_pru, y_ent, y_pru = train_test_split( X,y, test_size = 0.2, random_state = 43 )
X_ent = X_ent / 255
X_pru = X_pru / 255
clf = SVC( C = 1, gamma = 1 )
clf.fit( X_ent, y_ent )
print(f'\nExactitud del conjunto de entrenamiento: {clf.score(X_ent, y_ent):.5f}')

y_pred = clf.predict(X_pru)
print(f'Exactitud del conjunto de prueba: {accuracy_score(y_pru, y_pred):.5f}')

lda = LDA( n_components=5 )
ldaX_ent = lda.fit_transform( X_ent, y_ent )
ldaX_pru = lda.transform(X_pru)

X_entr, X_prue, y_entr, y_prue = train_test_split( ldaX_ent, y_ent, test_size = 0.1, random_state = 1 )
clf = SVC( C = 1, gamma = 1 )
clf.fit( X_entr, y_entr )
print(f'\nExactitud del conjunto de entrenamiento ADL: {clf.score(X_entr, y_entr):.5f}')

y_predi = clf.predict(X_prue)
print(f'Exactitud del conjunto de prueba ADL: {accuracy_score(y_prue, y_predi):.5f}')