##==================================================================================
## Modelos de clasificacion II
## Book: Aprendizaje Automatico y Profundo en Python, Capitulo 8
## Description: Esta segunda tercera contiene un ejemplo del modelo naive bayes
##    K vecinos más cercanos (KNN)
##    Dataset: Spam  No Spam.
## Author: Carlos M. Pineda Pertuz
##==================================================================================
## Programmer: Alberto Ramirez Bello
## Date: January 29th, 2024
##==================================================================================

import numpy as np
import pandas as pd
import pydot
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer ## contar las palabras
from sklearn.naive_bayes import MultinomialNB

print('\nComenzando con el modelo Naive Bayes')
df = pd.read_csv('..\\fuentes\\spam_no_spam.csv')
df2 = pd.read_csv('..\\fuentes\\spam_no_spam.csv', header=0) 
print('Contenido del dataset:\n', df.head(7))
X = df2.iloc[:,0]
y = df2.iloc[:,1]

## Convertir los mensajes de texto a formato númerico utilizando la tecina 'Bag of words'
mensajes = ['Hola, ¿Cómo estás?', 'Gana dinero sin trabajar', 'Hola, Contáctame ahora']
cont_vect = CountVectorizer()
X_t = cont_vect.fit_transform(X)
#cont_vect.fit(mensajes)
#nombres = cont_vect.get_feature_names_out()
#matriz = cont_vect.transform(mensajes).toarray()
nb = MultinomialNB()
nb.fit(X_t,y)
#print('\n',nombres)
#print('\nMatriz de frecuencias de palabras mostradas arriba: \n',matriz)

## Realizando pruebas al modelo despues del entrenamiento
mensaje_prueba = np.array(['Hola amigo hay una reunión importante'])
datos_prueba = cont_vect.transform(mensaje_prueba)
prediccion = nb.predict(datos_prueba)
print('Este es el mensaje de prueba utilizado:\n', mensaje_prueba)
print('\nEs esto spam?: ',prediccion)