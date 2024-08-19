##==================================================================================
## Modelos de clasificacion II
## Book: Aprendizaje Automatico y Profundo en Python, Capitulo 8
## Description: Sistema de recomendación basado en contenido, utilizando el 
## coeficiente de Pearson
## Author: Carlos M. Pineda Pertuz
##==================================================================================
## Programmer: Alberto Ramirez Bello
## Date: August 18th, 2024
##==================================================================================

import numpy as np
import json
import pandas as pd

datos = {
    "Guerra de las Galaxias": {"Calificacion":4.8, "Año":2015},
    "Rapido y Furioso": {"Calificacion":4.5, "Año":2013},
    "Toy Story": {"Calificacion":4, "Año":1995}
}   

def coef_pearson(X,pelis):
    '''
    Esta funcion calcula el coeficiente de correlacion de Pearson, 
    númerador y denominador se calculan de manera separada para un facil manejo.
    '''
    n  = len(pelis)
    p1 = pelis[0]
    p2 = pelis[1]

    for j in X[p1]:
        datos[j] = ( X[p1][j], X[p2][j] )

    sum_x = np.sum(datos['Calificacion'])
    sum_y = np.sum(datos['Año'])
    sum_x_cuadrada = np.sum( np.square(datos['Calificacion']) )
    sum_y_cuadrada = np.sum( np.square(datos['Año']) )
    suma_prod = np.dot(datos["Calificacion"], datos["Año"])
    sxy = n*suma_prod - ( sum_x * sum_y )
    sxx = n*sum_x_cuadrada - np.square(sum_x)
    syy = n*sum_y_cuadrada - np.square(sum_y)

    if sxx * syy == 0:
        return 0
    
    return sxy/np.sqrt(sxx*syy)

pelis = np.array( ['Guerra de las Galaxias', 'Rapido y Furioso'] )
r =  coef_pearson(datos,pelis)
print(f"\nCoeficiente de correlacion de la funcion {r:.5f}")

## Comprobando calculos con numpy
c = np.corrcoef([4.8, 2015], [4.8,2013]) [0,1]
print(f"Coeficiente de correlacion con numpy {c:.5f}\n")

## Continuando con el ejercicio
columnas = ['movie_id', 'title']
columnas2 = ['user_id', 'movie_id', 'rating']
peliculas2 = pd.read_csv("../peliculas.csv",
                         names = columnas, usecols = range(2))

puntajes = pd.read_csv("../puntajes.csv",
                         names = columnas2, usecols = range(3))

print(peliculas2)
print('\nIniciando el procedo de remover fechas del titulo....')

## Copiando el año de los titulos en una columna nueva
peliculas2['year'] = peliculas2.title.str.extract( r'(\(\d\d\d\d\))', expand=False)

## Removiendo los parentesis de la nueva columna
peliculas2['year'] = peliculas2.title.str.extract( '(\d\d\d\d)', expand = False)

## Removiendo los años de la columna titulo
peliculas2['title'] = peliculas2['title'].str.replace( '(\(\d\d\d\d\))', '', regex=True)
peliculas2['title'] = peliculas2['title'].apply(lambda x:x.strip()) # -> remueve espacios extra
print('\n')
print(peliculas2.head())
print(f'\nContenido de puntajes: \n {puntajes.sample(7)}')

## Mezclando ambos dataframes con la funcion merge()
puntajes = pd.merge(peliculas2, puntajes)
print(f'\nContenido de puntajes con peliculas: \n {puntajes.sample(7)}')

## Tabla pivot para ver patrones
puntajePeliculas = puntajes.pivot_table( index=['user_id'], columns=['title'], values='rating')
print(f'\nMuestra de la pivot table: \n {puntajePeliculas.head()}')

## Obteniendo detalles para una pelicula especifica
puntajeToyStory = puntajePeliculas['Toy Story']
print(f'\nEjemplo de detalle de pelicula: {puntajeToyStory.head()}')

## Obteniendo peliculas similares usando la funcion corrwith()
peliculasSimilares = puntajePeliculas.corrwith(puntajeToyStory)
peliculasSimilares = peliculasSimilares.dropna().sort_values( ascending=False )
print(f'\nEstan son las peliculas similares que se encontraron: \n{peliculasSimilares.head(7)}')
