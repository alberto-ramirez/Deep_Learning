##==================================================================================
## Modelos de clasificacion II
## Book: Aprendizaje Automatico y Profundo en Python, Capitulo 8
## Description: Sistema de recomendación basado en filtro colaborativo, utilizando el 
## coeficiente de Pearson
## Author: Carlos M. Pineda Pertuz
##==================================================================================
## Programmer: Alberto Ramirez Bello
## Date: August 18th, 2024
##==================================================================================

import numpy as np
import pandas as pd
import json

columnas = ['movie_id', 'title']
columnas2 = ['user_id', 'movie_id', 'rating']

peliculas = pd.read_csv("../peliculas.csv",
                         names = columnas, usecols = range(2))

puntajes = pd.read_csv("../puntajes.csv",
                         names = columnas2, usecols = range(3))

def obtener_puntaje(id_usu, id_peli):
    ''' Funcion para extraer valores especificos de un dataframe bajo ciertas condiciones '''
    return (puntajes.loc[ (puntajes.user_id == id_usu) & (puntajes.movie_id == id_peli), 'rating' ].iloc[0])

def obtener_idPelis(userid):
    ''' Funcion para obtener ids especificos de peliculas y convertirlos a una lista '''
    pnt = (puntajes.loc[ (puntajes.user_id == userid ), 'movie_id'].tolist())
    return pnt

def obtener_titPeli(movieid):
    ''' Funcion para obtener los ids especificos de una pelicula '''
    pnt = (peliculas.loc[ (peliculas.movie_id == movieid), 'title'].iloc[0])
    return pnt

def correlacion_pearson(usuario1, usuario2):
    ''' Funcion para calcular el coeficiente de correlacion de Pearson entre 2 usuarios '''
    cont = []
    
    # Peliculas vistas por ambos usuarios.
    for e in puntajes.loc[puntajes.user_id == usuario1, 'movie_id'].tolist():

        if e in puntajes.loc[puntajes.user_id == usuario2, 'movie_id'].tolist():
            cont.append(e)

        if len(cont) == 0:
            return 0

        # Calculando covarianzas.
        sumPuntajeUsuario1  = sum( [obtener_puntaje(usuario1,e) for e in cont] )
        sumPuntajeUsuario2  = sum( [obtener_puntaje(usuario2,e) for e in cont] )
        cuadPuntajeUsuario1 = sum( [pow(obtener_puntaje(usuario1, e),2) for e in cont])
        cuadPuntajeUsuario2 = sum( [pow(obtener_puntaje(usuario2, e),2) for e in cont])
        prodSumPuntajes     = sum( [obtener_puntaje(usuario1, e) * obtener_puntaje(usuario2,e) for e in cont])
        n                   = len(cont)

        # Correlacion Pearson.
        numerador   = n * prodSumPuntajes - ( sumPuntajeUsuario1 * sumPuntajeUsuario2 )
        denominador = np.sqrt(n*cuadPuntajeUsuario1 - pow(sumPuntajeUsuario1,2)) * np.sqrt(n*cuadPuntajeUsuario2 - pow(sumPuntajeUsuario2,2) )

        if denominador == 0:
            return 0
        
        return numerador/denominador

## Probando la funcion con usuarios 23 y 34
print(f'\nCorrelacion de Pearson para los usuarios 23 y 34: {correlacion_pearson(23,35)}')

def obtener_recomendaciones(idUsuario):

    ids_usuarios = puntajes.user_id.unique().tolist()

    total = {}
    suma_sim = {}

    # iteracion a traves de un subconjunto de ids de usuarios
    for usuario in ids_usuarios[:100]:
    
        # No se tiene en cuenta el mismo usuario.
        if usuario == idUsuario:
            continue

        # Obteniendo similitud entre usuarios.
        coef = correlacion_pearson(idUsuario, usuario)

        #if coef <= 0:
            #continue

        # obteniendo calificacion de similitud ponderada y suma de las similitudes entre ambos usuarios.
        for idPeli in obtener_idPelis(usuario):
            
            # Se tienen en cuenta solo las peliculas no vistas o calificadas
            if idPeli not in obtener_idPelis(idUsuario) or obtener_puntaje(idUsuario, idPeli) == 0:
                total[idPeli] = 0
                total[idPeli] += obtener_puntaje(usuario,idPeli) * coef
                suma_sim[idPeli] = 0
                suma_sim[idPeli] += coef

        # Normalizar los puntajes
        clasif = [ (tot/suma_sim[idPeli], idPeli) for idPeli, tot in total.items() ]
        clasif.sort()
        clasif.reverse()

        # Se obtienen los titulos de las peliculas.
        recomendaciones = [obtener_titPeli(idPeli) for coef, idPeli in clasif]
        return recomendaciones[:5]
    
print(obtener_recomendaciones(2921))    
