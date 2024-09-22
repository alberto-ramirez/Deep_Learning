##=======================================================================================
## Introducción al Procesamiento del lenguaje natural
## Book: Aprendizaje Automatico y Profundo en Python, Capitulo 14
## Description: Word2Vec.
## Author: Carlos M. Pineda Pertuz
##=======================================================================================
## Programmer: Alberto Ramirez Bello
## Date: September 22th, 2024
##=======================================================================================

import numpy as np
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import PunktTokenizer
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

## Datos de entrada.
data_in = "datos.txt"

## Funcion para extraer palabras.
def cargar_datos(text_file):
    lista = []
    oraciones = []

    with open(text_file, 'r') as f:

        for i, line in enumerate(f):
            lista.append(line)
        sentencias = list([word_tokenize(frase) for frase in lista])
        return sentencias

sentencias = cargar_datos(data_in)
print(f'\nDimensiones de los datos: {len(sentencias)}\n\nTitulo del primer proyecto: \n{sentencias[1]}\n\nTitulo del 7 proyecto: \n{sentencias[6]}')

## Definicion del modelo
modelo = Word2Vec(
    sentences=sentencias, vector_size=300, window=10, min_count=5,
    workers=8, sg=1, hs=1, negative=0
)

modelo.train(sentencias, total_examples=len(sentencias), epochs=20)
palabras_incrustadas = modelo.wv
lista_palabras = list(modelo.wv.key_to_index)

## Probando el sistema, similitud entre "sistemas" y "algoritmos"
def norma_vectorial(vector):
    return np.sqrt( np.sum( [v**2 for v in vector] ) )

def similitud_coseno(v1, v2):
    return np.dot(v1, v2) / float( norma_vectorial(v1)*norma_vectorial(v2) )

print(f'Similitud = {similitud_coseno( palabras_incrustadas[lista_palabras.index("sistema")], palabras_incrustadas[lista_palabras.index("algoritmos")] ):.3f}')

## Usando el metodo similarity de gensim
s = modelo.wv.similarity(w1 = 'sistema', w2 = 'algoritmos')
print(f'\nSimilitud calculada con similarity = {s:.3f}')

## Metodo most_similar, 10 palabras relacionadas a una palabara principal, con mayor probabilidad
palabra = 'sistema'
print(f'\nLa palabra más similar a {palabra} es:\n\n{modelo.wv.most_similar(positive=palabra)}')
