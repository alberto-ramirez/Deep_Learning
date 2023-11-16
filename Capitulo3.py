import pandas as pd
import numpy as np

df = pd.read_csv("C:\\Users\\alber\\Documents\\Progra\\Deep_Learning\\aprendizaje_automatico_y_profundo\\fuentes\\casas\\precios_casas.csv")

print('Mostrando el contenido del data frame:\n',df)
print('\nUsando la funcion head', df.head())
print('\nUsando la funcion tail', df.tail())
print('\nUsando la funcion columns\n', df.columns)
print('\nUsando la funcion shape', df.shape)
print('\nUsando la funcion sample', df.sample(7))
print('\nUsando iloc----------------------------')
print('--------> iloc[9]\n', df.iloc[9])
print('\n--------> iloc[:10]\n', df.iloc[:10])
print('\n--------> iloc[0,1]\n', df.iloc[0,1])
print('\n--------> iloc[[0, 1, 2], 0:5]\n', df.iloc[[0, 1, 2], 0:5])
print('\nUsando loc----------------------------')
print('--------> loc[0,price]\n', df.loc[0,'price'])
print('--------> loc[:3,price:bathrooms]\n', df.loc[:3,'price':'bathrooms'])
print('\nUsando el metodo describe\n', df.describe())
print('\nAgregando include al metodo describe\n', df.describe(include='all'))
print('\n========================= LIBRERIA NUMPY ===========================================================================\n')

vec1    = np.array([1,2,3,4,5]) ## Vector de 5 elementos
matriz  = np.array([ [1, 2, 3], [4, 5, 6], [7, 8, 9]])
vec2    = np.zeros(7) ## Vector de 7 ceros
matriz2 = np.zeros((3,3)) ## Matriz cuadrada de ceros
matriz3 = np.random.random((3,3)) ## Matriz cuadrada con numeros aleatorios
lista1  = [5, 6, 7, 8, 9 ,0]
vec3    = np.array(lista1)
vec4    = np.arange(7) ## Vector secuencial
vec5    = np.arange(1, 7, 0.2) ## Vector secuencial de 1 a 7 con incrementos de 0.2

print(f"El vector creado contiene: {vec1}\ny sus dimensiones son:{vec1.shape}")
print(f"\nEl vector creado con ceros contiene: {vec2}\ny sus dimensiones son:{vec2.shape}")
print(f"\nLa matriz creada contiene: \n{matriz}\ny sus dimensiones son:{matriz.shape}")
print(f"\nLa matriz2 creada contiene: \n{matriz2}\ny sus dimensiones son:{matriz2.shape}")
print(f"\nLa matriz3 creada contiene: \n{matriz3}\ny sus dimensiones son:{matriz3.shape}")
print(f"\nLa lista1 creada contiene: {lista1} y type es :{type(lista1)}\nconvirtiendo la lista a vector {vec3} y su type es: {type(vec3)}")
print(f"\nEl vector creado con arange contiene: {vec4}\ny su type es {type(vec4)}")
print(f"\nEl vector creado con arange contiene: {vec5}\ny su type es {type(vec5)}")