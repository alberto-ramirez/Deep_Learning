import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split ## Para dividir los datos
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

df = pd.DataFrame( [ ['1', 1, 30], ['2', 1, 32], ['3', 0] ] )
df.columns = ["Codigo", "Credito", "Edad"]
print('Data frame recien creado:\n',df)
df.replace('?',np.nan, inplace=True)
print('Mostrando el reemplazo del df\n', df)
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp = imp.fit(df.values)
imp_datos = imp.transform(df.values)
print('Datos arreglados:\n', imp_datos)

## Datos categoricos
df2=pd.DataFrame( [ ['M', 30, 'Amarillo', 'Clase 1'], ['P', 28, 'Azul', 'Clase 2'], ['J', 21, 'Rojo', 'Clase 1'] ])
df2.columns=['Nombre', 'Edad', 'Color', 'Etiqueta']
print('\nRecientemente creado el df2\n', df2)
le_clase = LabelEncoder()
y = le_clase.fit_transform(df2.Etiqueta)
print('\nLe_etiquetes de le_clase.. uhlala\n', y)
clase_inv = le_clase.inverse_transform(y)
print('\ndevolviendo valores con inverse_transform\n', clase_inv)

## One hot encoder
le_color = LabelEncoder()
ohe_color = OneHotEncoder(categories='auto')
df2['color_cod'] = le_color.fit_transform(df2.Color)
print('\nDataFrame 2 con one hot encoder\n', df2)

'''Transformando la caracteristica color, con el objeto de one hot encoder'''
datos_ohe = ohe_color.fit_transform(df2.color_cod.values.reshape(-1,1)).toarray()
dfOneHot = pd.DataFrame(datos_ohe, columns=[ 'Color_'+str(int(i)) for i in range(len(df2.Color)) ] )
df3 = pd.concat( [df2,dfOneHot], axis=1 )
print('\nMostrando data frame 3 el concatenado\n', df3)

'''eliminando las columnas color y color_cod que no son necesarias... '''
df3.drop( ['Color', 'color_cod'], axis=1 )
print('\nDataframe3 sin columnas innecesarias\n', df3)
print('\nUsando el metodo dummies de pandas\n', pd.get_dummies(data=df2, columns=['Color'], drop_first=True) )

## Escalamiento de caracteristicas
df4 = pd.read_csv("precios_casas.csv")
esc = MinMaxScaler()
est = StandardScaler()
X_ent = esc.fit_transform( df4.price.values.reshape(-1,1) )
X_est = est.fit_transform( df4.price.values.reshape(-1,1) )
print('\nMostrando valores normalizados:\n', X_ent)
print('\nMostrando valores estandarizados:\n', X_est)
