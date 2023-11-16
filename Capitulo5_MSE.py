##==============================================================================
## MSE approach
## Book: Aprendizaje Automatico y Profundo en Python, Capitulo 5
## Description: Regression models, solution with MSE 
## Author: Carlos M. Pineda Pertuz
##==============================================================================
## Programmer: Alberto Ramirez Bello
## Date: October 18th, 2023
##==============================================================================

import time
inicio = time.time()

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

df    = pd.read_csv("C:\\Users\\alber\\Documents\\Progra\\Deep_Learning\\aprendizaje_automatico_y_profundo\\fuentes\\casas\\precios_casas.csv")
X     = df.iloc[0:10,3].values
y     = df.iloc[0:10,1].values
mms_x = MinMaxScaler()
mms_y = MinMaxScaler()
X_std = mms_x.fit_transform( X.reshape(-1,1) )
y_std = mms_y.fit_transform( y.reshape(-1,1) )
y_p1  = X_std - 2
y_p2  = 2*X_std+5
y_p3  = 0.1*X_std + 0.05

fig,axs = plt.subplots(1, 3, figsize=(15,4) )

axs[0].scatter(X_std, y_std, color='orange')
axs[0].plot(X_std, y_p1, color='black')
axs[0].set_title('y = X - 2')

axs[1].scatter(X_std, y_std, color='gray')
axs[1].plot(X_std, y_p2, color='black')
axs[1].set_title('y = 2*X + 5')

axs[2].scatter(X_std, y_std, color='green')
axs[2].plot(X_std, y_p3, color='black')
axs[2].set_title('y = 0.1*X + 0.05')

fin = time.time()
ejecucion = (fin - inicio)
print(f'Tiempo total de ejecucion del programa antes de mostrar las graficas: {ejecucion}')
plt.show()

### En la primera ejecucion exitosa -> Tiempo total de ejecucion del programa antes de mostrar las graficas: 2.2080483436584473