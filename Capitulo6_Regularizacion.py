##==============================================================================
## Regularization, Metric Evaluation and Hyperparameters adjustment
## Book: Aprendizaje Automatico y Profundo en Python, Capitulo 6
## Description: Regularization, Metric Evaluation and Hyperparameters adjustment
## Author: Carlos M. Pineda Pertuz
##==============================================================================
## Programmer: Alberto Ramirez Bello
## Date: November 11th, 2023
##==============================================================================

from turtle import color
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

df = pd.read_csv("precios_casas.csv")
df1 = pd.read_csv("emisiones.csv")

## Regresion rigida
rig = Ridge(alpha=10)
X = df[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot']].values
y = df['price'].values

# Estandarizando las variables
sc_x = StandardScaler()
sc_y = StandardScaler()
X = sc_x.fit_transform(X)
y = sc_y.fit_transform(y[:, np.newaxis]).flatten()
X_ent, X_pru, y_ent, y_pru = train_test_split(X, y, test_size=0.3)
modelo_rig = rig.fit(X_ent, y_ent)
print('Coeficientes: ', modelo_rig.coef_)
print('R2= {:.3f}'.format(modelo_rig.score(X_pru, y_pru)))

reg = LinearRegression()
X_esc = sc_x.fit_transform(X)
y_esc = sc_y.fit_transform(y[:, np.newaxis]).flatten()
reg.fit(X_esc, y_esc)
print('w0: {:.3f}'.format(reg.intercept_))
print('w1: {:.3f}'.format(reg.coef_[0]))

# Graficas
plt.plot(rig.coef_, 'r-', label='Coeficientes Ridge')
plt.plot(reg.coef_, 'b*', label='Coeficientes OLS')
plt.xlabel('Coeficiente')
plt.show()

regCV = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 20.0, 50.0, 100.0])
modelo_rigCV = regCV.fit(X,y)
print('\nCoeficientes: ', modelo_rigCV.coef_)
print('Valor alfa: ', modelo_rigCV.alpha_)

## Regresion Lasso
lasso = Lasso(alpha=0.1)
modelo_lasso = lasso.fit(X_ent, y_ent)
print('\nCoeficientes lasso: ', modelo_lasso.coef_)
print('R2 Lasso: {:.3f}'.format(modelo_lasso.score(X_pru, y_pru)))

## Red elastica
modelo_Elastico = ElasticNet(alpha=0.1, l1_ratio=1)
modelo_Elastico = lasso.fit(X_ent, y_ent)
print('\nCoeficientes Elasticos: ', modelo_Elastico.coef_)
print('R2 Lasso: {:.3f}'.format(modelo_lasso.score(X_pru, y_pru)))

y_pred = reg.predict(X_pru)

## Error Absoluto Medio (MAE)
mae_pru = mean_absolute_error(y_pru, y_pred)
print('\nError Absoluto Medio MAE {:.3f}'.format(mae_pru))

## Error Cuadratico Medio
mse_pru = mean_squared_error(y_pru, y_pred)
print('Error Cuadratico Medio MSE {:.3f}'.format(mse_pru))

## Coeficiente de determinacion R^2
r2_pru = r2_score(y_pru, y_pred)
print('Coeficiente R2 = {:.3f}'.format(r2_pru))

## Cross validation K iterations
Xk = df[['sqft_living']].values 
yk = df[['price']].values
vals = cross_val_score(reg, Xk, yk, cv=5)
print(f'\nValidacion cruzada: {vals}\nMedia: {np.mean(vals)}\nDesviacion estandar: {np.std(vals)}')

## Curvas de validacion
plt.scatter(df1['anio'], df1['cantidad'], color='r')
plt.xlabel('año')
plt.xlim(1965,2010)
plt.ylabel('emision')
plt.show()

Xp_test = np.arange(1970, 2020, 1)
Xp = df1['anio'].values.reshape(-1,1)
yp = df1['cantidad'].values.reshape(-1,1)
plt.scatter(Xp, yp, color='black')

for grado in [1, 2, 3, 6, 7, 10, 15]:
    poli = PolynomialFeatures(degree = grado)
    Xp_p = poli.fit_transform(Xp)
    modelo_p = LinearRegression().fit(Xp_p,yp)
    to_predict = poli.fit_transform(Xp_test.reshape(-1,1))
    yp_pred = modelo_p.predict(to_predict).reshape(1,50)[0]
    plt.plot(Xp_test, yp_pred, label='Grado = {0}'.format(grado))

plt.legend(loc='best')
plt.xlabel('año')
plt.xlim(1965, 2010)
plt.ylabel('emision')
plt.show()

Xp_ent, Xp_pru, yp_ent, yp_pru = train_test_split(Xp, yp, test_size=0.1, random_state=1)
grados2 = [1,2,3,6,8,10,20]

# Creando un pipeline para encadenar el trabajo con PolyNomialFeatures y LinearRegression
pipe = Pipeline( [ ('scaler', StandardScaler() ), ( 'poly', PolynomialFeatures(degree=2) ), ('regl', LinearRegression() ) ] )
train_scores, test_scores = validation_curve(pipe, Xp_ent, yp_ent, param_name='poly__degree', param_range=grados2, cv=10)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis = 1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis = 1)

plt.plot(grados2, train_mean, color='blue', marker='o', markersize=5, label='Exactitud de Entrenamiento')
plt.fill_between(grados2, train_mean + train_std, train_mean - train_std, alpha = 0.15, color='blue')
plt.plot(grados2, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Exactitud de Validacion')
plt.fill_between(grados2,test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.legend(loc='best')
plt.ylim(0,1)
plt.xlabel('Grado')
plt.ylabel('Puntaje')
plt.show()


## Usando un polinomio de grado 8 y el algoritmo busqueda de cuadriculas para ver el hiperparametro optimo
plt.scatter(Xp_ent.ravel(), yp_ent)
lim = plt.axis()
pipe2 = Pipeline( [ ('scaler', StandardScaler() ), ( 'poly', PolynomialFeatures(degree=8) ), ('regl', LinearRegression() ) ] )
y_test = pipe2.fit(Xp_ent, yp_ent).predict(Xp_test.reshape(-1,1))
plt.plot(Xp_test.ravel(), y_test)
plt.axis(lim)
plt.xlabel('año')
plt.xlim(1965,2010)
plt.ylabel('emision')
plt.show()

## Usando learning_curve
train_sizes, train_scores2, test_scores2 = learning_curve(estimator=pipe2, X=Xp_ent, y=yp_ent, train_sizes=np.linspace(0.1, 1.0, 10), cv=10)
train_mean2 = np.mean(train_scores2, axis=1)
train_std2  = np.std(train_scores2, axis=1)
test_mean2  = np.mean(test_scores2, axis=1)
test_std2   = np.std(test_scores2, axis=1)
plt.plot(train_sizes, train_mean2, color='blue', marker='o', markersize=5, label='Exactitud de Entrenamiento 2') 
plt.fill_between( train_sizes, train_mean2 + train_std2, train_mean2 - train_std2, alpha=0.15, color='blue' )
plt.plot(train_sizes, test_mean2, color='green', linestyle='--', marker='s', markersize=5, label='Exactitud de Validacion 2')
plt.fill_between( train_sizes, test_mean2 + test_std2, test_mean2 - test_std2, alpha=0.15, color='green' )
plt.grid()
plt.xlabel('Numero de muestras de entrenamiento')
plt.ylabel('Exactitud')
plt.legend(loc='lower right')
plt.ylim([-0.5, 1.1])
plt.tight_layout()
plt.show()

## Usando GridSearchCV
model_grid = Pipeline( [ ('scaler', StandardScaler() ), ( 'poly', PolynomialFeatures(degree=2) ), ('regl', LinearRegression() ) ] )
params = {'poly__degree':np.arange(2,20)}
gscv = GridSearchCV(model_grid, params, cv=10, scoring='neg_mean_squared_error')
gscv.fit(Xp_ent.reshape(-1,1), yp_ent)
space=np.linspace(1970,2008,50).reshape(-1,1)
est_deg = gscv.best_params_['poly__degree']

plt.scatter(Xp_ent.ravel(), yp_ent)
plt.plot(space, gscv.predict(space), color='red')
plt.title(f'Grado estimado {est_deg}')
plt.xlabel('año')
plt.xlim(1965, 2010)
plt.ylabel('emision')
plt.show()

## Usando Randomized Search CV
rscv = RandomizedSearchCV(model_grid, params, cv=10)
rscv.fit(Xp_ent.reshape(-1,1), yp_ent)
est_deg2 = rscv.best_params_['poly__degree']

plt.scatter(Xp_ent.ravel(), yp_ent)
plt.plot(space, rscv.predict(space), color='blue')
plt.title(f'Grado estimado {est_deg2}')
plt.xlabel('año')
plt.xlim(1965, 2010)
plt.ylabel('emision')
plt.show()
