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

df = pd.read_csv("C:\\Users\\alber\\Documents\\Progra\\Deep_Learning\\aprendizaje_automatico_y_profundo\\fuentes\\casas\\precios_casas.csv")
df1 = pd.read_csv("C:\\Users\\alber\\Documents\\Progra\\Deep_Learning\\aprendizaje_automatico_y_profundo\\fuentes\\emisiones2.csv")

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
# se obtienen las primeras 10 instancias
##X = df[['sqft_living']].values[:10]
##y = df['price'].values[:10]
# se estandarizan las variables
X_esc = sc_x.fit_transform(X)
y_esc = sc_y.fit_transform(y[:, np.newaxis]).flatten()
reg.fit(X_esc, y_esc)
print('w0: {:.3f}'.format(reg.intercept_))
print('w1: {:.3f}'.format(reg.coef_[0]))

# Graficas
#plt.plot(rig.coef_, 'r-', label='Coeficientes Ridge')
#plt.plot(reg.coef_, 'b*', label='Coeficientes OLS')
#plt.xlabel('Coeficiente')
#plt.show()

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
'''
si no suelto tu pasado, es porque.. me das la impresión de que tu tampoco lo has hecho... no importa lo mucho que te repitas que tu no fuiste hecha para vivir en el pasado y la fregada no es cierto... son solo palabras.. toda tu actitud en este ultimo año me lo ha dejado claro, y hay dos cosas que no ves, 

1. - no tienes idea lo mucho que me duele que sea asi
2. - al fin pude entender la situación, y creeme no te culpo

tu y aquel vivieron demasiadas cosas juntos, llamale como quieras, pero estuvieron casados 8 años, le diste un hijo hermoso,
sigues viviendo con su familia, ese vato era el hombre de tu vida, no manches hasta hace unos meses me dolia entrar a tu face y ver sus pinches fotos esas de estado
que no la quitas por ser macha segun.. ahorita ya no duele, apenas dejo de doler
todo lo que vivieron juntos, tu y yo apenas llevamos 1 año  y a distancia, es normal que pongas tantos muros, pero a veces siento que ya no puedo son demasiados muros 
a veces son muy altos 
recuerdas la ultima vez que me mencionaste a elenita, que me enoje? yo estaba en el parque, queria decirte el porque me enoje de frente pero bueno quien sabe cuando sea 
era algo sobre la popis, que ella te dijo que porque la justificabas, y tu le dijiste que no lo hacias, y yo te dije.. por que tu eres de los que piensan que es bueno tener a tus amigos cerca pero tener a tus enemigos mas cerca
y me dijiste que si... bueno, cuando me dijiste.. es que elenita me dice que la justifico.. inmediatamente dije.. elenita tiene razon, es lo que haces
siempre lo haces, siempre justificas a la gente, mas a las personas que te tratan mal.. por que? no estoy muy seguro, solo tengo hipotesis
pero dije... esa cabrona de elenita te conoce bien, te lee, y si no te ayuda en algunas cosas es por que no quiere...no le veo otra razon
por eso justificas al negro, apesar de que fue un completo hijo de puta contigo desde el dia 1
siempre terminas dandole gusto a oscar... por X o Y razon ...
bueno.. siempre le das gusto a elenita.. aunque eso te provoque un dolor de estomago! no manches, siempre hay una razon valida para darle gusto
y el alberto? ahhh ese wey puede esperar... siempre puede esperar.. dice que me quiere no? yo se lo deje claro desde el comienzo y el dijo que si.. entonces.. que se espere

por eso te dije que me sentia como tu amante.. porque... a pesar de que en algunas circunstancias nos favorecia que tu hablaras de mi para poder vernos... decidiste no hacerlo.. porque?
por que los amantes no se presumen, son algo vergonzoso.. que debe permanecer oculto... sabes que es lo peor? que a los amantes los desean, le traen ganas, los quieren ver
y yo ni eso provoco en ti

si te preguntas porque quiero seguir contigo... es porque he decidido creer que todo esto que te digo es circunstancial, es por la depresion en la que estas, 
por todos esos problemas que tienes ahorita , por la situacion tan complicada, quiero creer que esto es momentaneo, que realmente me quieres, pero las circunstancias no nos favorecen
pero que si podremos salir juntos adelante, quiero tener fe 
obviamente todo llegara a un punto de no retorno, pero ese momento no llega todavia 

Data Scientist, Model dev background - Python3, Numpy/Pandas, Spark, SQL, SAS, matplot, scikit - m.ammu@tcs.com
'''