##=======================================================================================
## Reduccion de la dimensionalidad
## Book: Aprendizaje Automatico y Profundo en Python, Capitulo 13
## Description: Aumento de datos y transferencia de aprendizaje - Generador de datos
## de imagenes.
## Author: Carlos M. Pineda Pertuz
##=======================================================================================
## Programmer: Alberto Ramirez Bello
## Date: September 8th, 2024
##=======================================================================================

import os
import numpy as np
import shutil
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

carpeta_principal = "..horse_human"

dir_ent = os.path.join(carpeta_principal,'entrenamiento')
os.mkdir(dir_ent)

dir_ent_cab = os.path.join(dir_ent,'caballos')
os.mkdir(dir_ent_cab)

dir_ent_hum = os.path.join(dir_ent,'humanos')
os.mkdir(dir_ent_hum)

dir_val = os.path.join(carpeta_principal,'validacion')
os.mkdir(dir_val)

dir_val_cab = os.path.join(dir_val,'caballos')
os.mkdir(dir_val_cab)

dir_val_hum = os.path.join(dir_val,'humanos')
os.mkdir(dir_val_hum)

num_ima_cab = 500
num_ima_hum = 527
porc_separacion = 0.2

# Caballos
num_ima_pru_cab = np.round(porc_separacion * num_ima_cab)
num_ima_ent_cab = np.round(num_ima_cab - num_ima_pru_cab)

# Humanos
num_ima_pru_hum = np.round(porc_separacion * num_ima_hum)
num_ima_ent_hum = np.round(num_ima_hum - num_ima_pru_hum)

# img caballos
fnames = ['horse({}).png'.format(i+1) for i in range(int(num_ima_ent_cab))]

for fname in fnames:
    src = os.path.join(carpeta_principal+"/caballos",fname)
    dst = os.path.join(dir_ent_cab,fname)
    shutil.copyfile(src,dst)

fnames = ['horse ({}).png'.format(i+1) for i in range(int(num_ima_ent_cab), num_ima_cab)]

for fname in fnames:
    src = os.path.join(carpeta_principal+"/caballos", fname)
    dst = os.path.join(dir_val_cab, fname)
    shutil.copyfile(src,dst)

# img humanos
fnames = ['human({}).png'.format(i+1) for i in range(int(num_ima_ent_hum))]

for fname in fnames:
    src = os.path.join(carpeta_principal+"/humanos",fname)
    dst = os.path.join(dir_ent_cab,fname)
    shutil.copyfile(src,dst)

fnames = ['human ({}).png'.format(i+1) for i in range(int(num_ima_ent_hum), num_ima_hum)]

for fname in fnames:
    src = os.path.join(carpeta_principal+"/humanos", fname)
    dst = os.path.join(dir_val_hum, fname)
    shutil.copyfile(src,dst)

print('\nTotal de muestras de entrenamiento de caballos: ', len(os.listdir(dir_ent_cab)))
print('Total de muestras de entrenamiento de humanos: ', len(os.listdir(dir_ent_hum)))
print('Total de muestras de validacion de caballos: ', len(os.listdir(dir_val_cab)))
print('Total de muestras de validacion de humanos: ', len(os.listdir(dir_val_hum)))

# Mostrando imagenes
nom_cab_ent = os.listdir(dir_ent_cab)
nom_hum_ent = os.listdir(dir_ent_hum)

plt.figure(figsize=(15,5))
img_caballos = [ os.path.join( dir_ent_cab, fname ) for fname in nom_cab_ent[:5] ]
img_humanos = [ os.path.join( dir_ent_hum, fname ) for fname in nom_hum_ent[:5] ]

for i, img_path in enumerate(img_caballos + img_humanos):
    sp = plt.subplot(2, 5, i + 1)
    sp.axis('Off')
    img = mpimg.imread(img_path)
    plt.imshow(img)
plt.show()    

model = models.Sequential()
model.add(layers.Conv2D( 32, (3, 3), activation = 'relu', input_shape = (300, 300)) )
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D( 64, (3, 3), activation = 'relu' ))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D( 128, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D( 128, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))
model.compile( loss = 'binary_crossentropy', optimizer = optimizers.RMSprop(lr = 1e-4), metrics = ['acc'])

# Pre-Procesamiento de datos
## entrenamiento y pruebas
gen_datos_ent = ImageDataGenerator( rescale=1./255 )
gen_datos_pru = ImageDataGenerator( rescale=1./255 )
gen_ent = gen_datos_ent.flow_from_directory( dir_ent, target_size=(300,300), batch_size=32, class_mode='binary' )
gen_val = gen_datos_pru.flow_from_directory( dir_val, target_size=(300,300), batch_size=32, class_mode='binary' )
history = model.fit(gen_ent, steps_per_epoch=10, epochs=10, validation_data = gen_val, validation_steps=50, verbose=1)
print(f"Exactitud de datos de validacion: {history.history['val_acc'][-1]:.3f}")

## graficas de exactitud y perdidas
plt.figure( figsize=(10,5) )
n = np.arange(0,10)
plt.subplot(121)
plt.title('Exactitud de Entrenamiento')
plt.plot(n, history.history['acc'], 'b')
plt.xlabel('Epocas')
plt.ylabel('Exactitud')

plt.subplot(122)
plt.title('Perdida de Entrenamiento')
plt.plot(n, history.history['loss'], 'r')
plt.xlabel('Epocas')
plt.ylabel('Perdida')
plt.show()