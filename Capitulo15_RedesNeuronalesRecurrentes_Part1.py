##=======================================================================================
## Redes Neuronales Recurrentes (RNN) Parte 1
## Book: Aprendizaje Automatico y Profundo en Python, Capitulo 15
## Description: Sentiment Analysis with Keras
## Author: Carlos M. Pineda Pertuz
##=======================================================================================
## Programmer: Alberto Ramirez Bello
## Date: September 22nd, 2024
##=======================================================================================

import numpy as np
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

(X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data( num_words=10000 )

# Ajustando las entradas al mismo numero de longitud
X_train = pad_sequences(X_train, maxlen=150)
X_test = pad_sequences(X_test, maxlen=150)
palabra_a_id = keras.datasets.imdb.get_word_index()
id_a_palabra = {i:palabra for palabra , i in palabra_a_id.items()}

print(f"\nComentario: {str([id_a_palabra.get(i,'') for i in X_train[6]])}")
print(f'Etiqueta: {str(y_train[6])}')

# Creando modelo de red neuronal
model =  keras.models.Sequential( [keras.layers.Embedding( input_dim=10000, output_dim=32, input_length=150 ),
                                   keras.layers.LSTM( 32, dropout=0.1, recurrent_dropout=0.2, return_sequences=True ),
                                   keras.layers.LSTM( 32, dropout=0.1, recurrent_dropout=0.2),
                                   keras.layers.Dense( 1, activation='sigmoid') ] )

model.summary()

# Compilando y entrenando el modelo
model.compile( loss='binary_crossentropy', optimizer='adam', metrics=['acc'] )
model.fit( X_train, y_train, epochs=5, batch_size=64, verbose=1, validation_data=(X_test, y_test))

# Exactitud del modelo con datos de prueba
scores = model.evaluate( X_test, y_test, verbose=1)
print(f"\nExactitud: {scores[1]:.3f}")

# Probando el rendimiento
texto_neg = X_test[9]
texto_pos = X_test[13]
texts = ( texto_neg, texto_pos )
textos = pad_sequences( texts, maxlen=150, value=0.0 )
preds = model.predict(textos)
print('Predicciones: \n', preds)
