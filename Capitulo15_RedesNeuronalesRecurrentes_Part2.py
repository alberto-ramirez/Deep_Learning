##=======================================================================================
## Redes Neuronales Recurrentes (RNN) Parte 2
## Book: Aprendizaje Automatico y Profundo en Python, Capitulo 15
## Description: Text Generation
## Author: Carlos M. Pineda Pertuz
##=======================================================================================
## Programmer: Alberto Ramirez Bello
## Date: September 30th, 2024
##=======================================================================================

import numpy as np 
import pandas as pd
import string 
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.models import load_model

path = tf.keras.utils.get_file('Shakespear.txt', 'https://www.gutenberg.org/files/55206/55206-0.txt')
text = open( path, 'rb' ).read().decode( encoding='utf-8' )
print('\n',text[:500])
print(f"Longitud del texto: {len(text)} carácteres.")
vocab = sorted(set(text))
print(f'El texto esta compuesto de: {len(vocab)} carácteres únicos.')

char_a_ind = {char:i for i, char in enumerate(vocab)}
ind_a_char = np.array(vocab)
codif_text = np.array([char_a_ind[c] for c in text])
long_seq = 100
dataset_caract = tf.data.Dataset.from_tensor_slices(codif_text)
secuencias = dataset_caract.batch(long_seq+1, drop_remainder=True)

for i in secuencias.take(2):
    print(ind_a_char[i.numpy()])

def separa_entrada_etiqueta(p):
    texto_entrada = p[:-1]
    texto_destino = p[1:]
    return texto_entrada, texto_destino

dataset = secuencias.map(separa_entrada_etiqueta)
TAM_LOTE = 64
TAM_BUFFER = 10000

dataset = dataset.shuffle(TAM_BUFFER).batch(TAM_LOTE, drop_remainder=True)
tam_vocab = len(vocab)
tam_embedding = 256
unidades_rnn=1024

def crear_modelo(tam_vocab, tam_embedding, unidades_rnn, tam_lote):
    model = Sequential()
    model.add( Embedding( input_dim=tam_vocab, output_dim=tam_embedding,
                         batch_input_shape=[tam_lote, None] ) )
    model.add( LSTM(unidades_rnn, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform') )
    model.add( Dense(tam_vocab) )
    return model

model = crear_modelo( tam_vocab= len(vocab), tam_embedding=tam_embedding, unidades_rnn=unidades_rnn, tam_lote=TAM_LOTE)
print(f'\nModel Summary:\n{model.summary()}')

def perdida(etiquetas, preds):
    return tf.keras.losses.sparse_categorical_crossentropy(etiquetas, preds, from_logits=True)

model.compile(optimizer='adam', loss=perdida)
epocas = 5
history = model.fit(dataset, epochs=epocas, verbose=1)
model.save('modelo.h5.keras')
model = crear_modelo( tam_vocab, tam_embedding, unidades_rnn, tam_lote=1 )
model.load_weights('modelo.h5')
model.build(tf.TensorShape([1, None]))

def generar_texto(model, cad_inicio):
    total_gen = 100
    ent_e = [char_a_ind[s] for s in cad_inicio]
    ent_e = tf.expand_dims(ent_e, 0)
    texto_generado = []
    temperature = 0.5
    model.reset_states()

    for i in range(total_gen):
        predicciones = model(ent_e)
        predicciones = tf.squeeze(predicciones, 0)
        predicciones = predicciones / temperature
        id_prediccion = tf.random.categorical(predicciones, num_samples=1)[-1,0].numpy()
        ent_e = tf.expand_dims([id_prediccion], 0)
        texto_generado.append(ind_a_char[id_prediccion])

    return (cad_inicio +"".join(texto_generado))

print(f'\n{generar_texto(model, cad_inicio=u"libro")}')