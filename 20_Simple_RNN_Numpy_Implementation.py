# Possiamo considerare una Recurrent Neural Network (RNN) e` un ciclo for che ri-utilizza le
# quantita` calcolate durante l'iterazione precedente del loop. 

'''
# Pseudocodice per la RNN:
la trasformazione dell'input e dello stato in un output e` parametrizzata da 2 matrici, 'W' e 'U',
e da un vettore di bias.    [bias (statistica) = un elemento distorsivo del campione]

state_t = 0
for input_t in input_sequence:
    output_t = activation(dot(W, input_t) + dot(U, state_t) + b)            <- Step Function della RNN
    state_t = output_t

'''

import numpy as np


def main():
    ## 197 - Implementazione Numpy di una semplice RNN
    # Numero di timesteps nella sequenza di input
    timesteps = 100
    # Dimensionalita` dello spazio delle caratteristiche di input
    input_features = 32
    # Dimensionalita` dello spazio delle caratteristiche di output
    output_features = 64

    # Input Data: rumore casuale per questo esempio
    inputs = np.random.random((timesteps, input_features))                  # inputs.shape = (100, 32)
    # Initial State: vettore di zeri
    state_t = np.zeros((output_features, ))
    # Creazione casuale delle matrici dei pesi e del vettore di bias
    # Bias (Statistica): un elemento distorsivo del campione.
    W = np.random.random((output_features, input_features))                 # W.shape = (64, 32)
    U = np.random.random((output_features, output_features))                # U.shape = (64, 64)
    b = np.random.random((output_features,))                                # b.shape = (64,)
    
    successive_outputs = []
    # input_t e` un vettore di forma: (input_features, )
    for input_t in inputs:
        # Combina l'input con lo stato corrente (output precedente) per ottenere l'output corrente
        output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)     # output_t.shape = 64
        successive_outputs.append(output_t)
        # Aggiorna lo stato della rete per il successivo timestep
        state_t = output_t
    # Essendo "len(successive_outputs) = 6400 = 100 * 64"
    # Trasformo in array n-dim 'successive_outputs' e ottengo:
    # 'final_output_sequences.shape = (timesteps, output_features) = (100, 64)'
    final_output_sequences = np.array(successive_outputs)


if __name__ == "__main__":
    main()
