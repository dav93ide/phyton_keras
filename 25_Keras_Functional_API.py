from keras.models import Sequential, Model
from keras import layers
from keras import Input
import numpy as np

def main():
    ## 237 - L'esempio minimale seguente mostra passo a passo un semplice modello Sequential
    # e il suo corrispettivo equivalente nell'API funzionale.
    seq_model = Sequential()        # Modello 'Sequential'
    seq_model.add(layers.Dense(32, activation='relu', input_shape=(64, )))
    seq_model.add(layers.Dense(32, activation='relu'))
    seq_model.add(layers.Dense(10, activation='softmax'))

    # L'equivalente 'Funzionale' del Modello 'Sequential'
    input_tensor = Input(shape=(64, ))
    x = layers.Dense(32, activation='relu', input_shape=(64, ))(input_tensor)
    x = layers.Dense(32, activation='relu')(x)
    output_tensor = layers.Dense(10, activation='softmax')(x)

    # La classe 'Model' trasforma un tensore in input e un tensore in output in un modello.
    model = Model(input_tensor, output_tensor)
    model.summary()

    # 238 - Quando si deve compilare, addestrare e valutare un'istanza di 'Model',
    # l'API e' uguale a Sequential.
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')     # Compila il modello
    # Genera dei dati Numpy dummy su cui addestrare
    x_train = np.random.random((1000, 64))
    y_train = np.random.random((1000, 10))
    # Addestra il modello per 10 epoche
    model.fit(x_train, y_train, epochs=10, batch_size=128)
    # Valuta il modello
    score = model.evaluate(x_train, y_train)



if __name__ == "__main__":
    main()


'''
|======================|
| Summary of the Model |
|======================|-----------------------------------------
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_2 (InputLayer)         (None, 64)                0         
_________________________________________________________________
dense_5 (Dense)              (None, 32)                2080      
_________________________________________________________________
dense_6 (Dense)              (None, 32)                1056      
_________________________________________________________________
dense_7 (Dense)              (None, 10)                330       
=================================================================
Total params: 3,466
Trainable params: 3,466
Non-trainable params: 0
_________________________________________________________________
-----------------------------------------------------------------
'''
