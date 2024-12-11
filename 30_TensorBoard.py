# Per lanciare il server di TensorBoard da shell specificando in quale directory
# reperire i logs:
#       $ tensorboard --logdir=tensor_board_logs 

# Una volta lanciato il server di TensorBoard e` necessario navigare all'indirizzo:
#       http://127.0.0.1:6006

import keras
from keras import layers
from keras.datasets import imdb
from keras.preprocessing import sequence

## 253 - Modello di Classificazione Testuale da utilizzare con TensorBoard
def text_classification_model():
    # Numero di parole da considerare come caratteristiche
    max_features = 2000
    # Taglia i testi dopo questo numero di parole
    max_len = 500

    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    x_train = sequence.pad_sequences(x_train, maxlen=max_len)
    x_test = sequence.pad_sequences(x_test, maxlen=max_len)

    model = keras.models.Sequential()
    model.add(layers.Embedding(max_features, 128, input_length=max_len, name="embed"))
    model.add(layers.Conv1D(32, 7, activation="relu"))
    model.add(layers.MaxPooling1D(5))
    model.add(layers.Conv1D(32, 7, activation="relu"))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(1))
    
    model.summary()
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])
    callback = [
        keras.callbacks.TensorBoard(
            log_dir="tensor_board_logs",    # Files di log salvati su questa directory
            histogram_freq=1,               # Registra gli istogrammi delle attivazioni ogni 1 epoca
            embeddings_freq=1,              # Registra i dati di incorporamento ogni 1 epoca
            embeddings_data=x_test[:100]    # Dati che devono essere incorporati agli strati specificati
        )                                   # in 'embeddings_layers_names'. 
    ]                                       # Numpy array (se il modello ha un singolo input) o lista di
                                            # arrays Numpy (se il modello ha piu` inputs)
    history = model.fit(x_train, y_train, epochs=20, batch_size=128, validation_split=0.2, callbacks=callback)

    ## RISULTATI ##
    # 2s 87us/step - loss: 0.0886 - acc: 0.1406 - val_loss: 1.2197 - val_acc: 0.2302

def main():
    text_classification_model()    

if __name__ == "__main__":
    main()
