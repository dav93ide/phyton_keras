from keras.datasets import imdb
from keras.preprocessing import sequence
from keras import layers
from keras.models import Sequential
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_results(history):
    acc = history["acc"]
    val_acc = history["val_acc"]
    loss = history["loss"]
    val_loss = history["val_loss"]
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, "bo", label="Training Accuracy")
    plt.plot(epochs, val_acc, "b", label="Validation Accuracy")
    plt.title("Training & Validation Accuracy")
    plt.legend()
    
    plt.figure()

    plt.plot(epochs, loss, "bo", label="Training Loss")
    plt.plot(epochs, val_loss, "b", label="Validation Loss")
    plt.title("Training & Validation Loss")
    plt.legend()

    plt.show()

def get_samples():
    # Numero di parole da considerare come caratteristiche
    max_features = 10000
    # Taglia i testi dopo questo numero di parole
    maxlen = 500
    # Carica i dati:    x_train.shape = (25000,)    x_test.shape = (25000,)
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

    # Inverte le sequenze:  len(x_train) = 25000    len(x_test) = 25000
    x_train = [x[::-1] for x in x_train]
    x_test = [x[::-1] for x in x_test]
    
    # Padding delle sequenze, trasforma le liste ottenute in tensori 2D di forma:   
    #       x_train.shape = (25000, 500)    x_test.shape = (25000, 500)
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    return max_features, x_train, y_train

## 220 - Addestrare e valutare un LSTM utilizzando sequenze invertite
def train_lstm_reversed_sequences(max_features, x_train, y_train):
    model = Sequential()
    model.add(layers.Embedding(max_features, 128))
    model.add(layers.LSTM(32))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=['acc'])
    history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

    ## RISULTATI ##
    # loss: 0.1226 - acc: 0.9593 - val_loss: 0.4928 - val_acc: 0.8602
    return history.history

## 221 - Addestrare e valutare un LSTM bidirezionale
def train_bidirectional_lstm(max_features, x_train, y_train):
    model = Sequential()
    model.add(layers.Embedding(max_features, 128))
    model.add(layers.Bidirectional(layers.LSTM(32)))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])
    history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

    ## RISULTATI ##
    # loss: 0.1283 - acc: 0.9578 - val_loss: 0.4169 - val_acc: 0.8644
    return history.history

## 222 - Addestrare un GRU bidirezionale
def train_bidirectional_gru():
    # Utilizza i dati dell'esercizio:   22_Jena_Weather_Temperature_Forecasting_Problem.py
    model = Sequential()
    model.add(layers.Bidirectional(layers.GRU(32), input_shape=(None, float_data.shape[-1])))
    model.add(layers.Dense(1))
    model.compile(optimizer=RMSprop(), loss="mae")
    history = model.fit_generator(train_gen, steps_per_epoch=500, epochs=40, validation_data=val_gen, validation_steps=val_steps)
    # I risultati sono quasi come quelli dell'esercizio precendete con strato GRU non Unidirezionale.

## 220 - Addestrare e valutare un LSTM utilizzando sequenze inverse
def main():
    max_features, x_train, y_train = get_samples()

    #   history = train_lstm_reversed_sequences(max_features, x_train, y_train)
    #   plot_results(history)

    #   history = train_bidirectional_lstm(max_features, x_train, y_train)
    #   plot_results(history)

    history = train_bidirectional_gru()
    plot_results(history)

if __name__ == "__main__":
    main()
