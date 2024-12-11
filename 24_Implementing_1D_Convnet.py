from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
from keras.datasets import imdb
from keras.preprocessing import sequence
import numpy as np
import matplotlib.pyplot as plt
import os

def parse_data(rows, columns):
    float_data = np.zeros((len(rows), len(columns) - 1))
    for i, row in enumerate(rows):
        values = [float(x) for x in row.split(',')[1:]]
        float_data[i, :] = values
    return float_data

def get_data_jena():
    data_dir = "/home/z3r0/Desktop/All/Exercises/Deep_Learning/jena_climate"
    fname = os.path.join(data_dir, "jena_climate_2009_2016.csv")
    f = open(fname)
    data = f.read()
    f.close()
    lines = data.split('\n')
    header = lines[0].split(',')
    lines = lines[1:]
    return header, lines

def normalize_data(float_data):
    mean = float_data[:200000].mean(axis=0)
    float_data -= mean
    std = float_data[:200000].std(axis=0)
    float_data /= std

    return float_data, std

def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows), ))

        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets

def get_generators(float_data):
    global train_steps, val_steps, test_steps
    global train_samples, val_samples, test_samples

    train_gen = generator(float_data, lookback=lookback, delay=delay, min_index=0, 
                            max_index=200000, shuffle=True, step=step, batch_size=batch_size)

    val_gen = generator(float_data, lookback=lookback, delay=delay, min_index=200001, 
                            max_index=300000, step=step, batch_size=batch_size)

    test_gen  = generator(float_data, lookback=lookback, delay=delay, min_index=300001, 
                            max_index=None, step=step, batch_size=batch_size)

    train_samples = 200000
    val_samples = 300000 - 200001
    test_samples = (len(float_data) - 300001)
    train_steps = train_samples / batch_size
    val_steps = val_samples / batch_size
    test_steps = test_samples / batch_size
    return train_gen, val_gen, test_gen

def plot_results(history):
    #acc = history["acc"]
    #val_acc = history["val_acc"]
    loss = history["loss"]
    val_loss = history["val_loss"]
    epochs = range(1, len(loss) + 1)

    #plt.plot(epochs, acc, "bo", label="Training Accuracy")
    #plt.plot(epochs, val_acc, "b", label="Validation Accuracy")
    #plt.title("Training & Validation Accuracy")
    #plt.legend()
    
    #plt.figure()

    plt.plot(epochs, loss, "bo", label="Training Loss")
    plt.plot(epochs, val_loss, "b", label="Validation Loss")
    plt.title("Training & Validation Loss")
    plt.legend()

    plt.show()

def get_data_imdb():
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    print("[+] len(x_train): %s" % len(x_train))        # 25000
    print("[+] len(y_train): %s" % len(y_train))        # 25000
    print("[+] len(x_test): %s" % len(x_test))          # 25000
    print("[+] len(y_test): %s" % len(y_test))          # 25000

    # Padding delle sequenze
    x_train = sequence.pad_sequences(x_train, maxlen=max_len)
    x_test = sequence.pad_sequences(x_test, maxlen=max_len)
    print("[+] x_train.shape: %s" % (x_train.shape, ))
    print("[+] x_test.shape: %s" % (x_test.shape, ))
    
    return x_train, y_train

## 227 - Addestrare e valutare una semplice convnet 1D sui dati IMDB
def train_and_evaluate_model(x_train, y_train):
    model = Sequential()
    model.add(layers.Embedding(max_features, 128, input_length=max_len))
    model.add(layers.Conv1D(32, 7, activation='relu'))
    model.add(layers.MaxPooling1D(5))
    model.add(layers.Conv1D(32, 7, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(1))
    model.summary()

    model.compile(optimizer=RMSprop(lr=1e-4), loss="binary_crossentropy", metrics=["acc"])
    return model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

    ## RISULTATI ##
    # loss: 0.2276 - acc: 0.8030 - val_loss: 0.5116 - val_acc: 0.7446 - Speed: 90us/step

    # La velocita` risulta essere molto superiore rispetto a quella ottenuta
    # durante l'addestramento e la convalida dello strato LSTM.

## 228 - Addestrare e valutare una semplice convnet 1D sui dati Jena
def train_and_evaluate_model_jena_data(float_data, train_gen, val_gen):
    model = Sequential()
    model.add(layers.Conv1D(32, 5, activation="relu", input_shape=(None, float_data.shape[-1])))
    model.add(layers.MaxPooling1D(3))
    model.add(layers.Conv1D(32, 5, activation="relu"))
    model.add(layers.MaxPooling1D(3))
    model.add(layers.Conv1D(32, 5, activation="relu"))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(1))

    model.compile(optimizer=RMSprop(), loss="mae")
    return model.fit_generator(train_gen, steps_per_epoch=500, epochs=20, validation_data=val_gen, validation_steps=val_steps)

    ## RISULTATI ##
    # loss: 0.2475 - val_loss: 0.4813 - Speed: 8ms/step

    # Poiche' e` stata utilizzata "MAE" (Mean Absolute Error) come Loss Function 
    # l'oggetto dizionario "history.history" non conterra` informazioni relative
    # all'accuratezza ottenuta durante l'addestramento e la convalida.

## 230 - Preparare generatori di dati con risoluzione maggiore per il dataset Jena
def get_higher_resolution_generators(float_data):
    # Combinando una Convnet 1D per preprocessare i dati e una RNN diviene possibile processare
    # sequenze piu` lunghe di dati. Di conseguenza, nell'esercizio di previsione del meteo 
    # con dataset Jena, e` possibile guardare a dati piu` lontani nel tempo (aumentando il valore
    # del parametro "lookback" del generatore di dati) o guardare a serie temporali ad alta risoluzione 
    # (decrementando il valore del parametro "step" del generatore di dati)
    global val_steps, test_steps

    step = 3            # Precedentemente settato a 6 (1 punto all'ora), ora a 3 (1 punto ogni mezz'ora)

    train_gen = generator(float_data, lookback=lookback, delay=delay, min_index=0, 
                            max_index=200000, shuffle=True, step=step, batch_size=batch_size)
    val_gen = generator(float_data, lookback=lookback, delay=delay, min_index=200001, 
                            max_index=300000, step=step, batch_size=batch_size)
    test_gen  = generator(float_data, lookback=lookback, delay=delay, min_index=300001, 
                            max_index=None, step=step, batch_size=batch_size)
    
    # // -> Floor Division -> Divisione che elimina la parte decimale del numero
    val_steps = (300000 - 200001 - lookback) // batch_size              # val_steps = 769
    test_steps = (len(float_data) - 300001 - lookback) // batch_size    # test_steps = 930

    return train_gen, val_gen, test_gen


## 230 - Modello combinando una Base Convoluzionale 1D e uno strato GRU
def model_1d_convolutional_base_and_gru(float_data, train_gen, val_gen):
    model = Sequential()
    model.add(layers.Conv1D(32, 5, activation="relu", input_shape=(None, float_data.shape[-1])))
    model.add(layers.MaxPooling1D(3))
    model.add(layers.Conv1D(32, 5, activation="relu"))
    model.add(layers.GRU(32, dropout=0.1, recurrent_dropout=0.5))
    model.add(layers.Dense(1))

    model.compile(optimizer=RMSprop(), loss="mae")
    return model.fit_generator(train_gen, steps_per_epoch=500, epochs=20, validation_data=val_gen, validation_steps=val_steps)

    ## RISULTATI ##
    # loss: 0.2223 - val_loss: 0.2934 - Speed: 134ms/step

    # Giudicando dalla perdita di convalida questo setup del modello non e' buono
    # come quello con soltanto lo strato GRU regolarizzato, sebbene sia molto piu` veloce.

## 221 - Preparare i dati di IMDB
def main():
    #   x_train, y_train = get_data_imdb()
    #   history = train_and_evaluate_model(x_train, y_train)
    #   plot_results(history.history)
    
    # Le funzioni sono state eliminate per questioni di spazio, esse si trovano
    # nel file: "22_Jena_Weather_Temperature_Forecasting_Problem.py"
    global lookback, step, delay, batch_size
    lookback = 1440
    step = 6
    delay = 144
    batch_size = 128
    header, lines = get_data_jena()
    float_data, std = normalize_data(parse_data(lines, header))
    
    #   train_gen, val_gen, test_gen = get_generators(float_data)
    #   history = train_and_evaluate_model_jena_data(float_data, train_gen, val_gen)
    #   plot_results(history.history)

    train_gen, val_gen, test_gen = get_higher_resolution_generators(float_data)
    history = model_1d_convolutional_base_and_gru(float_data, train_gen, val_gen)
    plot_results(history.history)

if __name__ == "__main__":
    global max_features, max_len
    max_features = 10000
    max_len = 500
    main()


'''
|======================|
| Summary of the Model |
|======================|-----------------------------------------
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, 500, 128)          1280000   
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 494, 32)           28704     
_________________________________________________________________
max_pooling1d_1 (MaxPooling1 (None, 98, 32)            0         
_________________________________________________________________
conv1d_2 (Conv1D)            (None, 92, 32)            7200      
_________________________________________________________________
global_max_pooling1d_1 (Glob (None, 32)                0         
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 33        
=================================================================
Total params: 1,315,937
Trainable params: 1,315,937
Non-trainable params: 0
_________________________________________________________________
-----------------------------------------------------------------
'''
