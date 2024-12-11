from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
import os
import numpy as np
import matplotlib.pyplot as plt

## 208 - Parsare i dati
def parse_data(rows, columns):
    float_data = np.zeros((len(rows), len(columns) - 1))
    for i, row in enumerate(rows):
        values = [float(x) for x in row.split(',')[1:]]
        float_data[i, :] = values
    return float_data
    
## 208 - Ispezionare i dati del Dataset del meteo di Jena
def get_data():
    data_dir = "/home/z3r0/Desktop/All/Exercises/Deep_Learning/jena_climate"
    fname = os.path.join(data_dir, "jena_climate_2009_2016.csv")

    f = open(fname)
    data = f.read()
    f.close()

    lines = data.split('\n')
    # Contenuto di Header sotto script entry-point
    header = lines[0].split(',')
    # len(lines) = 420551
    lines = lines[1:]
    return header, lines

## 209 - Tracciare graficamente le serie temporali delle temperature
def plot_temp_timeseries(float_data):
    #  len(float_data) = 420551                     len(temp) = 420551
    temp = float_data[:, 1]
    plt.plot(range(len(temp)), temp)
    plt.show()

    ## 209 - Tracciare graficamente le serie temporali delle temperature dei primi 10 giorni
    plt.plot(range(1440), temp[:1440])
    plt.show()

## 210 - Normalizzare i dati
def normalize_data(float_data):
    # Prendo i primi 200000 timesteps come Training Data
    mean = float_data[:200000].mean(axis=0)
    float_data -= mean
    std = float_data[:200000].std(axis=0)           # Deviazione Standard
    float_data /= std

    return float_data, std

## 211 - Generatore che produce campioni di serie temporali e i loro targets.
# Argomenti della funzione:
#   data        ->  Array originale di dati a virgola mobile        
#   lookback    ->  Di quanti timesteps indietro dovrebbero andare i dati di input
#   delay       ->  Di quanti timesteps nel futuro dovrebbero essere i targets
#   min_index & max_index ->  Indici nell'array 'data' che delimitano da quali timesteps attingere.
#   shuffle     ->  Se mescolare i dati o tracciarli in ordine cronologico
#   batch_size  ->  Il numero di campioni per batch
#   step        ->  Il periodo, in timesteps, 
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

## 211 - Preparare i generatori di addestramento, convalida e test.
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

    ## IMPORTANTE! ##
    # (A pagina 230 c'e` la versione corretta dove si divide (con '//' -> floor division) per 'batch_size'(128))
    # A pagina 212 del libro mettono che:    val_steps = (300000 - 200001 - lookback) 
    # (val_steps = 98559) pero` facendo in questo modo il generatore continua a generare output 
    # mentre la rete e` come freezzata per molto tempo!

    # Cercando su internet dicono che gli steps di training e di convalida usando un generatore devono essere
    # uguali a:         num_samples / batch_size
    # Quindi per far funzionare l'addestramento e la convalida ho messo che i vari steps corrispondono a: 
    #   train_steps = train_samples / batch_size    =>  200000 / 128                        =>  train_steps = 1562
    #   val_steps = val_samples / batch_size        =>  (300000 - 200001) / 128             =>  val_steps = 781
    #   test_steps = test_samples / batch_size      =>  (len(float_data) - 300001) / 128    =>  test_steps = 941

    train_steps = train_samples / batch_size
    # Quanti steps comporre da 'val_gen' per poter vedere l'intero set di convalida.
    val_steps = val_samples / batch_size
    # Quanti steps comporre da 'test_gen' per poter vedere l'intero set di test.
    test_steps = test_samples / batch_size
    return train_gen, val_gen, test_gen

## 213 - Calcolare l' Errore Assoluto Medio (Mean Absolute Error - MAE) della linea base di buon-senso
# Per essere effettivamente utile un modello di deep-learning deve essere in grado di battere una linea base di buon-senso ottenuta
# cercando di risolvere il problema alla mano in maniera molto semplice.
# In questo caso per prevedere la temperatura considereremo che la temperatura nelle successive 24 ore sia uguale alla temperatura attuale.
def evaluate_naive_method(val_steps, val_gen, std):
    batch_maes = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))                              # MAE   ->  [ np.mean(np.abs(preds - targets)) ]
        batch_maes.append(mae)
    print("[+] MAE = np.mean(batch_maes) = %s" % (np.mean(batch_maes)))     # MAE = 2.56456384786

    # Convertire MAE all'errore in Celsius
    # Poiche` i dati della temperatura sono stati normalizzati per essere centrati in 0 e per avere una deviazione standard di 1
    # il MAE ottenuto si traduce in un Errore Assoluto Medio di:    0.29 * temperature_std  gradi Celsius = 2.57 C
    celsius_mae = 0.29 * std[1]

## 213 - Addestrare e valutare un modello densamente connesso
# Allo stesso modo in cui e` utile utilizzare un linea base di buon-senso prima di iniziare un approccio di machine-learning complesso,
# e` utile provare con semplici modelli di machine-learning prima di procedere con modelli piu` complicati che utilizzato gli RNN.
def simple_fully_connected_model(float_data, train_gen, val_gen):
    # Useremo un modello totalmente connesso che parte appiattendo i dati ed eseguendoli attraverso 2 strati Dense.
    # Da Notare l'assenza di funzione d'attivazione nell'ultimo strato Dense, tipica dei problemi di regressione.
    # Si utilizzera` l'Errore Assoluto Medio come perdita.
    model = Sequential()
    model.add(layers.Flatten(input_shape=(lookback // step, float_data.shape[-1])))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer=RMSprop(), loss="mae")
    history = model.fit_generator(train_gen, steps_per_epoch=train_steps, epochs=20, validation_data=val_gen, validation_steps=val_steps)
    # Con questa nuova impostazione degli steps, seppur avendo il triplo di train_steps rispetto a prima, un'epoca viene completata in media
    # in 7 secondi rispetto ai piu` di 3 minuti usando l'impostazione del libro (si bloccava al 99-esimo campione di ogni epoca...)
    
    ## RISULTATI ##
    # loss: 0.1639 - val_loss: 0.3425

    return history.history

## 214 - Tracciare i risultati
def plot_results(history):
    loss = history["loss"]
    val_loss = history["val_loss"]
    epochs = range(1, len(loss) + 1)

    plt.figure()
    plt.plot(epochs, loss, 'bo', label="Training Loss")
    plt.plot(epochs, val_loss, 'b', label="Validation Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.show()

## 215 - Addestrare e valutare un modello basato su GRU (Gated Recurrent Unit)
def gru_based_model(float_data, train_gen, val_gen):
    model = Sequential()
    model.add(layers.GRU(32, input_shape=(None, float_data.shape[-1])))
    model.add(layers.Dense(1))

    model.compile(optimizer=RMSprop(), loss='mae')
    # Per addestrare per 100 steps di addestramento (invece dei 781 di train_steps) pongo gli steps di convalida a
    # 100 (invece dei 781 di val_steps, come scritto sul libro). Ponendo 100 l'addestramento viene eseguito abbastanza 
    # linearmente soffermandosi per poco al 99-esimo step, mentre usando 'val_steps' si sofferma per molto
    # tempo al 99-esimo step.
    history = model.fit_generator(train_gen, steps_per_epoch=100, epochs=20, validation_data=val_gen, validation_steps=100)
    
    ## RISULTATI ##
    # loss: 0.2704 - val_loss: 0.2857

    # Il grafico dei risultati ottenuti mostra che quasi subito il modello va in overfitting

    return history.history

## 216 - Addestrare e valutare un modello basato su GRU regolarizzato da dropout
def gru_based_model_dropout(float_data, train_gen, val_gen):
    model = Sequential()
    model.add(layers.GRU(32, dropout=0.2, recurrent_dropout=0.2, input_shape=(None, float_data.shape[-1])))
    model.add(layers.Dense(1))
    model.compile(optimizer=RMSprop(), loss='mae')
    history = model.fit_generator(train_gen, steps_per_epoch=500, epochs=40, validation_data=val_gen, validation_steps=val_steps)

    ## RISULTATI ##
    # loss: 0.2799 - val_loss: 0.2739

    # Dal grafico e` possibile vedere come, dopo aver aggiunto il DROPOUT, il modello non vada piu` subito
    # in overfitting, ma sembra che vi sia un 'collo di bottiglia' delle performance pertanto bisognerebbe
    # aumentare la capacita` della rete. Per aumentare la capacita` della rete e` possibile aumentare
    # il numero di unita` negli strati o aggiungere ulteriori strati.

    return history.history

## 218 - Addestrare e valutare un modello GRU impilato regolarizzato da dropout
def gru_stacked_model_dropout(float_data, train_gen, val_gen):
    model = Sequential()
    # Per poter impilare degli strati GRU e` necessario che gli strati GRU intermedi ritornino la loro intera sequenza
    # di outputs (un tensore 3D), invece dell'output dell'ultimo timestep.
    # Per fare cio` in Keras e` necessario specificare negli strati intermedi l'argomento 'return_sequences=True'. 
    model.add(layers.GRU(32, dropout=0.1, recurrent_dropout=0.5, return_sequences=True, input_shape=(None, float_data.shape[-1])))
    model.add(layers.GRU(64, activation='relu', dropout=0.1, recurrent_dropout=0.5))
    model.add(layers.Dense(1))
    model.compile(optimizer=RMSprop(), loss='mae')
    history = model.fit_generator(train_gen, steps_per_epoch=500, epochs=30, validation_data=val_gen, validation_steps=val_steps)

    ## RISULTATI ##
    # loss: 0.2609 - val_loss: 0.2695

    return history.history

def main():
    global lookback, step, delay, batch_size
    lookback = 1440
    step = 6
    delay = 144
    batch_size = 128

    header, lines = get_data()
    float_data, std = normalize_data(parse_data(lines, header))
    #   plot_temp_timeseries(float_data)

    train_gen, val_gen, test_gen = get_generators(float_data)
    #   evaluate_naive_method(val_steps, val_gen, std)  
    
    #   history = simple_fully_connected_model(float_data, train_gen, val_gen)
    #   plot_results(history)
    
    #   history = gru_based_model(float_data, train_gen, val_gen)
    #   plot_results(history)

    #   history = gru_based_model_dropout(float_data, train_gen, val_gen)
    #   plot_results(history)

    history = gru_stacked_model_dropout(float_data, train_gen, val_gen)
    plot_results(history)

if __name__ == "__main__":
    main()


'''
|================|
| Header Content |
|================|----------------------------

    '['"Date Time"',        '"p (mbar)"',               '"T (degC)"',       
    '"Tpot (K)"',           '"Tdew (degC)"',            '"rh (%)"', 
    '"VPmax (mbar)"',       '"VPact (mbar)"',           '"VPdef (mbar)"',   
    '"sh (g/kg)"',          '"H2OC (mmol/mol)"',        '"rho (g/m**3)"', 
    '"wv (m/s)"',           '"max. wv (m/s)"',          '"wd (deg)"']'

----------------------------------------------
'''
