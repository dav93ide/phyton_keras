from keras.datasets import imdb
from keras import models
from keras import layers
from keras import regularizers
import numpy as np

## Aggiungere DROPOUT agli strati del modello
def add_dropout():
    model = model.Sequential()
    model.add(layers.Dense(16, activation = 'relu', input_shape = (10000, )))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(16, activation = 'relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation = 'sigmoid'))

## Aggiungere REGOLARIZZAZIONE L2 al modello
# WEIGHT REGULARIZATION: vengono aggiunti dei vincoli alla complessita` della rete forzando i suoi pesi ad accettare solo PICCOLI VALORI, il che rende la distribuzione
# dei valori dei pesi piu` regolare. Alla LOSS FUNCTION della rete viene aggiunto un COSTO associato ad avere grandi pesi.
# L1 REGULARIZATION: il costo aggiunto e` proporzionale al "valore assoluto dei coefficienti dei pesi"
# L2 REGULARIZATION: il costo aggiunto e` proporzionale al "quadrato del valore dei coefficienti dei pesi"
def get_model_l2():
    model = models.Sequential()
    # L1 REGULARIZATION         ->  regularizers.l1(0.001)
    # L2 REGULARIZATION         ->  regularizers.l2(0.001)
    # L1 + L2 REGULARIZATION    ->  regularizers.l1_l2( l1 = 0.001, l2 = 0.001 )
    model.add(layers.Dense(16, kernel_regularizer = regularizers.l2(0.001)), activation = 'relu', input_shape = (10000, ))
    model.add(layers.Dense(16, kernel_regularizer = regularizers.l2(0.001), activation = 'relu'))
    model.add(layers.Dense(1, activation = 'sigmoid'))
    model.compile(
        optimizer = "rmsprop",
        loss = "binari_crossentropy",
        metrics = ["accuracy"]
    )
    return model


## Versione del modello con MAGGIOR CAPACITA`
def get_higher_model():
    model = models.Sequential()
    model.add(layers.Dense(512, activation = 'relu'), input_shape = (10000, ))
    model.add(layers.Dense(512, activation = 'relu'))
    model.add(layers.Dense(1, activation = 'sigmoid'))
    model.compile(
        optimizer = "rmsprop",
        loss = "binari_crossentropy",
        metrics = ["accuracy"]
    )
    return model


## Versione del modello con MINOR CAPACITA`
# Generalmente per trovare la dimensione appropriata del modello si parte con un modello con relativamente pochi strati e parametri, e si aumenta
# la dimensione degli strati o si aggiungo nuovi strati fino a quando non si notano rendimenti decrescenti riguardo la PERDITA DI CONVALIDA (VALIDATION LOSS)
def get_lower_model():
    model = models.Sequential()
    model.add(layers.Dense(4, activation = 'relu'), input_shape = (10000, ))
    model.add(layers.Dense(4, activation = 'relu'))
    model.add(layers.Dense(1, activation = 'sigmoid'))
    model.compile(
        optimizer = "rmsprop",
        loss = "binari_crossentropy",
        metrics = ["accuracy"]
    )
    return model


## Crea e ritorna il modello (modello puramente d'esempio)
def get_model():
    model = models.Sequential()
    model.add(layers.Dense(16, activation = 'relu', input_shape = (10000, )))
    model.add(layers.Dense(16, activation = 'relu'))
    model.add(layers.Dense(1, activation = 'sigmoid'))
    model.compile(
        optimizer = "rmsprop",
        loss = "binari_crossentropy",
        metrics = ["accuracy"]
    )
    return model;


## K-FOLD CROSS VALIDATION
# Si dividono i dati in K partizioni di grandezza uguale. Per ogni partizione 'i' si addestra un modello sulle rimanenti 'K-1' partizioni e lo si valuta
# sulla partizione 'i'. Il punteggio finale sara` la media dei 'K' punteggi ottenuti. Come per la VALIDAZIONE HOLD-OUT e` possibile riservare un SET DI CONVALIDA
# per la calibrazione del modello.
def kfold_cross_validation():
    k = 4
    num_validation_samples = len(train_data) // k
    np.random.shuffle(train_data)
    validation_scores = []
    for fold in range(k):
        # Seleziona i dati della partizione di convalida
        validation_data = train_data[ num_validation_samples * fold : num_validation_samples * (fold + 1)]
        # Utilizza i dati rimanenti come dati di addestramento. L'operatore '+' e` la concatenazione di liste.
        training_data = train_data[:num_validation_samples * fold] + train_data[num_validation_samples * (fold + 1)]

        model = get_model()
        model.train(training_data)
        validation_score = model.evaluate(validation_data)
        validation_scores.append(validation_score)
    # PUNTEGGIO DI CONVALIDA: media dei punteggi di validazione dei 'K' gruppi
    model = get_model()
    model.train(train_data)
    test_score = model.evaluate(test_data)

## HOLD-OUT VALIDATION
# Viene messa da parte una frazione dei dati come set di test. Si addestra sui dati rimanenti e si valuta sul set di test.
# Per evitare perdite di informazioni non si dovrebbe regolare il modello basandosi sul SET DI TEST, inoltre si dovrebbe anche riservare un SET DI CONVALIDA.
def hold_out_validation():
    num_validation_samples = 1000
    np.random.shuffle(train_data)
    # Definizione del set di convalida
    validation_data = train_data[:num_validation_samples]
    # Definizione del set di addestramento
    training_data = train_data[num_validation_samples:]
    # Addestra il modello sui dati d'addestramento e valuta sui dati di convalida
    model = get_model()
    model.train(training_data)
    validation_score = model.evaluate( validation_data )
    # A questo punto si regola il modello e i suoi iperparametri, lo si riaddestra e lo si riperfeziona ecc...
    # [...]
    model = get_model()
    # Una volta aver regolato gli iperparametri e` comune addestrare il modello finale da zero su tutti i dati non di test disponibili.
    model.train(np.concatenate([training_data, validation_data]))
    test_score = model.evaluate(test_data)


## CARICAMENTO DATASETS
def load_data():
    global train_data, train_labels, test_data, test_labels
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data( num_words = 10000 )

## MAIN FUNCTION
def main():
    load_data()
    hold_out_validation()
    kfold_cross_validation()
    add_dropout()

## ENTRY POINT
if __name__ == "__main__":
    main()
