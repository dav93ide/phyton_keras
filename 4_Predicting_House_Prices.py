# Predizione di un valore continuo invece che di un'etichetta discreta.

from keras.datasets import boston_housing
from keras import models
from keras import layers
import matplotlib.pyplot as plt
import numpy as np
import os

def train_final_model():
    model = build_model()
    model.fit(
        train_data,
        train_targets,
        epochs = 80,
        batch_size = 16,
        verbose = 0
    )
    test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
    print("\n\n")
    print("[+] test_mae_score:\n%s\n" % test_mae_score)

# Grafico dei punteggi di validazione escludendo i primi 10 punti di dati, che sono su una scala differente rispetto al resto della curva,
# e sostituendo ogni punto con una media mobile esponenziale dei punti precedenti, per ottenere una curva piu` liscia
def smooth_curve(points, factor = 0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def plot_validation_scores_two():
    smooth_mae_history = smooth_curve(average_mae_history[10:])
    plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
    plt.xlabel("Epochs")
    plt.ylabel("Validation MAE")
    plt.show()
    raw_input("> Premere Enter per continuare...")

# Grafico dei punteggi di validazione
def plot_validation_scores():
    plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
    plt.xlabel("Epochs")
    plt.ylabel("Validation MAE")
    plt.show()
    raw_input("> Premere Enter per continuare...")

# A causa dell'esiguo numero di campioni di dati d'addestramento non e` possibile suddividerli in un set di addestramento e in uno di validazione
# poiche` il risultato non permetterebbe di valutare affidabilmente il modello.
# In questo caso viene utilizzata la K-FOLD CROSS-VALIDATION: consiste nel dividere i dati disponibili in K partizioni (tipicamente K = 4 o K = 5),
# istanziare K modelli identici e addestrare ognuno di questi su K - 1 partizioni mentre si va a valutare il modello sulla partizione rimanente
def kfold_cross_validation():
    global average_mae_history

    k = 4
    num_val_samples = len(train_data) // k
    num_epochs = 500
    all_mae_histories = []

    for i in range(k):
        print("[+] Processando Gruppo #", i)
        # Preparazione dati di validazione dalla partizione #K
        val_data = train_data[ i * num_val_samples : (i + 1) * num_val_samples]
        val_targets = train_targets[i * num_val_samples : (i + 1) * num_val_samples]

        # Prepara i dati d'addestramento da tutte le altre partizioni
        partial_train_data = np.concatenate(
            [train_data[ :i * num_val_samples],
            train_data[(i + 1) * num_val_samples : ]],
            axis = 0
        )
        partial_train_targets = np.concatenate(
            [train_targets[ : i * num_val_samples],
            train_targets[(i + 1) * num_val_samples : ]],
            axis = 0
        )

        model = build_model()
        # Verbose = 0 e` la modalita` silenziosa.
        history = model.fit(
            partial_train_data,
            partial_train_targets,
            epochs = num_epochs,
            batch_size = 1,
            #verbose = 0,
            validation_data = (val_data, val_targets)
        )
        mae_history = history.history["val_mean_absolute_error"]
        all_mae_histories.append(mae_history)

    average_mae_history = [
        np.mean([ x[i] for x in all_mae_histories]) for i in range(num_epochs)
    ]


def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation = 'relu', input_shape = (train_data.shape[1], )))
    model.add(layers.Dense(64, activation = 'relu'))
    model.add(layers.Dense(1))
    # LOSS FUNCTION 'mse'   =>  MEAN SQUARED ERROR, il quadrato della differenza tra le predizioni e i bersagli.
    # METRICS 'mae'         =>  MEAN ABSOLUTE ERROR (MAE), valore assoluto della differenza tra le predizioni e i bersagli.
    model.compile(
        optimizer = "rmsprop",
        loss = "mse",
        metrics = ["mae"]
    )
    return model

# Normalizzazione dei dati
def data_normalization():
    global train_data,  test_data
    # I valori in "_targets" hanno range di valori molto diversi, in questo caso viene effettuata una FEATURE-WISE NORMALIZATION:
    # per ogni caratteristica nei dati di input (una colonna nella matrice d'input) si SOTTRAE la MEDIA della caratteristica
    # e si DIVIDE per la DEVIAZIONE STANDARD, in questo modo la caratteristica e` centrata intorno allo 0 ed ha una DEVIAZIONE STANDARD UNITARIA.
    mean = train_data.mean(axis = 0)
    train_data -= mean
    std = train_data.std(axis = 0)
    train_data /= std

    test_data -= mean
    test_data /= std

# Inizializza i Sets di Dati
def init_dataset():
    global train_data, train_targets, test_data, test_targets
    (train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

    print("[+] train_data.shape:\n%s\n" % (train_data.shape, ))
    print("[+] test_data.shape:\n%s\n" % (test_data.shape, ))
    print("[+] len(train_data):\n%s\n" % len(train_data))
    print("[+] len(train_targets):\n%s\n" % len(train_targets))
    print("[+] len(test_data):\n%s\n" % len(test_data))
    print("[+] len(test_targets):\n%s\n" % len(test_targets))
    print("[+] train_targets:\n%s\n" % train_targets)
    print("\n")



def main():
    # Setto per usare CPU in questo caso poiche` la rete neurale e` piccola e la grandezza dei batch e` piccola. In questo caso la CPU e` piu` veloce.
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    print("\n## START ##\n")
    init_dataset()
    data_normalization()
    kfold_cross_validation()
    plot_validation_scores()
    plot_validation_scores_two()
    train_final_model()

if __name__ == "__main__":
    main()
