from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, SimpleRNN, LSTM
import matplotlib.pyplot as plt
import sys

## Tracciare graficamente i risultati
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

def main():
    ## 200 - Preparare i dati di IMDB
    # Numero di parole da considerare come caratteristiche
    max_features = 10000
    # Taglia i testi dopo 'maxlen' parole (tra le 'max_features' parole comuni)
    maxlen = 500
    batch_size = 32

    # len(input_train) = 25000      len(input_test) = 25000
    (input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)

    # Padding dei dati di input
    input_train = sequence.pad_sequences(input_train, maxlen=maxlen)    # input_train.shape = (25000, 500)
    input_test = sequence.pad_sequences(input_test, maxlen=maxlen)      # input_test.shape = (25000, 500)

    while(True):
        print("1: Model_1 Without LSTM\t\t2: Model_2 With LSTM\t\t3: Exit")
        try:
            user_input = raw_input("> ")
            choiche = int(user_input)
        except NameError:
            print("[!] NameError Exception: %s" % NameError.__cause__)
            choiche = 0
        except ValueError:
            print("[!] ValueError Exception: %s" % ValueError.__cause__)
            choiche = 0

        if choiche == 1:
            ## 200 - Addestrare il modello con strati Embedding e SimpleRNN
            model_1 = Sequential()
            model_1.add(Embedding(max_features, 32))
            model_1.add(SimpleRNN(32))
            model_1.add(Dense(1, activation="sigmoid"))

            model_1.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])
            history = model_1.fit(input_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

            ## RISULTATI ##
            # loss: 0.0178 - acc: 0.9949 - val_loss: 0.7619 - val_acc: 0.7808
            # Come e` possibile vedere dai valori e dal grafico il modello va in Overfitting tra l'epoca 4/5

            ## 200 - Tracciare i risultati
            plot_results(history.history)

            # results = model.evaluate(input_test, y_test)
            # Results = [test_loss, test_acc] = [0.79460195625305174, 0.77059999999999995]
        
        elif choiche == 2:
            ## 205 - Utilizzare lo strato LSTM in Keras
            model_2 = Sequential()
            model_2.add(Embedding(max_features, 32))
            model_2.add(LSTM(32))
            model_2.add(Dense(1, activation="sigmoid"))

            model_2.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])
            history = model_2.fit(input_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

            ## RISULTATI ##
            # loss: 0.1105 - acc: 0.9627 - val_loss: 0.3575 - val_acc: 0.8828

            plot_results(history.history)

            results = model_2.evaluate(input_test, y_test)
            print("[+] Results = %s" % results)

        elif choiche == 3:
            sys.exit()

        else:
            print("[-] Wrong Input!")


if __name__ == "__main__":
    main()
