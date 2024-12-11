from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Traccia graficamente Valori di Precisione e Valori di Perdita
def plot_results(history):
    acc = history.history["acc"]
    val_acc = history.history["val_acc"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, "bo", label="Training Acc")
    plt.plot(epochs, val_acc, "b", label="Validation Acc")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    
    plt.figure()

    plt.plot(epochs, loss, "bo", label="Training Loss")
    plt.plot(epochs, val_loss, "b", label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.show()

def main():
    ## 189 - Processare le etichette dei dati raw di IMDB
    imdb_dir = "/home/z3r0/Desktop/All/Exercises/Deep_Learning/imdb_text_recognization_dataset/aclImdb"
    train_dir = os.path.join(imdb_dir, "train")

    labels = []
    texts = []

    for label_type in ['neg', 'pos']:
        dir_name = os.path.join(train_dir, label_type)
        for fname in os.listdir(dir_name):
            if fname[-4:] == ".txt":
                f = open(os.path.join(dir_name, fname))
                texts.append(f.read())
                f.close()
                if label_type == "neg":
                    labels.append(0)
                else:
                    labels.append(1)

    ## 189 - Tokenizzare il testo dei dati raw di IMDB (set di addestramento e set di convalida)
    maxlen = 100                        # Taglia le recensioni dopo 100 parole
    training_samples = 200              # Addestra su 200 campioni
    validation_samples = 10000          # Valida su 10000 campioni
    max_words = 10000                   # Considera solo le prime 10000 parole del dataset

    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    print("[+] Trovati %s tokens univoci." % (len(word_index)))

    data = pad_sequences(sequences, maxlen=maxlen)

    labels = np.asarray(labels)
    print("[+] Forma del tensore dati: %s" % (data.shape, ))                # Shape: (25000, 100)
    print("[+] Forma del tensore etichetta: %s" % (labels.shape, ))         # Shape: (25000,)

    # Divide i dati in set di addestramento e set di validazone, ma prima mescola i dati.
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]

    x_train = data[:training_samples]
    y_train = labels[:training_samples]
    x_val = data[training_samples: training_samples + validation_samples]
    y_val = labels[training_samples: training_samples + validation_samples]

    ## 194 - Tokenizzare i dati del set di test
    test_dir = os.path.join(imdb_dir, "test")
    labels = []
    texts = []

    for label_type in ["neg", "pos"]:
        dir_name = os.path.join(test_dir, label_type)
        for fname in sorted(os.listdir(dir_name)):
            if fname[-4:] == ".txt":
                f = open(os.path.join(dir_name, fname))
                texts.append(f.read())
                f.close()
                if label_type == "neg":
                    labels.append(0)
                else:
                    labels.append(1)

    sequences = tokenizer.texts_to_sequences(texts)
    x_test = pad_sequences(sequences, maxlen=maxlen)
    y_test = np.asarray(labels)

    ## 190 - Parsare il file di immersione di parole GloVe
    glove_dir = "/home/z3r0/Desktop/All/Exercises/Deep_Learning/GloVe_Word_Embedding/"

    embeddings_index = {}
    f = open(os.path.join(glove_dir, "glove.6B.100d.txt"))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embeddings_index[word] = coefs
    f.close()
    print("[+] Trovati %s vettori di parole." % (len(embeddings_index)))    # Vettori: 400000

    ## 191 - Preparare la matrice di immersioni di parole di GloVe
    embedding_dim = 100
    embedding_matrix = np.zeros((max_words, embedding_dim))

    for word, i in word_index.items():
        if i < max_words:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # Le parole non trovate nell'indice di incorporamento saranno tutte zero.
                embedding_matrix[i] = embedding_vector

    while(True):
        print("[1] Model_1: With Word Embeddings\t\t[2] Model_2: Without Word Embeddings\t\t[3] Exit")
        try:            
            user_input = input("> ")
            choiche = int(user_input)
        except NameError:
            print("[!] NameError Exception: %s" % NameError.__cause__)
            choiche = 0
        except ValueError:
            print("[!] ValueError Exception: %s" % ValueError.__cause__)
            choiche = 0
        if choiche == 1:
            ## 190 - Definizione del Modello
            model_1 = Sequential()
            model_1.add(Embedding(max_words, embedding_dim, input_length=maxlen))
            model_1.add(Flatten())
            model_1.add(Dense(32, activation="relu"))
            model_1.add(Dense(1, activation="sigmoid"))
            model_1.summary()                                 # Print output in fondo

            ## 190 - Caricare le immersioni di parole pre-addestrate dentro lo Strato di Incorporamento
            model_1.layers[0].set_weights([embedding_matrix])
            model_1.layers[0].trainable = False

            ## 192 - Addestramento e convalida
            model_1.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])
            history = model_1.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val))
            # model_1.save_weights("19_Pre_Trained_GloVe_Model_1.h5")       # Salva il modello addestrato
            # model_1.load_weights("19_Pre_Trained_GloVe_Model_1.h5")       # Ricarica il modello salvato

            ## RISULTATI ##
            # loss: 0.0356 - acc: 1.0000 - val_loss: 0.7863 - val_acc: 0.5838
            # Come e` possibile notare dai valori riportati e dai grafici, il modello va subito in Overfitting.

            ## 192 - Tracciare graficamente i risultati
            plot_results(history)

            ## 195 - Valutare il modello sul set di test
            results = model_1.evaluate(x_test, y_test)
            print("[+] Risultati Modello '1' su Set di Test: %s" % results)
            # Results = [test_loss, test_acc] = [1.1356588978266715, 0.52559999999999996]

        elif choiche == 2:
            ## 193 - Addestrare lo stesso modello senza immersioni di parole pre-addestrate
            model_2 = Sequential()
            model_2.add(Embedding(max_words, embedding_dim, input_length=maxlen))
            model_2.add(Flatten())
            model_2.add(Dense(32, activation="relu"))
            model_2.add(Dense(1, activation="sigmoid"))
            model_2.summary()

            model_2.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])
            history = model_2.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=[x_val, y_val])

            ## RISULTATI ##
            # loss: 0.0030 - acc: 1.0000 - val_loss: 0.7896 - val_acc: 0.5156

            plot_results(history)

            results = model2.evaluate(x_test, y_test)
            print("[+] Risultati Modello '2' su Set di Test: %s" % results)
            # Results = [test_loss, test_acc] = [0.7905815578269958, 0.51339999999999997]
        
        elif choiche == 3:
            sys.exit()

        else:
            print("[-] Wrong Input!")


if __name__ == "__main__":
    main()



'''
## Print delle informazioni sul Modello_1 e sul Modello_2 ##
.___________________________.
|       model_1.summary()     |
|___________________________|____________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, 100, 100)          1000000   
_________________________________________________________________
flatten_1 (Flatten)          (None, 10000)             0         
_________________________________________________________________
dense_1 (Dense)              (None, 32)                320032    
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 33        
=================================================================
Total params: 1,320,065
Trainable params: 1,320,065
Non-trainable params: 0
_________________________________________________________________



.___________________________.
|       model_2.summary()     |
|___________________________|____________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, 100, 100)          1000000   
_________________________________________________________________
flatten_1 (Flatten)          (None, 10000)             0         
_________________________________________________________________
dense_1 (Dense)              (None, 32)                320032    
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 33        
=================================================================
Total params: 1,320,065
Trainable params: 1,320,065
Non-trainable params: 0
_________________________________________________________________
'''
