import keras
import random
import sys
import numpy as np
from keras import layers

# 274 - Scaricare e parsare il file di testo iniziale
def download_and_parse_file():
    # Not downloaded yet, got timeout (Errno 110) while downloading.
    path = keras.utils.get_file('text_datasets/nietzsche.txt', origin="https://s3.emazonaws.com/text-datasets/nietzsche.txt")
    text = open(path).read().lower()
    print("[+] Content length: %s" % len(text))

# 275 - Vettorizzare le sequenze di caratteri
def vectorize_sequences():
    # Estrarremo sequenze di 60 caratteri
    maxlen = 60
    # Campioneremo una nuova sequenza ogni 3 caratteri
    step = 3
    # Mantiene le sequenze estratte
    sentences = []
    # Mantiene i bersagli (i carratteri di "azione supplementare"("follow-up"))
    next_chars = []
    
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])

    print("[+] Number of sentences: %s" % len(sentences))

    # Lista di caratteri univoci nel corpo del testo
    chars = sorted(list(set(text)))
    print("[+] Unique Characters: %s" % len(chars))
    # Dizionario che mappa i caratteri univoci al loro indice all'interno della lista "chars"
    char_indices = dict((char, chars.index(char)) for char in chars)
    print("[+] Vectorization...")

    # Codifica Unica dei caratteri in arrays binary (matrici di zeri con 1 nella posizione del carattere)
    # Array a 3D: len(sentences), maxlen, len(chars) con tutti zeri ad eccezione dei punti in cui
    # vi e` un carattere che appare nella 'i-esima' sentenza al 't-esimo' indice in chars e al corrispondente indice
    # in "char_indices".
    x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    # Array a 2D: len(sentences), len(chars) con tutti zeri ad eccezione dei punti in cui e` presente
    # il carattere successivo di ogni singola sentenza.
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(chars):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

## 275 - Modello LSTM a singolo strato per la predizione del carattere successivo
def single_layer_lstm_model():
    model = keras.models.Sequential()
    model.add(layers.LSTM(128, input_shape=(maxlen, len(chars))))
    model.add(layers.Dense(len(chars), activation="softmax"))

    ## 276 - Configurazione della compilazione del modello
    optimizer = keras.optimizers.RMSprop(lr=0.01)
    # Poiche` e` stata utilizzata la Codifica Unica per codificare i bersagli, si utilizzera` 
    # "categorical_crossentropy" come perdita per addestrare il modello.
    model.compile(loss="categorical_crossentropy", optimizer=optimizer)

## 276 - Funzione per campionare il carattere successivo date le predizioni del modello
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

## 270 - Ciclo di Generazione del Testo
def text_generation_loop():
    # Addestra il modello per 60 epoche
    for epoch in range(1, 60):
        print("[+] Epoch Number: %s", % epoch)
        # Adatta il modello per una iterazione sui dati
        model.fit(x, y, batch_size=128, epochs=1)

        # Seleziona casualmente un seme per la generazione del testo
        start_index = random.randint(0, len(text) - maxlen - 1)
        generated_text = text[start_index: start_index + maxlen]
        print("[+] Generating with seed: \'%s\'" % generated_text)
        
        # Prova un range di differenti temperature di campionamento
        for temperature in [0.2, 0.5, 1.0, 1.2]:
            print("[+] Temperature Value: %s" % temperature)
            sys.stdout.write(generated_text)

            # Genera 400 caratteri partendo dal testo seme
            for i in range(400):
                # Codifica in Codifica Unica i caratteri generati fino ad ora
                sampled = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(generated_text):
                    sampled[0, t, char_indices[char]] = 1
                
                # Campiona il carattere successivo
                preds = model.predict(sampled, verbose=0)[0]
                next_index = sample(preds, temperature)
                next_char = chars[next_index]

                generated_text += next_char
                generated_text = generated_text[1:]
                sys.stdout.write(next_char)

def main():
    download_and_parse_file()
    vectorize_sequences()
    single_layer_lstm_model()

if __name__ == "__main__":
    main()
