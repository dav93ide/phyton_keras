from keras.preprocessing.text import Tokenizer
import numpy as np
import string

## 183 - Codifica Unica a livello di parola con trucco di Hashing
def word_level_one_hot_encoding_hashing_trick():
    samples = ["The cat sat on the mat.", "The dog ate my homework."]
    # Mantiene le parole come un vettore di dimensione 1000. Se si hanno quasi o piu` di 1000 parole
    # si avranno molte Hash Collision il che diminuira` notevolmente la precisione di questo metodo di codifica.
    dimensionality = 1000
    max_length = 10
    results = np.zeros((len(samples), max_length, dimensionality))
    for i, sample in enumerate(samples):
        for j, word in list(enumerate(sample.split()))[:max_length]:
            # Hasha la parola in un indice intero casuale tra 0 e 1000
            index = abs(hash(word)) % dimensionality
            results[i, j, index] = 1

## 183 - Utilizzare Keras per la Codifica Unica a livello di parola
def word_level_keras_one_hot_encoding():
    samples = ["The cat sat on the mat.", "The dog ate my homework."]
    tokenizer = Tokenizer(num_words=10000)
    # Costruisce l'indice delle parole
    tokenizer.fit_on_texts(samples)
    # Trasforma le stringhe in liste di indici interi
    sequences = tokenizer.texts_to_sequences(samples)
    # Si puo` anche ottenere direttamente le rappresentazioni binarie uniche (one-hot = uniche / unidirezionali)
    # Questo Tokenizer supporta anche altre modalita` di vettorizzazione oltre alla Codifica Unica
    one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')
    # Per ottenere l'indice delle parole calcolato precedentemente
    word_index = tokenizer.word_index
    print("[+] Trovati '%s' tokens univoci." % len(word_index))         # I tokens univoci trovati sono: 9.

## 182 - Codifica Unica a livello di parola
def word_level_one_hot_encoding():
    samples = ["The cat sat on the mat.", "The dog ate my homework."]
    token_index = {}

    for sample in samples:
        for word in sample.split():
            if word not in token_index:
                # Assegna un indice univoco ad ogni parola univoca. Da notare che l'indice 0 non viene attribuito a niente.
                # Contenuto di 'token_index': 
                # {'on': 4, 'ate': 8, 'homework.': 10, 'dog': 7, 'cat': 2, 'mat.': 6, 'The': 1, 'my': 9, 'the': 5, 'sat': 3}
                token_index[word] = len(token_index) + 1

    max_length = 10
    # Genera una matrice 3D di forma: 
    # (num campioni, num parole da considerare per campione, lunghezza della parola piu` lunga + 1)
    results = np.zeros(shape=(len(samples), max_length, max(token_index.values()) + 1))

    for i, sample in enumerate(samples):
        for j, word in list(enumerate(sample.split()))[:max_length]:
            index = token_index.get(word)
            results[i, j, index] = 1

## 182 - Codifica Unica a livello di carattere
def char_level_one_hot_encoding():
    samples = ["The cat sat on the mat.", "The dog ate my homework."]
    # Prende tutti i caratteri ASCII stampabili
    characters = string.printable

    ###
    # range(1, len(characters) + 1):
    #       Ritorna una sequenza di int da '1' a 'len(characters) + 1'
    ###
    # zip(range(1, len(characters) + 1), characters): 
    #       Ritorna un oggetto zip costituito da una sequenza di gruppi di 'n' elementi, con 'n = num args passati a zip', 
    #       formati dagli i-esimi elementi di ciascuna sequenza passatagli come argomenti.
    #       es: [(arg1[0], arg2[0], arg3[0] ...), (arg1[1], arg2[1], arg3[1]...)]
    ###
    # dict(zip(range(1, len(characters) + 1), characters)):
    #       Ritorna un dizionario creato dalla sequenza di coppie passatagli come argomento.
    ###
    token_index = dict(zip(range(1, len(characters) + 1), characters))

    max_length = 50
    results = np.zeros((len(samples), max_length, max(token_index.keys()) + 1))

    for i, sample in enumerate(samples):
        for j, character in enumerate(sample):
            index = token_index.get(character)
            results[i, j, index] = 1


def main():
    word_level_one_hot_encoding()
    char_level_one_hot_encoding()
    word_level_keras_one_hot_encoding()
    word_level_one_hot_encoding_hashing_trick()


if __name__ == "__main__":
    main()
