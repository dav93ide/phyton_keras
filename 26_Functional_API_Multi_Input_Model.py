# Implementazione di un Esempio di  Modello Domanda/Risposta con 2 inputs

from keras.models import Model
from keras import layers
from keras import Input
import numpy as np

## 239 - Fornire dati ad un modello multi-input
def feeding_multi_inputs_model_data(text_size, question_size, answer_size):
    num_samples = 1000
    max_length = 100

    # Genera dei dati Numpy fantocci
    text = np.random.randint(1, text_size, size=(num_samples, max_length))
    question = np.random.randint(1, question_size, size=(num_samples, max_length))
    
    # Le risposte sono a Codifica Unica, non interi
    answers = np.random.randint(0, 1, size=(num_samples, answer_size))

    return text, question, answers

## 239 - Implementazione di un modello domanda/risposta a 2 input utilizzando l'API funzionale.
def two_inputs_question_answering_model():
    text_vocabulary_size = 10000
    question_vocabulary_size = 10000
    answer_vocabulary_size = 500

    # Il testo di input e` una sequenza di interi con lunghezza variabile. 
    # Da notare la possibilita` di nominare gli inputs.
    text_input = Input(shape=(None, ), dtype='int32', name='text')

    # Incorpora gli inputs in una sequenza di vettori di dimensione 64
    embedded_text = layers.Embedding(64, text_vocabulary_size)(text_input)

    # Codifica i vettori in un singolo vettore utilizzando uno strato LSTM
    encoded_text = layers.LSTM(32)(embedded_text)

    # Stesso processo (con differenti istanze degli strati) per la domanda
    question_input = Input(shape=(None, ), dtype='int32', name='question')
    embedded_question = layers.Embedding(32, question_vocabulary_size)(question_input)
    encoded_question = layers.LSTM(16)(embedded_question)

    # Concatena la domanda codificata e il testo codificato
    concatenated = layers.concatenate([encoded_text, encoded_question], axis=-1)

    # Aggiunge in cima un classificatore "softmax"
    answer = layers.Dense(answer_vocabulary_size, activation="softmax")(concatenated)

    # Nell'istanziazione del Modello e` necessario specificare i 2 inputs e l'output.
    model = Model([text_input, question_input], answer)
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["acc"])

    text, question, answers = feeding_multi_inputs_model_data(text_vocabulary_size, question_vocabulary_size, answer_vocabulary_size)
    
    # Fitting (Montaggio) usando una lista di inputs
    model.fit([text, question], answers, epochs=10, batch_size=128)

    # Fitting usando un dizionario di inputs (solo se gli inputs hanno un nome)
    #   model.fit(['text': text, 'question':question], answers, epochs=10, batch_size=128)

def main():
    two_inputs_question_answering_model()

if __name__ == "__main__":
    main()
