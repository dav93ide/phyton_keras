from keras.layers import Embedding
from keras.datasets import imdb
from keras import preprocessing
from keras.models import Sequential
from keras.layers import Flatten, Dense

def main():
    ## 186 - Istanziare uno Strato di Incorporamento
    # Lo Strato Embedding prende almeno 2 argomenti:
    #   1- Il numero di possibili Tokens
    #   2- La dimensionalita` degli Incorporamenti (Embeddings)
    embedding_layer = Embedding(1000, 64)

    ## 187 - Caricare  i dati di IMDB per utilizzarli con uno Strato di Incorporamento
    max_features = 10000        # numero di parole da considerare come caratteristiche
    maxlen = 20                 # numero di parole dopo il quale tagliare il testo
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    # Trasforma le liste di interi caricate sopra in un tensore 2D di forma: (samples, maxlen)
    x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

    ## 187 - Utilizzare uno Strato di Incorporamento e classificare sui dati di IMDB
    model = Sequential()
    # Specifica la massima lunghezza dell'input dello Strato di Incorporamento in modo da poter
    # successivamente appiattire gli inputs integrati. Dopo lo strato di Incorporamento le
    # attivazioni hanno forma (samples, maxlen, 8) 
    model.add(Embedding(10000, 8, input_length=maxlen))
    # Appiattisce il tensore 3D degli incorporamenti in un tensore 2D di forma: (samples, maxlen * 8)
    model.add(Flatten())
    # Aggiunge in cima il classificatore
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])
    model.summary()
    history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

if __name__ == "__main__":
    main()
