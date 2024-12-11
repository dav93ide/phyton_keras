# Implementazione di un Esempio di Modello Multi-Output utilizzando l'API Funzionale per la previsione di
# determinate proprieta` di un utente a partire da un post su un social network.

from keras.models import Model
from keras import layers
from keras import Input
import numpy as np

## 240 - Implementazione di un modello con 3 outputs utilizzando l'API funzionale
def three_outputs_model():
    vocabulary_size = 50000
    num_income_groups = 10

    posts_input = Input(shape=(None, ), dtype='int32', name='posts')
    embedded_posts = layers.Embedding(256, vocabulary_size)(posts_input)
    x = layers.Conv1D(128, 5, activation="relu")(embedded_posts)
    x = layers.MaxPooling1D(5)(x)
    x = layers.Conv1D(256, 5, activation="relu")(x)
    x = layers.Conv1D(256, 5, activation="relu")(x)
    x = layers.MaxPooling1D(5)(x)
    x = layers.Conv1D(256, 5, activation="relu")(x)
    x = layers.Conv1D(256, 5, activation="relu")(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)

    # Notare che agli output sono stati dati dei nomi
    age_prediction = layers.Dense(1, name="age")(x)
    income_prediction = layers.Dense(num_income_groups, activation="softmax", name="income")(x)
    gender_prediction = layers.Dense(1, activation="sigmoid", name="gender")(x)

    model = Model(posts_input, [age_prediction, income_prediction, gender_prediction])

    ## 241 - Opzioni di compilazione di un Modello Multi-Output: perdite multiple
    #   model.compile(optimizer="rmsprop", loss=["mse", "categorical_crossentropy", "binary_crossentropy"])

    # Equivalente, ma possibile solo se sono stati dati dei nomi alle varie heads (outputs) del modello
    #   model.compile(optimizer="rmsprop", loss={"age":"mse", "income":"categorical_crossentropy", "gender":"binary_crossentropy"})

    ## 241 - Opzioni di compilazione di un Modello Multi-Output: ponderazione della perdita
    #   model.compile(optimizer="rmsprop", loss=["mse", "categorical_crossentropy", "binary_crossentropy"], loss_weights=[0.25, 1., 10.])

    # Equivalente, ma possibile solo se sono stati dati dei nomi alle varie heads del modello
    model.compile(optimizer="rmsprop", loss={"age":"mse", "income":"categorical_crossentropy", "gender":"binary_crossentropy"}, loss_weights={"age":0.25, "income":1., "gender":10.})

    ## 242 - Fornire dati ad un modello multi-output
    # "age_targets", "income_targets" e "gender_targets" sono degli arrays Numpy
    #   model.fit(posts, [age_targets, income_targets, gender_targets], epochs=10, batch_size=64)

    # Equivalente, possibile solo se sono stati dati dei nomi ai vari outputs del modello
    model.fit(posts, {"age":age_targets, "income":income_targets, "gender":gender_targets}, epochs=10, batch_size=64)
    

def main():
    three_outputs_model()

if __name__ == "__main__":
    main()
