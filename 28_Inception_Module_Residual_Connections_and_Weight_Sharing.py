# Grafi Aciclici Diretti di Strati (Moduli Inception & Connessioni Residue) # 
# Condivisione dei pesi dello strato # 
# Modello Siamese di visione  (Base Convoluzionale condivisa) # 

from keras import layers
from keras import Input
from keras import applications
from keras.models import Model

# L'esempio assume l'esistenza di un 'x' tensore 4D di input
def inception_modules_example():
    # Ogni ramo ha lo stesso valore di "stride" (passo lungo) (2), il che e` necessario per
    # mantenere con la stessa dimensione tutti i rami di outputs al fine di poterli
    # successivamente concatenare 
    branch_a = layers.Conv2D(128, 1, activation="relu", strides=2)(x)
    
    # In questo ramo i "passi lunghi" avvengono nello strato di convoluzione spaziale
    branch_b = layers.Conv2D(128, 1, activation="relu")(x)
    branch_b = layers.Conv2D(128, 3, activation="relu", strides=2)(brach_b)

    # In questo ramo i "passi lunghi" avvengono nello strato di pooling medio
    branch_c = layers.AveragePoolin2D(3, strides=2)(x)
    branch_c = layers.Conv2D(128, 3, activation="relu")(branch_c)

    branch_d = layers.Conv2D(128, 1, activation="relu")(x)
    branch_d = layers.Conv2D(128, 3, activation="relu")(branch_d)
    branch_d = laeyers.Conv2D(128, 3, activation="relu", strides=2)(branch_d)

    # Concatena i rami di outputs per ottenere l'output del modulo
    output = layers.concatenate([branch_a, branch_b, branch_c, branch_d], axis=-1)

# L'esempio assume l'esistenza di un 'x' tensore 4D di input
def residual_connections_example():
    # La seguente e` un'implementazione di una Connessione Residua quando le dimensioni
    # delle mappe delle caratteristiche sono le stesse.
    # Applica una trasformazione a 'x'
    y = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    y = layers.Conv2D(128, 3, activation="relu", padding="same")(y)
    y = layers.Conv2D(128, 3, activation="relu", padding="same")(y)

    # Aggiunge la 'x' originale alle caratteristiche di output
    y = layres.add([y, x])

    # La seguente e` un'implementazione di una Connessione Residua quando le dimensioni
    # delle mappe delle caratteristiche non sono le stesse.
    y = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    y = layers.Conv2D(128, 3, activation="relu", padding="same")(y)
    y = layers.MaxPooling2D(2, strides=2)(y)

    # Utilizza una convoluzione 1x1 per ridimensionare linearmente il tensore 'x' originale
    # alla stessa forma di 'y'
    residual = layers.Conv2D(128, 1, strides=2, padding="same")(x)

     # Aggiunge il tensore 'residual' indietro alle caratteristiche di output
     layers.add([y, residual])

## 247 - Modello d'esempio che tenta di valutare la somiglianza semantica tra 2 sentenze.
# Il modello processa entrambi gli input con un singolo strato LSTM cui rappresentazioni
# apprese (i suoi pesi) sono imparate basandosi su entrambe le sentenze (la somiglianza
# semantica e` una relazione simmetrica, quindi avere 2 modelli non avrebbe senso)
def layer_weight_sharing_example():
     # Istanzia un singolo strato LSTM una sola volta
     lstm = layers.LSTM(32)

     # Costruisce il ramo sinistro del modello: gli inputs sono sequenze di lunghezza 
     # variabile di vettori di dimensione 128
     left_input = Input(shape=(None, 128))
     left_ouput = lstm(left_input)

     # Costruisce il ramo destro del modello: quando si chiama un'istanza esistente di 
     # uno strato vengono riutilizzati i suoi pesi
     right_input = Input(shape=(None, 128))
     right_output = lstm(right_input)

     # Costruisce in cima il classificatore
     merged = layers.concatenate([left_output, right_output], axis=-1)
     predictions = layers.Dense(1, activation="sigmoid")(merged)

     # Istanzia e addestra il modello: quando si addestra un modello di questo genere
     # i pesi dello strato LSTM sono aggiornati basandosi su entrambi gli inputs.
     model = Model([left_input, right_input], predictions)
     model.fit([left_input, right_input], targets)

## 248 - Modello Siamese di visione (Base Convoluzionale Condivisa): l'esempio ipotizza di
# ricevere 2 inputs differenti da due videocamere posta una a sinistra e una a destra
# al fine di aggiungere l'analisi della profondita`.
def siamese_vision_model_example():
     # La base del modello di processazione dell'immagine e la rete Xception (solo Base Convoluzionale)     
     xception_base = applications.Xception(weights=None, include_top=False)

     # Gli inputs sono delle immagini RGB 250x250
     left_input = Input(shape=(250, 250, 3))
     right_input = Input(shape=(250, 250, 3))

     # Chiama lo stesso modello di visione due volte
     left_features = xception_base(left_input)
     right_input = xception_base(right_input)

     # Le caratteristiche combinate contengono informazioni dalla sorgente visiva destra
     # e dalla sorgente visiva sinistra
     merged_features = layers.concatenate([left_features, right_input], axis=-1)

def main():
    inception_modules_example()
    residual_connections_example()
    layer_weight_sharing_example()
    siamese_vision_model_example()

if __name__ == "__main__":
    main()
