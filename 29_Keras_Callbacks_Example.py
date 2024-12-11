import keras
import numpy as np

## 250 - Esempi di callbacks
def dummy_callbacks_example():
    callbacks_list = [
        # Interrompe l'addestramento quando terminano i miglioramenti
        keras.callbacks.EarlyStopping(
            monitor="acc",          # Monitora l'accuratezza di convalida del modello
            patience=1              # Interrompe l'addestramento dopo che l'accuratezza ha smesso
        ),                          # di migliorare per piu` di 1 epoca
        # Salva i pesi correnti dopo ogni epoca
        keras.callbacks.ModelCheckpoint(
            filepath="keras_callback_model.h5",
            monitor="val_loss",     # Il file del modello non verra` sovrascritto fino a quando
            save_best_only=True     # 'val_loss' non sara` migliorata
        ),
        # Riduce il rateo di apprendimento quando una metrica definita smette di migliorare
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",     # Monitora la perdita di convalida del modello
            factor=0.1,             # Divide il rateo di apprendimento di 10 quando triggerato
            patience=10             # La callback viene innescata dopo 10 epoche senza miglioramenti
        )
    ]
    # Poiche` viene monitorata l'accuratezza essa deve essere parte delle metriche del modello 
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])

    # Poiche` la callback monitorera` la perdita di convalida e l'accuratezza di convalida
    # e` necessario passare 'validation_data' alla chiamata.
    model.fit(x, y, batch_size=32, callbacks=callbacks_list, validation_data=(x_val, y_val))

## 251 - Definizione di una Callback personalizzata: la callback salva su disco le attiviazioni
# (come arrays Numpy) di ogni strato del modello alla fine di ogni epoca, calcolate sul primo 
# campione del set di convalida.
class ActivationLogger(keras.callbacks.Callback):
    # Chiamato dal modello genitore prima dell'addestramento per informare la callback
    # di quale modello la sta chiamando.
    def set_model(self, model):
        self.model = model
        layers_outputs = [layer.output for layer in model.layers]
        # Istanza del modello che ritorna le attivazioni di ogni strato
        self.activation_model = keras.models.Model(model.input, layers_outputs)

    def on_epoch_end(self, epoch, logs=None):
        if self.validation_data is None:
            raise RuntimeError("[!] Requires validation_data.")
        # Ottiene il primo campione di input dei dati di convalida
        validation_sample = self.validation_data[0][0:1]
        activations = self.activations_model.predict(validation_sample)
        # Salva gli arrays su disco
        f = open("activations_at_epoch_" + str(epoch) + ".npz", 'w')
        np.savez(f, activations)
        f.close()

def main():
    # dummy_callbacks_example()


if __name__ == "__main__":
    main()
