from keras.applications import VGG16
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np

# Funzione per generare visualizzazioni di filtro
def generate_pattern(layer_name, filter_index, size=150):
    global model
    # Costruisce una Loss Function (Funzione Perdita) che massimizza l'attivazione dell'n-esimo filtro
    # dello strato preso in considerazione
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])

    # Calcola il gradiente dell'immagine di input rispetto a questa perdita
    grads = K.gradients(loss, model.input)[0]

    # Trucco per la Normalizzazione del Gradiente (Gradient-Normalization)
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # Trucco per la Normalizzazione: normalizza il gradiente
    iterate = K.function([model.input], [loss, grads])

    # Parte da un'immagine grigia con un po' di rumore
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.

    # Esegue l'ascesa del gradiente per 40 steps
    step = 1
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
    
    img = input_img_data[0]
    return deprocess_image(img)

# Funzione Utility per convertire un tensore in un'immagine valida
def deprocess_image(x):
    # Normalizza il tensore: centra in 0, assicura che 'std' sia 0.1 (std = Standard Deviation, Deviazione Standard)
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    x += 0.5
    x = np.clip(x, 0, 1)

    # Converte ad un array RGB
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def main():
    global model
    # Definire il tensore perdita per la visualizzazione dei filtri
    model = VGG16(weights='imagenet', include_top=False)
    layer_name = 'block3_conv1'
    filter_index = 0

    plt.title("block3_conv1")
    plt.imshow(generate_pattern('block3_conv1', 0))
    plt.grid(False)
    plt.show()

    # Generare una griglia di tutti i modelli di risposta del filtro in uno strato
    layer_name = 'block1_conv1'
    size = 64
    margin = 5

    # Immagine vuota (nera) per memorizzare i risultati
    results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))

    # Itera sulle righe della griglia dei risultati
    for i in range(8):
        # Itera sulle colonne della griglia dei risultati
        for j in range(8):
            # Genera il pattern per il filtro 'i + (j * 8)' in 'layer_name'
            filter_img = generate_pattern(layer_name, i + (j * 8), size=size)

            horizontal_start = i * size + i * margin
            horizontal_end = horizontal_start + size
            vertical_start = j * size + j * margin
            vertical_end = vertical_start + size
            results[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = filter_img

    plt.figure(figsize=(20, 20))
    plt.imshow(results)
    plt.title("Results Image")
    plt.show()


if __name__ == "__main__":
    main()
