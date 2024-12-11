# Visualizzare cio` che una Convnet ha imparato usando il metodo di 'Visualizzazione delle Attivazioni Intermedie' #

from keras import models
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import sys

def main():
    model = load_model('Convnet_with_DataAugmentation_Generators.h5')
    # 160 - Visualizza struttura del modello
    model.summary()

    # 161 - Preprocessare una singola immagine
    img_path = "/home/z3r0/Desktop/All/Exercises/Deep_Learning/dogs_vs_cats_classification/datasets/test/cats/cat.1500.jpg"

    img = image.load_img(img_path, target_size = (150, 150))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis = 0)
    img_tensor /= 255.
    print "[+] Forma di 'img_tensor': ", img_tensor.shape

    plt.title("Base Picture 'cat.1500.jpg'")
    plt.imshow(img_tensor[0])
    plt.show()

    # 162 - Istanziare un modello da un tensore di input e una lista di tensori di output
    layer_outputs = [layer.output for layer in model.layers[:8]]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

    # 163 - Eseguire il modello in modalita` di predizione
    activations = activation_model.predict(img_tensor)
    first_layer_activation = activations[0]
    print("[+] First Activation Layer's Shape: '%s'" % (first_layer_activation.shape, ))

    # Visualizzare il quarto canale
    plt.title("Fourth Channel")
    plt.grid(False)
    plt.imshow(first_layer_activation[0, :, :, 4], cmap="viridis")
    plt.show()

    # Visualizzare il settimo canale
    plt.title("Seventh Channel")
    plt.imshow(first_layer_activation[0, :, :, 7], cmap="viridis")
    plt.grid(False)
    plt.show()

    # 164 - Visualizzare tutti i canali in ogni attivazione intermedia
    layer_names = []
    for layer in model.layers[:8]:
        layer_names.append(layer.name)

    images_per_row = 16

    for layer_name, layer_activation in zip(layer_names, activations):
        n_features = layer_activation.shape[-1]

        size = layer_activation.shape[1]

        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))

        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0, :, :, col * images_per_row + row]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image

        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
        plt.title("Layer = " + layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.show()
        print("[+] Press <Enter> to Show Next Layer or Digit 'exit' and Press <Enter> to Exit Program")
        exit = raw_input("> ")
        if(exit == "exit"):
            sys.exit()


if __name__ == "__main__":
    main()
