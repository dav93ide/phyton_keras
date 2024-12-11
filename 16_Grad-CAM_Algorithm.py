from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import cv2

def main():
    ## 173 - Caricare la rete VGG16 con pesi gia` addestrati
    model = VGG16(weights='imagenet')

    ## 174 - Processare un'immagine di input per VGG16
    img_path = "/home/z3r0/Desktop/All/Exercises/Deep_Learning/images/African_Elephants.jpeg"

    # Immagine "Python Imaging Library" (PIL) di dimensioni 224x224
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # Preprocessa il batch (questo effettua la normalizzazione del colore del canale)
    x = preprocess_input(x)
    
    ## 174 - Preparare l'algoritmo Grad-CAM
    african_elephant_output = model.output[:, 386]

    # Mappa delle caratteristiche di output dello strato "block5_conv3", l'ultimo strato di convoluzione in VGG16
    last_conv_layer = model.get_layer("block5_conv3")

    # Gradiente della classe "African Elephant" rispetto alla mappa delle caratteristiche di output di "block5_conv3"
    grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]

    # Vettore di forma (512, ), in cui ogni entrata corrisponde all'intensita` media del gradiente su uno
    # specifico canale della mappa delle caratteristiche
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    # Permette di accedere ai valori delle quantita` appena definite: "pooled_grads" e la mappa delle caratteristiche
    # di output di "block5_conv3", data un'immagine campione
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

    # Valori di queste due quantita`, come arrays Numpy, data l'immagine campione di due elefanti
    pooled_grads_value, conv_layer_output_value = iterate([x])

    # Moltiplica ogni canale nell'array della mappa delle caratteristiche dell' "importanza del canale" rispetto alla classe "elephant"
    for i in range(512):
        conv_layer_output_value[:, :, 1] *= pooled_grads_value[i]

    # La media del canale della mappa delle caratteristiche risultate e` la heatmap (mappa di calore) dell'attivazione della classe
    heatmap = np.mean(conv_layer_output_value, axis=-1)

    ## 175 - Post-processazione dell'heatmap (Normalizzazione tra 0 e 1 dei valori dell'heatmap)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    plt.matshow(heatmap)

    ## 176 - Sovrapporre l'heatmap sull'immagine originale (Utilizzando OpenCV)
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))         # Ridimensione l'heatmap alle dimensioni dell'immagine originale
    heatmap = np.uint8(255 * heatmap)                                   # Converte l'heatmap in RGB
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img                              # 0.4 -> fattore d'intensita` della heatmap
    cv2.imwrite("/home/z3r0/Desktop/All/Exercises/Deep_Learning/images/elephant_cam.jpg", superimposed_img)


if __name__ == "__main__":
    main()
