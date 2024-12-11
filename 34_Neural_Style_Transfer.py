from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
from keras.preprocessing.image import load_img, img_to_array
from keras.applications import vgg19
from keras import backend as K
import numpy as np
import time

## 289 - Funzioni ausiliarie
def preprocess_image(img_path):
    img = load_img(img_path, target_size=(img_height, img_width))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

def deprocess_image(x):
    # Centra sullo zero rimuovendo il valore di pixels della media da ImageNet.
    # Questo rovescia una trasformazione fatta da "vgg19.preprocess_input"
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # Converte le immagini da "BGR" a "RGB". Anche questo e` parte del rovesciamento
    # di "vgg19.preprocess_input"
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype("uint8")
    return x

## 290 - Perdita di Contenuto (Content Loss)
# Assicura che lo strato superiore della convnet VGG19 abbia 
# un'immagine similare dell'immagine bersaglio e dell'immagine generata
def content_loss(base, combination):
    return K.sum(K.square(combination - base))

## 290 - Perdita di Stile (Style Loss)
# Utilizza una funzione ausiliaria per calcolare la Matrice di Gram di una matrice di input:
# una mappa di correlazioni trovate nella matrice delle caratteristiche originale
def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_height * img_width
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

## 291 - Perdita Totale di Variazione (Total Variation Loss)
# Opera sui pixels dell'immagine di combinazione generata: incoraggia la continuita` spaziale
# nell'immagine generata, cosi` da evitare risultati eccessivamente pixellati.
def total_variation_loss(x):
    a = K.square(
        x[:, :img_height - 1, :img_width - 1, :] -
        x[:, 1:, :img_width - 1, :])
    b = K.square(
        x[:, :img_height - 1, :img_width - 1, :] -
        x[:, :img_height - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

## 292 - Evaluator class
# Questa classe racchiude "fetch_loss_and_grads" in un modo tale da permettere di
# recuperare le perdite e i gradienti tramite due metodi separati, il che e` richiesto
# dall'ottimizzare "SciPy" che si andra` ad utilizzare.
class Evaluator(object):
     
    def __init__(self):
        self.loss_value = None
        self.grad_values = None

    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape((1, img_height, img_width, 3))
        outs = fetch_loss_and_grads([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype("float64")
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

def main():
    global img_width, img_height, fetch_loss_and_grads
    ## 289 - Definire le variabili iniziali
    # Percorso all'immagine che si vuole trasformare
    target_image_path = "images/African_Elephants.jpeg"
    # Percorso all'immagine di riferimento per lo stile
    style_reference_image_path = "images/style_image.jpg"
    # Dimensioni delle immagini generate
    img_width, img_height = load_img(target_image_path).size

    ## 290 - Carica la rete VGG19 pre-addestrata e applica ad essa le tre immagini
    # L'immagine bersaglio e l'immagine di riferimento per lo stile sono statiche
    target_img = K.constant(preprocess_image(target_image_path))
    style_img = K.constant(preprocess_image(style_reference_image_path))

    # Placeholder che conterra` l'immagine generata
    combination_img = K.placeholder((1, img_height, img_width, 3))

    # Combina le 3 immagini in un singolo batch
    input_tensor = K.concatenate([target_img, style_img, combination_img], axis=0)

    # Costruisce la rete VGG19 con il batch di 3 immagini come input.
    # Il modello verra` caricato con i pesi pre-addestrati di ImageNet.
    model = vgg19.VGG19(input_tensor=input_tensor, weights="imagenet", include_top=False)
    print("[+] VGG19 Model Loaded...")

    ## 291 - Definire la Perdita Finale (Final Loss) che si andra` a minimizzare
    # Dizionario che mappa i nomi degli strati ai tensori di attivazione
    outputs_dict = dict([layer.name, layer.output] for layer in model.layers)
    
    # Strato utilizzato per la Perdita di Contenuto (Content Loss)
    content_layer = "block5_conv2"
    # Strati utilizzati per la Perdita di Stile (Style Loss)
    style_layers = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]
    # Pesi nella Media Ponderata dei componenti Perdita
    total_variation_weight = 1e-4
    style_weigth = 1.
    content_weigth = 0.025

    # Aggiunge la Perdita di Contenuto (Content Loss)
    loss = K.variable(0.)       # Si definira` la perdita aggiungendo tutti i componenti a questa variabile scalare
    layer_features = outputs_dict[content_layer]
    target_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss += content_weigth * content_loss(target_image_features, combination_features)

    # Aggiunge un componente perdita di stile per ogni strato bersaglio
    for layer_name in style_layers:
        layer_features = outputs_dict[layer_name]
        style_reference_fetures = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss(style_reference_fetures, combination_features)
        loss += (style_weigth / len(style_layers)) * sl

    # Aggiunge la Perdita Totale di Variazione (Total Variation Loss)
    loss += total_variation_weight * total_variation_loss(combination_img)

    ## 292 - Settare il processo di discesa del gradiente
    # Ottiene il gradiente dell'immagine generata rispetto alla perdita
    grads = K.gradients(loss, combination_img)[0]

    # Funzione per ottenere i valori della perdita corrente e i gradienti correnti
    fetch_loss_and_grads = K.function([combination_img], [loss, grads])

    evaluator = Evaluator()

    ## 293 - Ciclo di trasferimento dello stile
    result_prefix = "images/34_generated_img_at_iteration_%d.png"
    iterations = 20

    # Questo e` lo stato iniziale: l'immagine bersaglio
    x = preprocess_image(target_image_path)
    
    # Si appiattisce l'immagine perche` "scipy.optimize.fmin_l_bfgs_b"
    # puo` processare soltanto vettori piatti.
    x = x.flatten()

    for i in range(iterations):
        print("[+] Starting iteration number: %s" % i)
        start_time = time.time()
        
        # Esegue l'ottimizzazione "L-BFGS" sui pixels dell'immagine generata per minimizzare
        # la perdita di stile neurale. Notare che bisogna passare la funzione che calcola la perdita
        # e la funzione che calcola i gradienti come due argomenti separati.
        x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x, fprime=evaluator.grads, maxfun=20)

        # Salva l'attuale immagine generata
        print("[+] Current loss value: %d" % min_val)
        img = x.copy().reshape((img_height, img_width, 3))
        img = deprocess_image(img)
        fname = (result_prefix % i)
        imsave(fname, img)
        print("[+] Image saved as: \n\t%s\n" % fname)
        end_time = time.time()
        print("[+] Iteration %d completed in \'%d\' seconds" % (i, end_time - start_time))


if __name__ == "__main__":
    main()
