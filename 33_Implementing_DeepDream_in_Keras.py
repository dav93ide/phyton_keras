from keras.applications import inception_v3
from keras.preprocessing import image
from keras import backend as K
import numpy as np
import scipy

def eval_loss_and_grads(x):
    outs = fetch_loss_and_grads([x])
    loss_value = outs[0]
    grad_values = outs[1]
    return loss_value, grad_values

# Questa funzione esegue l'Ascesa del Gradiente per un numero di iterazioni.
def gradient_ascent(x, iterations, step, max_loss=None):
    for i in range(iterations):
        loss_value, grad_values = eval_loss_and_grads(x)
        if max_loss is not None and loss_value > max_loss:
            break
        print("[+] Loss value at %s: %s" % (i, loss_value))
        x = x + (step * grad_values)
    return x

## 284 - Funzioni Ausiliarie
def resize_img(img, size):
    img = np.copy(img)
    factors = (1, float(size[0]) / img.shape[1], float(size[1]) / img.shape[2], 1)
    return scipy.ndimage.zoom(img, factors, order=1)

def save_img(img, fname):
    pil_img = deprocess_image(np.copy(img))
    scipy.misc.imsave("DeepDream/" + fname, pil_img)

# Funzione Utility per aprire, ridimensionare e formattare immagini
# in tensori che 'Inception V3' puo` processare.
def preprocess_image(image_path):
    img = image.load_img(image_path)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = inception_v3.preprocess_input(img)
    return img

# Funzione Utility per convertire un tensore in un'immagine valida
def deprocess_image(x):
    if K.image_data_format() == "channels_first":
        x = x.reshape((3, x.shape[2], x.shape[3]))
        x = x.transpose((1, 2, 0))
    else:
        # Disfa la pre-processazione che era stata effettuata da
        # 'inception_v3.preprocess_input'
        x = x.reshape((x.shape[1], x.shape[2], 3))
    x /= 2.
    x += 0.5
    x *= 255.
    x = np.clip(x, 0, 255).astype('uint8')
    return x

## 281 - Caricare il modello pre-addestrato Inception V3
def main():
    global fetch_loss_and_grads
    # Poiche` il modello non verra` addestrato e` necessario disabilitare tutte
    # le operazioni specifiche dell'addestramento
    K.set_learning_phase(0)

    # Costruisce la rete Inception V3 senza base convoluzionale. Il modello
    # sara` caricato con pesi pre-addestrati di ImageNet.
    model = inception_v3.InceptionV3(weights="imagenet", include_top=False)

    ## 281 - Settare la configurazione di DeepDream
    # Dizionario che mappa i nomi degli strati ad un coefficiente che quantifica 
    # quanto lo strato contribuisca alla perdita che si cerchera` di massimizzare.
    # Notare che i nomi degli strati sono hardcoded nell'applicazione built-in 
    # "Inception V3". E` possibile listare tutti i nomi degli strati
    # utilizzando il comando:           "model.summary()"
    layer_contributions = {
        "mixed2": 0.2,
        "mixed3": 3.,
        "mixed4": 2.,
        "mixed5": 1.5
    }

    ## 282 - Definire la perdita da massimizzare
    # Crea un dizionario che mappa i nomi degli strati a istanze "strato".
    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    # La perdita verra` definita aggiungendo i contributi degli strati alla
    # seguente variabile scalare.
    loss = K.variable(0.)

    for layer_name in layer_contributions:
        coeff = layer_contributions[layer_name]
        # Ottiene l'output dello strato
        activation = layer_dict[layer_name].output 
        scaling = K.prod(K.cast(K.shape(activation), "float32"))
        
        # Aggiunge la norma L2 delle caratteristiche di uno strato alla perdita 
        # (loss). Si evitano gli artefatti dei bordi coinvolgendo nella perdita
        # solo pixels che non fanno parte del bordo.
        loss = loss + (coeff * K.sum(K.square(activation[:, 2: -2, 2: -2, :])) / scaling)

    ## 282 - Processo di Ascesa del Gradiente
    # Questo tensore mantiene l'immagine generata: il 'sogno'.
    dream = model.input

    # Calcola i gradienti del 'sogno' rispetto alla perdita
    grads = K.gradients(loss, dream)[0]

    # Normalizza i gradienti ( -> Stratagemma Importante! <- )
    grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)

    # Setta una funzione di Keras per recuperare il valore della perdita e
    # dei gradienti, data un'immagine di input.
    outputs = [loss, grads]
    fetch_loss_and_grads = K.function([dream], outputs)

    ## 283 - Eseguire l'Ascesa del Gradiente su differenti scale successive
    # Giocare con i seguenti Iperparametri permettera` di ottenere nuovi effetti:
    step = 0.01         #-> Dimensione step di ascesa del gradiente.
    num_octave = 3      #-> Numero di scale su cui eseguire l'ascesa del gradiente.
    octave_scale = 1.4  #-> Rapporto delle dimensioni tra le scale.
    
    # Numero di steps di ascesa da eseguire ad ogni scala.
    iterations = 20
    # Se la perdita cresce piu` di 10 il processo di ascesa del
    # gradiente viene interrotto per evitare brutti artefatti.
    max_loss = 10.      
    # Percorso dell'immagine da utilizzare.
    base_image_path = "/home/z3r0/Desktop/DeepDreamTestPicture.jpg"
    # Carica l'immagine di base in un array Numpy
    img = preprocess_image(base_image_path)

    # Prepara una lista di 'tuple di forma' definendo le differenti scale 
    # su cui eseguire l'ascesa del gradiente.
    original_shape = img.shape[1:3]
    successive_shapes = [original_shape]
    for i in range(1, num_octave):
        shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
        successive_shapes.append(shape)

    # Rovescia la lista di forme rendendole in ordine crescente
    successive_shapes = successive_shapes[::-1]
    # Ridimensiona alla scala piu` piccola l'array Numpy dell'immagine
    original_img = np.copy(img)
    shrunk_original_img = resize_img(img, successive_shapes[0])

    for shape in successive_shapes:
        print("[+] Processing image shape: %s" % (shape,))
        # Scala l'immagine 'dream'
        img = resize_img(img, shape)
        # Esegue l'Ascesa del Gradiente alterando 'dream'
        img = gradient_ascent(img, iterations=iterations, step=step, max_loss=max_loss)
        # Ingrandisce la versione piu` piccola dell'immagine originale: essa sara` pixellata.
        upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
        # Calcola la versione alta-qualita` dell'immagine originale a questa dimensione.
        same_size_original = resize_img(original_img, shape)
        # La differenza tra i due e` il dettaglio che e` stato perso ingrandendo
        lost_detail = same_size_original - upscaled_shrunk_original_img

        # Ri-inietta il dettaglio perduto dentro 'dream'
        img = img + lost_detail
        shrunk_original_img = resize_img(original_img, shape)
        save_img(img, fname="dream_at_scale_" + str(shape) + ".png")

    save_img(img, fname="final_dream.png")

if __name__ == "__main__":
    main()
