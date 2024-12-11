# Semplice Convnet per classificazione "dogs vs. cats" #

import os
import matplotlib.pyplot as plt
from keras import optimizers
from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator

def main():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation = 'relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation = 'relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation = 'relu'))
    model.add(layers.Dense(1, activation = 'sigmoid'))

    model.compile(
        loss = 'binary_crossentropy',
        optimizer = optimizers.RMSprop(lr = 1e-4),
        metrics = ['acc']
    )

    # Definizione directories dove sono contenute le immagini di addestramento e convalida
    base_dir = "/home/z3r0/Desktop/All/[Exercises]/[Deep_Learning]/dogs_vs_cats_classification/datasets"
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    # Preprocessazione dei dati (trasformando le immagini da JPEG a tensori preprocessati a virgola mobile) usando la classe ImageDataGenerator
    train_datagen = ImageDataGenerator(rescale = 1./255)
    test_datagen = ImageDataGenerator(rescale = 1./255)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size = (150, 150),
        batch_size = 20,
        class_mode = 'binary'
    )
    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size = (150, 150),
        batch_size = 20,
        class_mode = 'binary'
    )

    # Adattamento del modello usando un generatore di batch
    history = model.fit_generator(
        train_generator,
        steps_per_epoch = 100,
        epochs = 30,
        validation_data = validation_generator,
        validation_steps = 50
    )
    model.save('Small_Convnet.h5')

    # Grafico delle curve della Perdita e della Precisione durante l'addestramento
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label = 'Training Acc')
    plt.plot(epochs, val_acc, 'b', label = 'Validation Acc')
    plt.title('Training & Validation Accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label = 'Training Loss')
    plt.plot(epochs, val_loss, 'b', label = 'Validation Loss')
    plt.title('Training & Validation Loss')
    plt.legend()

    plt.show()


if __name__ == '__main__':
    main()
