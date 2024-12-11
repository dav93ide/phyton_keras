import os
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras import layers
from keras import optimizers
from keras import models

def main():
    conv_base = VGG16(
        weights = 'imagenet',
        include_top = False,
        input_shape = (150, 150, 3)
    )

    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation = 'relu'))
    model.add(layers.Dense(1, activation = 'sigmoid'))

    # Congela (Freeze) la Base Convoluzionale in modo che non venga modificata durante l'addestramento
    # per non perdere le rappresentazioni gia` apprese.
    conv_base.trainable = False

    base_dir = "/home/z3r0/Desktop/All/[Exercises]/[Deep_Learning]/dogs_vs_cats_classification/datasets"
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    test_dir = os.path.join(base_dir, 'test')

    train_datagen = ImageDataGenerator(
        rescale = 1./255,
        rotation_range = 40,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        fill_mode = 'nearest'
    )
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

    model.compile(
        loss = 'binary_crossentropy',
        optimizer = optimizers.RMSprop(lr = 2e-5),
        metrics = ['acc']
    )
    history = model.fit_generator(
        train_generator,
        steps_per_epoch = 100,
        epochs = 30,
        validation_data = validation_generator,
        validation_steps = 50
    )

    # Grafici dei risultati
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

if __name__ == "__main__":
    main()
