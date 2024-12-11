# Fine-Tuning by freezing the VGG16 Convolutional Base #

import os
import matplotlib.pyplot as plt
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
from keras import optimizers

# Funzione per regolare le curvature del grafico
def smooth_curve(points, factor = 0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor * point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

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

    conv_base.trainable = True

    # Si congelano tutti gli strati prima nella base fino allo strato specificato da cui si settano tutti gli strati, che sono
    # piu` in alto nella Base Convoluzionale e quindi codificano caratteristiche piu` specifiche, come addestrabili
    set_trainable = False
    for layer in conv_base.layers:
        if layer.name == 'block5_conv1':
            set_trainable = True
        layer.trainable = set_trainable

    # Fine-Tuning del modello
    model.compile(
        loss = 'binary_crossentropy',
        optimizer = optimizers.RMSprop(lr = 1e-5),
        metrics = ['acc']
    )
    history = model.fit_generator(
        train_generator,
        steps_per_epoch = 100,
        epochs = 100,
        validation_data = validation_generator,
        validation_steps = 50
    )

    # Grafici dei risultati
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, smooth_curve(acc), 'bo', label = 'Smoothed Training Acc')
    plt.plot(epochs, smooth_curve(val_acc), 'b', label = 'Smoothed Validation Acc')
    plt.title('Training & Validation Accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, smooth_curve(loss), 'bo', label = 'Smoothed Training Loss')
    plt.plot(epochs, smooth_curve(val_loss), 'b', label = 'Smoothed Validation Loss')
    plt.title('Training & Validation Loss')
    plt.legend()

    plt.show()

    # Convalida il modello sui dati di test
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size = (150, 150),
        batch_size = 20,
        class_mode = 'binary'
    )
    test_loss, test_acc = model.evaluate_generator(test_generator, steps = 50)
    print("[+] Test acc: '%s'" % test_acc)

if __name__ == '__main__':
    main()
