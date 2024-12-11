import matplotlib.pyplot as plt
from keras import optimizers
from keras import models
from keras import layers
from keras.preprocessing.image import ImageDataGenerator


def main():
    train_dir = "/home/z3r0/Desktop/All/[Exercises]/[Deep_Learning]/dogs_vs_cats_classification/datasets/train"
    validation_dir = "/home/z3r0/Desktop/All/[Exercises]/[Deep_Learning]/dogs_vs_cats_classification/datasets/validation"
    test_dir = "/home/z3r0/Desktop/All/[Exercises]/[Deep_Learning]/dogs_vs_cats_classification/datasets/test"

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
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation = 'relu'))
    model.add(layers.Dense(1, activation = 'sigmoid'))

    model.compile(
        loss = "binary_crossentropy",
        optimizer = optimizers.RMSprop(lr = 1e-4),
        metrics = ['acc']
    )

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
        batch_size = 32,
        class_mode = 'binary'
    )
    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size = (150, 150),
        batch_size = 32,
        class_mode = 'binary'
    )

    history = model.fit_generator(
        train_generator,
        steps_per_epoch = 100,
        epochs = 100,
        validation_data = validation_generator,
        validation_steps = 50
    )

    model.save('Convnet_with_DataAugmentation_Generators.h5')

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

if __name__ == "__main__":
    main()
