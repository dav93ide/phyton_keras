from keras.models import Sequential, Model
from keras import layers

## 262- Implementazione di una Convnet separabile in profondita` per operazione di classificazione di immagini
def main():
    height = 64
    width = 64
    channels = 3
    num_classes = 10

    model = Sequential()
    model.add(layers.SeparableConv2D(32, 3, activation="relu", input_shape=(height, width, channels, )))
    model.add(layers.SeparableConv2D(64, 3, activation="relu"))
    model.add(layers.MaxPooling2D(2))

    model.add(layers.SeparableConv2D(64, 3, activation="relu"))
    model.add(layers.SeparableConv2D(128, 3, activation="relu"))
    model.add(layers.MaxPooling2D(2))

    model.add(layers.SeparableConv2D(64, 3, activation="relu"))
    model.add(layers.SeparableConv2D(128, 3, activation="relu"))
    model.add(layers.GlobalAveragePooling2D())

    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dense(num_classes, activation="softmax"))

    model.compile(optimizer="rmsprop", loss="categorical_crossentropy")

if __name__ == "__main__":
    main()
