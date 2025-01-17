import os
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

def main():
    # Applicare Data Augmentation usando "ImageDataGenerator"
    datagen = ImageDataGenerator(
        rotation_range = 40,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        fill_mode = 'nearest'
    )

    train_cats_dir = "/home/z3r0/Desktop/All/[Exercises]/[Deep_Learning]/dogs_vs_cats_classification/datasets/train/cats"

    fnames = [ os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir) ]
    img_path = fnames[3]
    img = image.load_img( img_path, target_size = (150, 150))
    # Converte in array Numpy di forma (150, 150, 3)
    x = image.img_to_array(img)
    x = x.reshape((1, ) + x.shape)

    i = 0
    for batch in datagen.flow(x, batch_size = 1):
        plt.figure(i)
        imgplot = plt.imshow(image.array_to_img(batch[0]))
        i += 1
        if i % 4 == 0:
            break
    plt.show()


if __name__ == "__main__":
    main()
