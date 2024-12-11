# Copia immagini del Dataset nelle directories "training, validation e test" #

import os, shutil

def main():
    original_dataset_dir = "/home/z3r0/Desktop/All/[Exercises]/[Deep_Learning]/dogs_vs_cats_classification/original/train"

    base_dir = "/home/z3r0/Desktop/All/[Exercises]/[Deep_Learning]/dogs_vs_cats_classification/datasets"
    os.mkdir(base_dir)

    # Crea directories 'train, validation, test' per gatti e cani
    train_dir = os.path.join(base_dir, 'train')
    os.mkdir(train_dir)
    validation_dir = os.path.join(base_dir, 'validation')
    os.mkdir(validation_dir)
    test_dir = os.path.join(base_dir, 'test')
    os.mkdir(test_dir)
    train_cats_dir = os.path.join(train_dir, 'cats')
    os.mkdir(train_cats_dir)
    train_dogs_dir = os.path.join(train_dir, 'dogs')
    os.mkdir(train_dogs_dir)
    validation_cats_dir = os.path.join(validation_dir, 'cats')
    os.mkdir(validation_cats_dir)
    validation_dogs_dir = os.path.join(validation_dir, 'dogs')
    os.mkdir(validation_dogs_dir)
    test_cats_dir = os.path.join(test_dir, 'cats')
    os.mkdir(test_cats_dir)
    test_dogs_dir = os.path.join(test_dir, 'dogs')
    os.mkdir(test_dogs_dir)

    # Copia le immagini dei gatti dividendole nelle varie directories
    fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_cats_dir, fname)
        shutil.copyfile(src, dst)
    fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_cats_dir, fname)
        shutil.copyfile(src, dst)
    fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_cats_dir, fname)
        shutil.copyfile(src, dst)

    # Copia le immagini dei cani dividendole nelle varie directories
    fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_dogs_dir, fname)
        shutil.copyfile(src, dst)
    fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_dogs_dir, fname)
        shutil.copyfile(src, dst)
    fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_dogs_dir, fname)
        shutil.copyfile(src, dst)

    # Controllo del numero di immagini nelle varie cartelle
    print("[+] Total training cat images: '%s'" % len(os.listdir(train_cats_dir)))
    print("[+] Total validation cat images: '%s'" % len(os.listdir(validation_cats_dir)))
    print("[+] Total test cat images: '%s'" % len(os.listdir(test_cats_dir)))
    print("[+] Total training dog images: '%s'" % len(os.listdir(train_dogs_dir)))
    print("[+] Total validation dog images: '%s'" % len(os.listdir(validation_dogs_dir)))
    print("[+] Total test dog images: '%s'" % len(os.listdir(test_dogs_dir)))

if __name__ == "__main__":
    main()
