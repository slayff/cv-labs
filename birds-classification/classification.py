import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv2D
from keras.callbacks import ModelCheckpoint
from keras.layers import MaxPooling2D, Dense, Reshape, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam, Adadelta
from skimage.io import imread
from skimage.transform import resize, rotate
from sklearn.model_selection import train_test_split

from keras.applications import Xception
from keras.applications.xception import preprocess_input
from keras.utils import to_categorical

import os
import sys
from os.path import join, dirname, abspath
from os import makedirs, listdir

IMG_SIZE = 299
NB_CLASSES = 50

class Xception_birds():
    def __init__(self):
        self.base_model = Xception(input_shape=(IMG_SIZE, IMG_SIZE, 3), weights='imagenet', include_top=False)

        x = self.base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.2)(x)
        predictions = Dense(NB_CLASSES, activation='softmax')(x)

        self.model = Model(inputs=self.base_model.input, outputs=predictions)

    def get_model(self):
        return self.model

    def fit_model(self, x_train, y_train, x_test=None, y_test=None, batch_size=128, nb_epochs=50, fast_train=False):
        for layer in self.base_model.layers:
            layer.trainable = False

        self.model.compile(optimizer='rmsprop',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

        model_path = join(dirname(abspath(__file__)), 'models')
        train_datagen = ImageDataGenerator(rotation_range=45,
                                   zoom_range=.1,
                                   horizontal_flip=True,
                                   vertical_flip=True)
        train_generator = train_datagen.flow(x_train, y_train, batch_size=128, seed=181)
        STEPS_PER_EPOCH = x_train.shape[0] // batch_size

        if fast_train:
            self.model.fit(x_train,
                       y_train,
                       batch_size=8,
                       epochs=1,
                       verbose=1)
        else:
            makedirs(model_path, exist_ok=True)
            checkpointer = ModelCheckpoint(filepath=join(model_path, 'Xception_init.hdf5'),
                                           verbose=1, save_best_only=True, monitor='val_acc')

            self.model.fit_generator(train_generator,
                         STEPS_PER_EPOCH,
                         epochs=20,
                         verbose=1,
                         validation_data=(x_test, y_test),
                         callbacks=[checkpointer])

        if not fast_train:
            self.model.load_weights(join(model_path, 'Xception_init.hdf5'))

        for layer in self.model.layers[:105]:
            layer.trainable = False
        for layer in self.model.layers[105:]:
            layer.trainable = True

        self.model.compile(optimizer=SGD(lr=0.001, momentum=0.9),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

        if fast_train:
            self.model.fit(x_train,
                       y_train,
                       batch_size=8,
                       epochs=1,
                       verbose=1)
        else:
            checkpointer = ModelCheckpoint(filepath=join(model_path, 'Xception_final.hdf5'),
                                           verbose=1, save_best_only=True, monitor='val_acc')

            self.model.fit_generator(train_generator,
                             STEPS_PER_EPOCH + 5,
                             epochs=50,
                             verbose=1,
                             validation_data=(x_test, y_test),
                             callbacks=[checkpointer])

def load_train_data(train_gt, train_img_dir, small_size=False):
    X = []
    Y = []
    threshold = -1
    if small_size:
        threshold = 500
    cur_size = 0
    for filename, class_id in train_gt.items():
        X.append(resize(imread(join(train_img_dir, filename)),
                        (IMG_SIZE, IMG_SIZE, 3),
                        mode='constant',
                        anti_aliasing=False)
                )
        Y.append(class_id)
        cur_size += 1
        if cur_size == threshold:
            break
    X = np.array(X)
    X = preprocess_input(255 * X)
    Y = to_categorical(np.array(Y), num_classes=NB_CLASSES)
    return X, Y

def load_test_data(test_img_dir):
    files = sorted(listdir(test_img_dir))

    data = []
    for filename in files:
        data.append(resize(imread(join(test_img_dir, filename)),
                        (IMG_SIZE, IMG_SIZE, 3),
                        mode='constant',
                        anti_aliasing=False)
                   )
    data = np.array(data)
    return data, files

def files_generator(test_img_dir, batch_size=64):
    files = sorted(listdir(test_img_dir))
    f_count = len(files)
    begin = 0
    while begin < f_count:
        end = min(begin + batch_size, f_count)
        yield files[begin:end]
        begin = end

def train_classifier(train_gt, train_img_dir, fast_train=True):
    x_train, y_train = None, None
    birds_model = Xception_birds()
    if fast_train:
        x_train, y_train = load_train_data(train_gt, train_img_dir, small_size=True)
        birds_model.fit_model(x_train, y_train, fast_train=True)
    else:
        from sklearn.model_selection import train_test_split
        x_full, y_full = load_train_data(train_gt, train_img_dir)
        x_train, x_test, y_train, y_test = train_test_split(images, facepoints, test_size=0.2, random_state=177)
        birds_model.fit_model(x_train, y_train, batch_size=128, nb_epochs=50)

def classify(model, test_img_dir):
    ans = {}
    for files in files_generator(test_img_dir):
        batch = []
        for file in files:
            batch.append(resize(imread(join(test_img_dir, file)),
                            (IMG_SIZE, IMG_SIZE, 3),
                            mode='constant',
                            anti_aliasing=False))
        batch = np.array(batch)
        batch = preprocess_input(255 * batch)

        probs = model.predict_on_batch(batch)
        predictions = probs.argmax(axis=-1)
        for i in range(len(files)):
            ans[files[i]] = predictions[i]
    return ans

