import numpy as np
import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.callbacks import ModelCheckpoint
from keras.layers import MaxPooling2D, Dense, Reshape, Flatten, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam
from math import pi, cos, sin, degrees, radians
from random import random, uniform, choice
from skimage.io import imread
from skimage.transform import resize, rotate
from skimage.color import rgb2gray

from os.path import join, dirname, abspath
from os import makedirs, listdir

PROB = 0.4
ROTATION_ANGLE = 5
DELTA = 0.8
IMG_SIZE = 96

def resize_img(img, facepoints, target_size):
    img_new = resize(img, (target_size, target_size), mode='constant', anti_aliasing=False)
    ratio = float(target_size) / img.shape[0]
    facepoints_new = facepoints * ratio
    return img_new, facepoints_new

def rotate_img(img, facepoints, angle):
    '''
    rotate image clockwise by defined angle in degrees
    '''
    def get_rotation_matrix(alpha):
        a = radians(alpha)
        rm = np.array([[cos(a), -sin(a)], [sin(a), cos(a)]])
        return rm

    img_new = rotate(img, angle=-angle, mode='edge')
    center = img.shape[0] / 2 - 0.5
    facepoints_new = facepoints.copy()
    facepoints_new -= center
    rmat = get_rotation_matrix(angle)
    facepoints_new = rmat.dot(facepoints_new.T).T
    facepoints_new += center
    return img_new, facepoints_new

def flip_img(img, facepoints):
    indices = ((0, 3), (1, 2), (4, 9), (5, 8), (6, 7), (11, 13))
    img_new = img[:,::-1]
    center = img.shape[0] / 2 - 0.5
    facepoints_new = facepoints.copy()
    facepoints_new[:, 0] -= center
    facepoints_new[:, 0] *= -1
    facepoints_new[:, 0] += center
    fp_copy = facepoints_new.copy()
    for i, j in indices:
        facepoints_new[i, :], facepoints_new[j, :] = fp_copy[j, :], fp_copy[i, :]
    return img_new, facepoints_new

def cr_img(img, facepoints, delta=0.8):
    img_new = img.copy()
    facepoints_new = facepoints.copy()
    mean = np.mean(img_new)
    img_new = delta * img_new + (1 - delta) * mean
    return img_new, facepoints_new

def resize_and_get_ratio(img, target_size):
    img_new = resize(img, (target_size, target_size), mode='constant', anti_aliasing=False)
    ratio = img.shape[0] /  float(target_size)
    return img_new, ratio

class LeNet():
    def __init__(self):
        self.model = Sequential()

        self.model.add(Conv2D(32, (3, 3), input_shape=(96, 96, 1), kernel_initializer='he_normal', padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Dropout(0.1))

        self.model.add(Conv2D(64, (2, 2)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Dropout(0.1))

        self.model.add(Conv2D(128, (2, 2)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Dropout(0.1))

        self.model.add(Conv2D(256, (2, 2)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Dropout(0.1))

        self.model.add(Flatten())

        self.model.add(Dense(500, activation="relu"))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(28))
        self.model.add(Reshape((14, 2)))

        self.model.compile(loss='mean_squared_error', optimizer=Adam(), metrics=['accuracy'])

    def print_summary(self):
        self.model.summary()

    def get_model(self):
        return self.model

    def fit_model(self,
            x_train,
            y_train,
            x_test=None,
            y_test=None,
            batch_size=128,
            nb_epochs=100,
            fast_train=False):
        if fast_train:
            self.model.fit(x_train,
                          y_train,
                          batch_size=8,
                          epochs=1,
                          verbose=1)
        else:
            model_path = join(dirname(abspath(__file__)), 'models')
            makedirs(model_path, exist_ok=True)
            checkpointer = ModelCheckpoint(filepath=join(model_path, 'vgg_model.hdf5'),
                    verbose=1,
                    save_best_only=True)
            self.model.fit(x_train,
                      y_train,
                      batch_size=batch_size,
                      epochs=nb_epochs,
                      validation_data=(x_test, y_test),
                      verbose=1,
                      callbacks=[checkpointer])

def load_test_data(directory):
    files = sorted(listdir(directory))
    data = []
    ratios = []

    for filename in files:
        img = rgb2gray(imread(join(directory, filename)))
        resized_img, ratio = resize_and_get_ratio(img, IMG_SIZE)
        data.append(resized_img)
        ratios.append(ratio)
    data = np.array(data).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    return files, data, ratios

def load_train_data(train_gt, train_img_dir, augmentation=False, save=False, small_size=False):
    X = []
    Y = []
    threshold = -1
    if small_size:
        threshold = 64
    cur_size = 0
    for filename, points in train_gt.items():
        img = rgb2gray(imread(join(train_img_dir, filename)))
        fp1 = points[::2]
        fp2 = points[1::2]
        fp = np.concatenate((fp1.reshape(-1, 1), fp2.reshape(-1, 1)), axis=1)

        resized_img, resized_fp = resize_img(img, fp, IMG_SIZE)
        X_aug, Y_aug = [resized_img], [resized_fp]

        if augmentation:
            cur_prob = random()
            if cur_prob >= PROB:
                flipped_img, flipped_fp = flip_img(resized_img, resized_fp)
                X_aug.append(flipped_img)
                Y_aug.append(flipped_fp)

            X_aug_t, Y_aug_t = [], []
            for img, fp in zip(X_aug, Y_aug):
                cur_prob = random()
                if cur_prob >= PROB:
                    sign = np.sign(uniform(-1, 1))
                    rotated_img, rotated_fp = rotate_img(img, fp, ROTATION_ANGLE * sign)
                    X_aug_t.append(rotated_img)
                    Y_aug_t.append(rotated_fp)
            X_aug += X_aug_t
            Y_aug += Y_aug_t

            X_aug_t, Y_aug_t = [], []
            for img, fp in zip(X_aug, Y_aug):
                cur_prob = random()
                if cur_prob >= PROB:
                    crd_img, crd_fp = cr_img(img, fp)
                    X_aug_t.append(crd_img)
                    Y_aug_t.append(crd_fp)
            X_aug += X_aug_t
            Y_aug += Y_aug_t

        X += X_aug
        Y += Y_aug
        cur_size += 1
        if cur_size == threshold:
            break
    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    Y = np.array(Y)

    if save:
        code_dir = dirname(abspath(__file__))
        np.savez(join(code_dir, 'train_aug'), images=X, facepoints=Y)
    return X, Y

def detect(model, test_img_dir):
    files, data, ratios = load_test_data(test_img_dir)
    predictions = model.predict(data, verbose=1)
    answer = {}
    for num in range(predictions.shape[0]):
        predictions[num] *= ratios[num]
        predicted_facepoints = predictions[num].reshape(-1).tolist()
        answer[files[num]] = predicted_facepoints

    return answer

def train_detector(train_gt, train_img_dir, fast_train=False):
    model = LeNet()
    if fast_train:
        x_train, y_train = load_train_data(train_gt, train_img_dir, small_size=True)
        model.fit_model(x_train, y_train, fast_train=True)
    else:
        from sklearn.model_selection import train_test_split
        x_full, y_full = load_train_data(train_gt, train_img_dir, augmentation=True, save=True)
        x_train, x_test, y_train, y_test = train_test_split(x_full,
                                                            y_full,
                                                            test_size=0.2,
                                                            random_state=177)
        model.fit_model(x_train, y_train, batch_size=128, nb_epochs=100)

