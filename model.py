import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import cv2
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Convolution2D, MaxPooling2D, Flatten, Lambda, Cropping2D
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2, activity_l2
from keras.optimizers import Adam
from keras.models import model_from_json
import json

matplotlib.style.use('ggplot')

data_dir = './data/'
data_csv = 'driving_log.csv'
model_weights = 'model.h5'

training_dat = pd.read_csv(data_dir+data_csv, names=None)

training_dat[['left', 'center', 'right']]
X_train = training_dat[['left', 'center', 'right']]
Y_train = training_dat['steering']

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)

# get rid of the pandas index after shuffling
X_left = X_train['left'].as_matrix()
X_right = X_train['right'].as_matrix()
X_train = X_train['center'].as_matrix()
X_val = X_val['center'].as_matrix()
Y_val = Y_val.as_matrix()
Y_train = Y_train.as_matrix()

Y_train = Y_train.astype(np.float32)
Y_val = Y_val.astype(np.float32)

def normalize_path(img_file):
    """ Normalizes the path to a give image file """

    img_file = img_file.split('/')[-1]
    img_file = 'data/IMG/'+img_file.split('\\')[-1]
    return img_file

def read_next_image(m, lcr, X_center, X_left, X_right, Y_train):
    """ Prepares the next imae and the corresponding steering angle """
    offset = 1.0
    dist = 20.0
    steering = Y_train[m]

    if lcr == 0:
        image = plt.imread(normalize_path(X_left[m]))
        dsteering = offset / dist * 360 / (2 * np.pi) / 25.0
        steering += dsteering
    elif lcr == 1:
        image = plt.imread(normalize_path(X_center[m]))
    elif lcr == 2:
        image = plt.imread(normalize_path(X_right[m]))
        dsteering = -offset / dist * 360 / (2 * np.pi)  / 25.0
        steering += dsteering
    else:
        print('Invalid lcr value :', lcr)

    return image, steering

def random_crop(image, steering = 0.0, tx_lower = -20, tx_upper = 20, ty_lower = -2, ty_upper = 2, rand = True):
    """ Randomlly crops an image based on some bounds """

    shape = image.shape
    (col_start, col_end) = (abs(tx_lower), shape[1] - tx_upper)
    horizon = 60
    bonnet = 136
    if rand:
        tx = np.random.randint(tx_lower, tx_upper + 1)
        ty = np.random.randint(ty_lower, ty_upper + 1)
    else:
        (tx, ty) = (0, 0)

    crop = image[horizon + ty: bonnet + ty, col_start + tx: col_end + tx, :]
    image = cv2.resize(crop, (320, 160), cv2.INTER_AREA)
    # the steering variable needs to be updated to counteract the shift 
    if tx_lower != tx_upper:
        dsteering = -tx / (tx_upper - tx_lower) / 3.0
    else:
        dsteering = 0
    steering += dsteering

    return image, steering

def random_shear(image, steering, shear_range):
    """ Randomly warps an image """

    rows, cols, _ = image.shape
    dx = np.random.randint(-shear_range, shear_range + 1)

    random_point = [(cols / 2) + dx, rows / 2]
    pts1 = np.float32([[0, rows], [cols, rows], [cols / 2, rows / 2]])
    pts2 = np.float32([[0, rows], [cols, rows], random_point])
    dsteering = dx / (rows / 2) * 360 / (2 * np.pi * 25.0) / 6.0
    matrix = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, matrix, (cols, rows), borderMode=1)
    steering += dsteering

    return image, steering

def random_brightness(image):
    """ Add random brightness to an image """
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_bright = 0.9 + 0.5 * ((2 * np.random.uniform()) - 1.0)
    image1[:, :, 2] = image1[:, :, 2] * random_bright
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    return image1

def random_flip(image, steering):
    """
    Randomly decides wheter to flip an image
    return image and corrected steering if the image has been flipped
    """

    coin = np.random.randint(0, 2)

    if coin == 0:
        image, steering = cv2.flip(image, 1), -steering

    return image, steering

def generate_training_example(X_train, X_left, X_right, Y_train):
    """ Generates an example """

    m = np.random.randint(0, len(Y_train))
    lcr = np.random.randint(0, 3)

    image, steering = read_next_image(m, lcr, X_train, X_left, X_right, Y_train)
    image, steering = random_shear(image, steering, shear_range=100)
    image, steering = random_crop(image, steering,
                                  tx_lower=-20, tx_upper=20, ty_lower=-10, ty_upper=10)
    image, steering = random_flip(image, steering)
    image = random_brightness(image)

    return image, steering

def get_validation_set(X_val, Y_val):
    """ Prepares validation set """
    X = np.zeros((len(X_val), 160, 320, 3))
    Y = np.zeros(len(X_val))

    for i in range(len(X_val)):
        x, y = read_next_image(i, 1, X_val, X_val, X_val, Y_val)

        X[i], Y[i] = random_crop(x, y, tx_lower=0, tx_upper=0, ty_lower=0, ty_upper=0)

    return X, Y

def generate_train_batch(X_train, X_left, X_right, Y_train, batch_size=32):
    """ Generator for keras """

    batch_images = np.zeros((batch_size, 160, 320, 3))
    batch_steering = np.zeros(batch_size)
    while 1:
        for i_batch in range(batch_size):
            x, y = generate_training_example(X_train, X_left, X_right, Y_train)
            batch_images[i_batch] = x
            batch_steering[i_batch] = y
        yield batch_images, batch_steering

def resize_function(input):
    """ Resize function for the lambda layer """

    from keras.backend import tf as ktf
    return ktf.image.resize_images(input, (64, 64))

batch_size = 256
train_generator = generate_train_batch(X_train, X_left, X_right, Y_train, batch_size)
X_val, Y_val = get_validation_set(X_val, Y_val)

model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(160, 320, 3)))
model.add(Lambda(resize_function, input_shape=(160, 320, 3), output_shape=(64, 64, 3)))

# Convolution layers with Relu activation
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))

# Flatten layer
model.add(Flatten())

# Dense layers with Dropout
model.add(Dense(16896))
model.add(ELU())
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(ELU())
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(ELU())
model.add(Dropout(0.5))
model.add(Dense(1))

model.summary()

adam = Adam(lr=1e-4)
model.compile(optimizer=adam, loss='mse')
nb_epoch = 20
np_samples = 32768
history = model.fit_generator(train_generator,
                              samples_per_epoch=np_samples,
                              nb_epoch=nb_epoch,
                              validation_data=(X_val, Y_val),
                              verbose=1)

print('Save the model')

model.save(model_weights)

print('Done')
