# Loading Libraries

import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import RMSprop


# Preprocessing the images for neuralnet

def Prep(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # making image grayscale
    img = cv2.equalizeHist(img)  # Histogram equalization to enhance contrast
    img = img/255  # normalizing
    return img


def preprocessing():

    data = os.listdir(os.path.abspath('./static/dataset/Digits'))
    data_X = []
    data_y = []
    data_classes = len(data)
    for i in range(0, data_classes):
        data_list = os.listdir(os.path.abspath(
            './static/dataset/Digits') + "/"+str(i))
        for j in data_list:
            pic = cv2.imread(
                os.path.abspath('./static/dataset/Digits') + "/"+str(i)+"/"+j)
            pic = cv2.resize(pic, (32, 32))
            data_X.append(pic)
            data_y.append(i)

    # Spliting the train validation and test sets

    train_X, test_X, train_y, test_y = train_test_split(
        data_X, data_y, test_size=0.05)
    train_X, valid_X, train_y, valid_y = train_test_split(
        train_X, train_y, test_size=0.2)

    # Labels and images
    data_X = np.array(data_X)
    data_y = np.array(data_y)

    train_X = np.array(list(map(Prep, train_X)))
    test_X = np.array(list(map(Prep, test_X)))
    valid_X = np.array(list(map(Prep, valid_X)))

    # Reshaping the images
    train_X = train_X.reshape(
        train_X.shape[0], train_X.shape[1], train_X.shape[2], 1)
    test_X = test_X.reshape(
        test_X.shape[0], test_X.shape[1], test_X.shape[2], 1)
    valid_X = valid_X.reshape(
        valid_X.shape[0], valid_X.shape[1], valid_X.shape[2], 1)

    # Augmentation
    datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1,
                                 zoom_range=0.2, shear_range=0.1, rotation_range=10)
    datagen.fit(train_X)

    # One hot encoding of the labels

    train_y = to_categorical(train_y, data_classes)
    test_y = to_categorical(test_y, data_classes)
    valid_y = to_categorical(valid_y, data_classes)

    return datagen, train_X, train_y, valid_X, valid_y


def ModelTraining():

    datagen, train_X, train_y, valid_X, valid_y = preprocessing()

    # Creating a Neural Network
    model = Sequential()

    model.add((Conv2D(60, (5, 5), input_shape=(32, 32, 1),
              padding='Same', activation='relu')))
    model.add((Conv2D(60, (5, 5), padding="same", activation='relu')))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add((Conv2D(30, (3, 3), padding="same", activation='relu')))
    model.add((Conv2D(30, (3, 3), padding="same", activation='relu')))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    # Compiling the model

    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy', metrics=['accuracy'])

    # Fit the model

    model.fit(datagen.flow(train_X, train_y, batch_size=32),
              epochs=12, validation_data=(valid_X, valid_y),
              verbose=2, steps_per_epoch=200)

    return model


if __name__ == '__main__':
    model = ModelTraining()
    model.save(os.path.abspath('./static') + '/model')
