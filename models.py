import os
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input, UpSampling2D, Dropout
from keras.optimizers import Adam
from keras.models import Sequential
import generators
from keras.callbacks import ModelCheckpoint, History, EarlyStopping, LearningRateScheduler
from keras.layers import ReLU
from keras import backend as K
import tensorflow as tf

def quadriview_model(name=None, weights=None):
    """
    Generates a Keras.Sequential taking as input a quadriview image
    """
    Q_cor_model = Sequential(name = name)
    Q_cor_model.add(Conv2D(16, (3,3), activation='relu', input_shape=(400, 400, 1)))
    Q_cor_model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    Q_cor_model.add(Conv2D(16, (3,3), activation='relu'))
    Q_cor_model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    Q_cor_model.add(Conv2D(32, (3,3), activation='relu'))
    Q_cor_model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    Q_cor_model.add(Conv2D(32, (3,3), activation='relu'))
    Q_cor_model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    Q_cor_model.add(Conv2D(64, (3,3), activation='relu'))
    Q_cor_model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    Q_cor_model.add(Conv2D(64, (3,3), activation='relu'))
    Q_cor_model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    Q_cor_model.add(Flatten())
    Q_cor_model.add(Dropout(rate=0.2))
    Q_cor_model.add(Dense(8, activation = "relu"))
    Q_cor_model.add(Dropout(rate=0.2))
    Q_cor_model.add(Dense(2, activation = "softmax"))
    if weights is not None:
        Q_cor_model.load_weights(weights)
    return Q_cor_model

def one_slice_model(name=None, weights = None):
    """
    Generates a Keras.Sequential taking as input a slice image
    """
    cor_model = Sequential(name = name)
    #cor_model.add(Dropout(rate=0.2))
    cor_model.add(Conv2D(16, (3,3), activation='relu', input_shape=(182, 182, 1)))
    cor_model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    #cor_model.add(Dropout(rate=0.3))
    cor_model.add(Conv2D(16, (3,3), activation='relu'))
    cor_model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    #cor_model.add(Dropout(rate=0.4))
    cor_model.add(Conv2D(16, (3,3), activation='relu'))
    cor_model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    #cor_model.add(Dropout(rate=0.5))
    cor_model.add(Conv2D(16, (3,3), activation='relu'))
    cor_model.add(MaxPool2D(pool_size=(2, 2)))
    cor_model.add(Conv2D(16, (3,3), activation='relu'))
    cor_model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    cor_model.add(Flatten())
    cor_model.add(Dropout(rate=0.4))
    cor_model.add(Dense(8, activation = "relu"))
    cor_model.add(Dropout(rate=0.4))
    cor_model.add(Dense(2, activation = "softmax"))
    if weights is not None:
        cor_model.load_weights(weights)
    return cor_model
