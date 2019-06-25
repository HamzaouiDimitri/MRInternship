import os
from keras.layers import Conv2D, Conv3D, MaxPool2D, MaxPool3D, Flatten, Dense, Input, UpSampling2D, Dropout, BatchNormalization, ReLU, Softmax, Add, Concatenate
from keras.initializers import Constant
from keras.optimizers import Adam
from keras.models import Sequential, Model
import generators
from keras.callbacks import ModelCheckpoint, History, EarlyStopping, LearningRateScheduler
from keras.layers import ReLU
from keras import backend as K
import tensorflow as tf
import numpy as np

def quadriview_model(name=None, weights=None):
    """
    Generates a Keras.Sequential taking as input a quadriview image
    """
    model = Sequential(name = name)
    model.add(Conv2D(16, (3,3), activation='relu', input_shape=(400, 400, 1)))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(16, (3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(rate=0.2))
    model.add(Dense(8, activation = "relu"))
    model.add(Dropout(rate=0.2))
    model.add(Dense(2, activation = "softmax"))
    if weights is not None:
        model.load_weights(weights)
    return model

def BN_quadriview_model(name=None, weights=None, BN_momentum = 0.99):
    """
    Generates a Keras.Sequential taking as input a quadriview image
    """
    model = Sequential(name = name)
    model.add(Conv2D(16, (3,3), activation='relu', input_shape=(400, 400, 1)))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(16, (3,3), activation='relu'))
    model.add(BatchNormalization(momentum = BN_momentum))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(BatchNormalization(momentum = BN_momentum))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(BatchNormalization(momentum = BN_momentum))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(BatchNormalization(momentum = BN_momentum))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(BatchNormalization(momentum = BN_momentum))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(rate=0.2))
    model.add(Dense(8, activation = "relu"))
    model.add(Dropout(rate=0.2))
    model.add(Dense(2, activation = "softmax"))
    if weights is not None:
        model.load_weights(weights)
    return model


def one_slice_model(name=None, weights = None):
    """
    Generates a Keras.Sequential taking as input a slice image
    """
    model = Sequential(name = name)
    #model.add(Dropout(rate=0.2))
    model.add(Conv2D(16, (3,3), activation='relu', input_shape=(182, 182, 1)))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    #model.add(Dropout(rate=0.3))
    model.add(Conv2D(16, (3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    #model.add(Dropout(rate=0.4))
    model.add(Conv2D(16, (3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    #model.add(Dropout(rate=0.5))
    model.add(Conv2D(16, (3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(rate=0.4))
    model.add(Dense(8, activation = "relu"))
    model.add(Dropout(rate=0.4))
    model.add(Dense(2, activation = "softmax"))
    if weights is not None:
        model.load_weights(weights)
    return model

def model_3D(name = None, weights = None):
    model = Sequential(name = name)
    model.add(Conv3D(16, (3,3,3), activation='relu', input_shape=(60, 60, 60, 1), padding = "same"))
    model.add(MaxPool3D((30, 30, 30)))
    model.add(Flatten())
    model.add(Dense(2, activation = "sigmoid"))
    return model

def kustner_model(name = None, weights = None):
    model = Sequential(name = name)
    model.add(Conv3D(32, (14, 14, 1), activation='relu', input_shape=(60, 60, 1, 1)))
    model.add(Conv3D(64, (7, 7, 32), activation='relu'))
    model.add(Conv3D(128, (3, 3, 64), activation='relu'))
    model.add(Flatten())
    model.add(Dense(2, activation = "softmax"))
    model.compile(loss = "binary_crossentropy", optimizer = Adam(beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8))
    return model

from keras import backend as K

def my_init(channels_in, channels_out):
    def init_true(shape, dtype=None):
        tensor = tf.eye(channels_in, num_columns=channels_out)
        tensor =  K.expand_dims(tensor, axis = 0)
        tensor =  K.expand_dims(tensor, axis = 0)
        tensor =  K.expand_dims(tensor, axis = 0)
        return tensor
    return init_true

def HighRes3D_model(name = None, weights = None):
    inputs = Input(shape = (60, 60, 60, 1))
    lay_1 = Conv3D(16, (3, 3, 3), padding = "same")(inputs)
    lay_1 = BatchNormalization()(lay_1)
    lay_1 = ReLU()(lay_1)

    lay_2 = BatchNormalization()(lay_1)
    lay_2 = ReLU()(lay_2)
    lay_2 = Conv3D(16, (3, 3, 3), padding = "same")(lay_2)

    lay_3 = BatchNormalization()(lay_2)
    lay_3 = ReLU()(lay_3)
    lay_3 = Conv3D(16, (3, 3, 3), padding = "same")(lay_3)
    lay_3 = Add()([lay_3, lay_1])

    lay_4 = BatchNormalization()(lay_3)
    lay_4 = ReLU()(lay_4)
    lay_4 = Conv3D(16, (3, 3, 3), padding = "same")(lay_4)

    lay_5 = BatchNormalization()(lay_4)    
    lay_5 = ReLU()(lay_5)
    lay_5 = Conv3D(16, (3, 3, 3), padding = "same")(lay_5)
    lay_5 = Add()([lay_5, lay_3])

    lay_6 = BatchNormalization()(lay_5)
    lay_6 = ReLU()(lay_6)
    lay_6 = Conv3D(16, (3, 3, 3), padding = "same")(lay_6)

    lay_7 = BatchNormalization()(lay_6)    
    lay_7 = ReLU()(lay_7)
    lay_7 = Conv3D(16, (3, 3, 3), padding = "same")(lay_7)
    lay_7 = Add()([lay_7, lay_5])

    lay_1_1 = BatchNormalization()(lay_7)
    lay_1_1 = ReLU()(lay_1_1)
    lay_1_1 = Conv3D(32, (3, 3, 3), dilation_rate = (2, 2, 2), padding = "same")(lay_1_1)
    
    lay_1_2 = BatchNormalization()(lay_1_1)
    lay_1_2 = ReLU()(lay_1_2)
    lay_1_2 = Conv3D(32, (3, 3, 3), dilation_rate = (2, 2, 2), padding = "same")(lay_1_2)
    lay_7_concat = Conv3D(32, (1, 1, 1), padding = "same",  bias_initializer='zeros', trainable = False,
    kernel_initializer = my_init(16, 32), name = "first_concat")(lay_7)
    lay_1_2 = Add()([lay_1_2, lay_7_concat])

    lay_1_3 = BatchNormalization()(lay_1_2)
    lay_1_3 = ReLU()(lay_1_3)
    lay_1_3 = Conv3D(32, (3, 3, 3), dilation_rate = (2, 2, 2), padding = "same")(lay_1_3)

    lay_1_4 = BatchNormalization()(lay_1_3)
    lay_1_4 = ReLU()(lay_1_4)
    lay_1_4 = Conv3D(32, (3, 3, 3), dilation_rate = (2, 2, 2), padding = "same")(lay_1_4)
    lay_1_4 = Add()([lay_1_4, lay_1_2])

    lay_1_5 = BatchNormalization()(lay_1_4)    
    lay_1_5 = ReLU()(lay_1_5)
    lay_1_5 = Conv3D(32, (3, 3, 3), dilation_rate = (2, 2, 2), padding = "same")(lay_1_5)

    lay_1_6 = BatchNormalization()(lay_1_5)
    lay_1_6 = ReLU()(lay_1_6)
    lay_1_6 = Conv3D(32, (3, 3, 3), dilation_rate = (2, 2, 2), padding = "same")(lay_1_6)
    lay_1_6 = Add()([lay_1_6, lay_1_4])

    lay_2_1 = BatchNormalization()(lay_1_6)
    lay_2_1 = ReLU()(lay_2_1)
    lay_2_1 = Conv3D(64, (3, 3, 3), dilation_rate = (4, 4, 4), padding = "same")(lay_2_1)
    
    lay_2_2 = BatchNormalization()(lay_2_1)
    lay_2_2 = ReLU()(lay_2_2)
    lay_2_2 = Conv3D(64, (3, 3, 3), dilation_rate = (4, 4, 4), padding = "same")(lay_2_2)
    lay_1_6_concat = Conv3D(64, (1,1,1), padding = "same", bias_initializer='zeros', trainable = False,
    kernel_initializer = my_init(32, 64))(lay_1_6)
    lay_2_2 = Add()([lay_2_2, lay_1_6_concat])

    lay_2_3 = BatchNormalization()(lay_2_2)
    lay_2_3 = ReLU()(lay_2_3)
    lay_2_3 = Conv3D(64, (3, 3, 3), dilation_rate = (4, 4, 4), padding = "same")(lay_2_3)

    lay_2_4 = BatchNormalization()(lay_2_3)
    lay_2_4 = ReLU()(lay_2_4)
    lay_2_4 = Conv3D(64, (3, 3, 3), dilation_rate = (4, 4, 4), padding = "same")(lay_2_4)
    lay_2_4 = Add()([lay_2_4, lay_2_2])

    lay_2_5 = BatchNormalization()(lay_2_4)    
    lay_2_5 = ReLU()(lay_2_5)
    lay_2_5 = Conv3D(64, (3, 3, 3), dilation_rate = (4, 4, 4), padding = "same")(lay_2_5)

    lay_2_6 = BatchNormalization()(lay_2_5)
    lay_2_6 = ReLU()(lay_2_6)
    lay_2_6 = Conv3D(64, (3, 3, 3), dilation_rate = (4, 4, 4), padding = "same")(lay_2_6)
    lay_2_6 = Add()([lay_2_6, lay_2_4])

    final_lay = Conv3D(160, (1, 1, 1), padding = "same")(lay_2_6)
    output = Softmax()(final_lay)


    model = Model(input = inputs, outputs = final_lay)
    return(model)


 
  
  
