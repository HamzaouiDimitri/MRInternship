#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 18:03:35 2019

@author: dimitri.hamzaoui
"""

from generators import Quadriview_DataGenerator
from models import quadriview_model

import tensorflow as tf
import keras.backend as K
from model_utils import model_apply, test_noise_characs, test_slice, model_eval
from keras.optimizers import Adam


    
if __name__ == "__main__":

    generator_train = Quadriview_DataGenerator(csv_file = "dataset_train.csv", batch_size = 64, normalize=True, transfo=False,
                                               replace_option=True)
    generator_train.metadata;
    
    generator_val = Quadriview_DataGenerator(csv_file = "dataset_val.csv", batch_size = 64, normalize=True, transfo=False,
                                                   replace_option=True)
    generator_val.metadata;
    
    config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 24} ) 
    sess = tf.Session(config=config) 
    K.set_session(sess)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    weights_path = "Documents/Python_files/NN_saved/"
    weights_list = [weights_path+"Q_softmax_weights_60/Q_weights_42-0.24.hdf5", weights_path+"Q_softmax_weights/Q_weights_20-0.11.hdf5",
    weights_path+"Q_softmax_dispatch/Q_weights_29-0.27.hdf5"]
    models_name = ["model_60", "model_120", "model_dispatch_10"]
    
    for k in range(2, 3):
        model = quadriview_model(name = models_name[k], weights = weights_list[k])
        model.compile(loss = 'binary_crossentropy', optimizer = Adam(), metrics=['accuracy'])
        model_eval(model, "dataset_test.csv")
        #model_apply(model, "dataset_test.csv", seed = 43)
        #test_noise_characs(model, "dataset_test.csv")
        #test_slice(model, "dataset_test.csv")    
    sess.close()
    del sess
