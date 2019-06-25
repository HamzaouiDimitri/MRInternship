#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 18:03:35 2019

@author: dimitri.hamzaoui
"""


from generators import Quadriview_DataGenerator
from models import BN_quadriview_model2

import tensorflow as tf
import keras.backend as K
from model_utils import summary_model_training, test_noise_characs, test_slice, model_display, ROC_curve, model_eval, model_eval_test
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, History, EarlyStopping, LearningRateScheduler
import os


    

generator_train = Quadriview_DataGenerator(csv_file = "dataset_train.csv", batch_size = 64, normalize="brain", transfo=True,
                                                prefix="/network/lustre/iss01/cenir/analyse/irm/users/dimitri.hamzaoui/", mask_files = "/network/lustre/iss01/cenir/analyse/irm/users/dimitri.hamzaoui/masks/", conditions = [("RMS", ">", 10), ("|", "noise", "==", 1)])
generator_train.metadata;
    
generator_val = Quadriview_DataGenerator(csv_file = "dataset_val.csv", batch_size = 64, normalize="brain", transfo=False,
                                                    prefix="/network/lustre/iss01/cenir/analyse/irm/users/dimitri.hamzaoui/", mask_files = "/network/lustre/iss01/cenir/analyse/irm/users/dimitri.hamzaoui/masks/", conditions = [("RMS", ">", 10), ("|", "noise", "==", 1)])
generator_val.metadata;

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
config = tf.ConfigProto(device_count = {'GPU': 1 , 'CPU': 12}, gpu_options=gpu_options )

sess = tf.Session(config=config) 
K.set_session(sess)
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
model = BN_quadriview_model2(name = "BN_on_brain_dispatch", weights = "/network/lustre/iss01/cenir/analyse/irm/users/dimitri.hamzaoui/NN_saved/Q_BN_brain_ds_mom/Q_weights_200-0.04.hdf5")
optimizer = Adam(lr = 1e-5, beta_1=0.9, beta_2=0.999)
early_stop = EarlyStopping(monitor='val_loss', patience=200)
hist = History()
#if not os.path.isdir("/network/lustre/iss01/cenir/analyse/irm/users/dimitri.hamzaoui/NN_saved/Q_BN_brain_ds_mom"):
#        os.mkdir("/network/lustre/iss01/cenir/analyse/irm/users/dimitri.hamzaoui/NN_saved/Q_BN_brain_ds_mom")
#saveoint = ModelCheckpoint(filepath="/network/lustre/iss01/cenir/analyse/irm/users/dimitri.hamzaoui/NN_saved/Q_BN_brain_ds_mom/Q_weights_{epoch:02d}-{val_loss:.2f}.hdf5", verbose=1, save_best_only=False)
#model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics=['accuracy'])
#model.fit_generator(generator_train, epochs = 200, validation_data=generator_val, verbose=1,
#                   use_multiprocessing = True, workers = 12, callbacks = [hist, early_stop, saveoint])

print("Training finished!")
#summary_model_training(model, prefix="/network/lustre/iss01/cenir/analyse/irm/users/dimitri.hamzaoui/")
#model_eval(model, "dataset_test.csv", prefix="/network/lustre/iss01/cenir/analyse/irm/users/dimitri.hamzaoui/", normalize="brain",
#                                              replace_option=True, mask_files = "/network/lustre/iss01/cenir/analyse/irm/users/dimitri.hamzaoui/masks/", conditions = [("RMS", ">", 10), ("|", "noise", "==", 1)])
model_eval_test(model, size = 5000, prefix="/network/lustre/iss01/cenir/analyse/irm/users/dimitri.hamzaoui/", csv_file = "dataset_test.csv", normalize="brain", replace_option_noised=True, mask_files = "/network/lustre/iss01/cenir/analyse/irm/users/dimitri.hamzaoui/masks/", conditions = [("RMS", ">", 10), ("|", "noise", "==", 1)])
#model_display(model, csv_file = "dataset_test.csv", prefix="/network/lustre/iss01/cenir/analyse/irm/users/dimitri.hamzaoui/", batch_size = 64, normalize="brain",
#                                                 mask_files = "/network/lustre/iss01/cenir/analyse/irm/users/dimitri.hamzaoui/masks/", conditions = [("RMS", ">", 10), ("|", "noise", "==", 1)])
#test_noise_characs(model, "dataset_test.csv", prefix="/network/lustre/iss01/cenir/analyse/irm/users/dimitri.hamzaoui/", normalization="brain", conditions = [("RMS", ">", 10), ("|", "noise", "==", 1)])
#test_slice(model, "dataset_test.csv", prefix="/network/lustre/iss01/cenir/analyse/irm/users/dimitri.hamzaoui/", normalization="brain", conditions = [("RMS", ">", 10), ("|", "noise", "==", 1)])
#ROC_curve(model, prefix="/network/lustre/iss01/cenir/analyse/irm/users/dimitri.hamzaoui/", batch_size = 256, csv_file = "dataset_test.csv", normalize="brain",  replace_option=True, mask_files = "/network/lustre/iss01/cenir/analyse/irm/users/dimitri.hamzaoui/masks/", conditions = [("RMS", ">", 10), ("|", "noise", "==", 1)])    
#features_heatmap(model,prefix="/network/lustre/iss01/cenir/analyse/irm/users/dimitri.hamzaoui/",batch_size= 64, csv_file="dataset_test.csv",  normalize="brain",
#                 replace_option=True, mask_files = "/network/lustre/iss01/cenir/analyse/irm/users/dimitri.hamzaoui/masks/", conditions = [("RMS", ">", 20), ("|", "noise", "==", 1)])
sess.close()
del sess
