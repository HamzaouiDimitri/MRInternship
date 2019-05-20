#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 17:03:03 2019

@author: dimitri.hamzaoui
"""

import numpy as np
import keras
import nibabel as nb
import pandas as pd
import keras_preprocessing.image.affine_transformations as at
from utils import quadriview, apply_conditions_on_dataset, take_slice, normalization_func
from generate_df import generate_df
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw

font_path = "/usr/share/fonts/truetype/freefont/FreeMono.ttf"

class Quadriview_DataGenerator(keras.utils.Sequence):
    """
    Generates quadriview images.
    Takes as source input: the path to a csv file or the path to a directory containing the images.
    Options are:
        +Batch size. Default is 64
        +Number of output classes (Default = 2)
        +Dimension of the generated images
        +Number of channels of the images
        +shuffle: Says if we shuffle the lists containing the indexes between each epoch. True by default
        +transfo: Bool option indicating if we apply transformations to the images (rotation or translation). False by default
        +normalize: Indicating if we normalize the 3D images between 0 and 1. True by default
        +replace_option: Indicates if we can select several times the same clean images in a same mini-batch. False by default
        +args_transfo: Probabilities to apply (rotation, translation) if transfo = True.
        +seed: random seed (for reproductibility)
        +generator_type: "balanced" (50% clean images, 50% noised images), always_clean, always_noised, 
        fully_random (takes images without considering the label)". Balanced by default.
        +conditions: conditions to restrein the dataset (ex: only images with RMS > 10 => ("RMS", ">", 10)
        +prefix: To use in order to access the images from the csv_file in case of problem
    """
    def __init__(self, csv_file=None, img_root_dir=None, batch_size=64, n_classes=2, dims = (400, 400), channels = 1, shuffle=True,
                 transfo = False, normalize = True, replace_option = True, args_transfo = (0.5, 0),
                 seed=None, generator_type = "balanced", conditions = None, prefix = ""):
        
        self._csv_file = csv_file
        self._img_root_dir = img_root_dir
        self._metadata = None
        self.batch_size = batch_size
        self.n_classes = n_classes
        self._dims = dims
        self._channels = channels
        self._transfo = transfo
        self._normalize = normalize
        self.ind_zero = None
        self.ind_one = None
        self.yy_train = None
        self._replace = replace_option
        self._args = args_transfo
        self._seed = seed
        self._type = generator_type
        self._conditions = conditions
        self._prefix = prefix
        self._shuffle = shuffle
        
        if self._seed is not None:
            np.random.seed(self._seed)
            
        if self._type not in ["balanced", "always_clean", "always_noised", "fully_random", "fifty_fifty"]:
            raise ValueError("generator_type must be: balanced, always_clean, always_noised or fully_random")
        
    def _create_metadata_(self):
        """
        Creates the self._metadata used after, either by reading a csv_file or by creating the dataframe with generate_df, then apply the conditions
        to restrein the dataframe, and create the following atributes: yy_train, ind_zero (indexes in the dataframe of the noised images),
        ind_one (indexes in the dataframe of the clean images), n_samples (number of images)
        """
        if self._csv_file is None:
            metadata = generate_df(self._prefix + self._img_root_dir)
        else:
            metadata = pd.read_csv(self._prefix + self._csv_file, sep = ",", index_col = 0)
        if self._conditions is not None:
            metadata = metadata[apply_conditions_on_dataset(metadata, self._conditions)]
        self._metadata = metadata
        self.yy_train = self._metadata["noise"].values.astype(int)
        self.ind_zero = list(np.where(self.yy_train == 0)[0])
        self.ind_one = list(np.where(self.yy_train == 1)[0])
        self.n_samples = len(self._metadata)
    
    def _transfo_img(self, image, args): # Types of transformations and range inspired by Sujit 2019
        """
        Takes as input an image and transforms it (currently rotation and translation available)
        """  
        img = image         
        if np.random.rand(1)[0] < args[0]:
            angle = 10
            img = at.random_rotation(img, angle, row_axis=0, col_axis=1, channel_axis=2)
        if np.random.rand(1)[0] < args[1]:
            axs_0 = 21
            axs_1 = 6
            img = at.random_shift(img, axs_0, axs_1, row_axis=0, col_axis=1, channel_axis=2) 
        return img
    
    
    
    def _read_nii_file(self, file_name):
        return nb.load(file_name).get_fdata().astype('float32')
    
    
    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        return int(np.floor(len(self._metadata) / self.batch_size))

    def __data_generation(self, list_IDs_temp):
        """
        Generates data containing batch_size samples
        """
        # Initialization
        X = np.empty((self.batch_size, *self._dims, self._channels))
        y = np.empty((self.batch_size), dtype=int)
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
          # Store sample
          image = self._read_nii_file(self._prefix + self._metadata.iloc[ID].img_file)
          if self._transfo:
              image = self._transfo_img(image, self._args)
          if self._normalize:
              image = normalization_func(image)
          slice_sag = np.random.randint(0.2*image.shape[0], 0.8*image.shape[0])
          slice_orth = np.random.randint(0.2*image.shape[1], 0.8*image.shape[1])
          slice_cor_1 = np.random.randint(0.2*image.shape[2], 0.5*image.shape[2])
          slice_cor_2 = np.random.randint(0.5*image.shape[2], 0.8*image.shape[2])
          image = quadriview(image, slice_sag,  slice_orth,
                                 slice_cor_1, slice_cor_2)
          image = np.expand_dims(image, 2)        
          X[i, ] = image
          # Store class
          y[i] = self._metadata.iloc[ID].noise
      
        return (X, keras.utils.to_categorical(y, num_classes=self.n_classes)) 

    def __getitem__(self, index):
        """
        Generate one batch of data
        """
        # Generate indexes of the batch 
        if self._type == "balanced":
            idd_zero = np.random.choice(self.ind_zero ,size=self.batch_size//2, replace=False)
            idd_one = np.random.choice(self.ind_one ,size=self.batch_size//2, replace=self._replace)
            list_IDs_temp = (np.concatenate((idd_zero, idd_one)))
        elif self._type == "always_clean":
            list_IDs_temp = np.random.choice(self.ind_one ,size=self.batch_size, replace=True)
        elif self._type == "always_noised":
            list_IDs_temp = np.random.choice(self.ind_zero ,size=self.batch_size, replace=False)
        else:
            list_IDs_temp = np.random.choice(np.arange(self.n_samples) ,size=self.batch_size, replace=False)
        np.random.shuffle(list_IDs_temp)
      # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return (X, y) 
    
    def on_epoch_end(self):
        if self._shuffle:
            np.random.shuffle(self.ind_zero)
            np.random.shuffle(self.ind_one)
    
    def display(self, name = "display.jpeg", orientation = "vertical", abs_pos = 185):
        """
        Generates a jpeg or png file with a batch of data generated by the generator and the corresponding class for eacy of them.
        """
        X, y = self.__getitem__(0)
        X = X[:, :, :, 0]
        if orientation == "horizontal":
            X = np.hstack(tuple([X[k].T for k in range(X.shape[0])]))
        elif orientation == "vertical":
            X = np.vstack(tuple([X[k].T for k in range(X.shape[0]-1, -1, -1)]))
        #X = (255*X).astype(int)
        plt.imsave(name, X, origin = "lower", cmap = "Greys_r")
        img = Image.open(name)
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(font_path, 8, encoding="unic")
        if orientation == "vertical":
            for k in range(len(y)):        
                draw.text((abs_pos, 5+400*k),"Class: "+str(np.argmax(y[k])), (255, 255, 255, 255), font=font)
        else:
            for k in range(len(y)):        
                draw.text((abs_pos+400*k, 5),"Class: "+str(np.argmax(y[k])), (255, 255, 255, 255), font=font)
        img.save(name)     
        return y
        
    @property
    def metadata(self):
      if self._metadata is None:
        self._create_metadata_()
      return self._metadata
    
    @metadata.setter
    def metadata(self):
      raise ValueError('metadata cannot be set')

class DataGenerator(keras.utils.Sequence):
    """
    Generates slices images.
    Takes as source input: the path to a csv file or the path to a directory containing the images.
    Options are:
        +Batch size
        +Number of output classes
        +Dimension of the generated images
        +Number of channels of the images
        +shuffle: Says if we shuffle the lists containing the indexes between each epoch. True by default
        +view: sag, cor or ax - type of view generated. Coronal by default
        +transfo: Bool option indicating if we apply transformations to the images (rotation or translation). False by default
        +normalize: Indicating if we normalize the 3D images between 0 and 1. True by default.
        +replace_option: Indicates if we can select several times the same clean images in a same mini-batch. False by default
        +args_transfo: Probabilities to apply (rotation, translation) if transfo = True
        +seed: random seed (for reproductibility)
        +generator_type: "balanced" (50% clean images, 50% noised images), always_clean, always_noised, 
        fully_random (takes images without considering the label). Balanced by default
        +conditions: conditions to restrein the daaset (ex: only images with RMS > 10 => ("RMS", ">", 10)
        +prefix: To use in order to access the images from the csv_file in case of problem 
    """
    def __init__(self, csv_file = None, img_root_dir= None, batch_size=32, n_classes=2, dims = (182, 182), channels = 1, shuffle=True,
                 view = "cor", transfo = False, args_transfo = (0.5, 0), normalize = True,  replace_option = False,
                 seed = None, generator_type = "balanced", conditions = None, prefix = ""):
        
        self._csv_file = csv_file
        self._img_root_dir = img_root_dir
        self._metadata = None
        self.batch_size = batch_size
        self.n_classes = n_classes
        self._dims = dims
        self._channels = channels
        self._transfo = transfo
        self._normalize = normalize
        self._view = view
        self.ind_zero = None
        self.ind_one = None
        self.yy_train = None
        self._replace = replace_option
        self._args = args_transfo
        self._seed = seed
        self._type = generator_type
        self._conditions = conditions
        self._prefix = prefix
        self._shuffle = shuffle
        
        if self._seed is not None:
            np.random.seed(self._seed)
        
     
        if self._type not in ["balanced", "always_clean", "always_noised", "fully_random"]:
            raise ValueError("generator_type must be: balanced, always_clean, always_noised or fully_random")    
    
    def _create_metadata_(self):
        """
        Creates the self._metadata used after, either by reading a csv_file or by creating the dataframe with generate_df, then apply the conditions
        to restrein the dataframe, and create the following atributes: yy_train, ind_zero (indexes in the dataframe of the noised images),
        ind_one (indexes in the dataframe of the clean images), n_samples (number of images)
        """
        if self._csv_file is None:
            metadata = generate_df(self._prefix + self._img_root_dir)
        else:
            metadata = pd.read_csv(self._prefix + self._csv_file, sep = ";", index_col = 0)
        if self._conditions is not None:
            metadata = metadata[apply_conditions_on_dataset(metadata, self._conditions)]
        self._metadata = metadata
        self.yy_train = self._metadata["noise"].values.astype(int)
        self.ind_zero = list(np.where(self.yy_train == 0)[0])
        self.ind_one = list(np.where(self.yy_train == 1)[0])
        self.n_samples = len(self._metadata)
    
    def _transfo_img(self, image, args): # Types of transformations and range inspired by Sujit 2019
        """
        Takes as input an image and transforms it (currently rotation and translation available)
        """        
        img = image         
        if np.random.rand(1)[0] < args[0]:
            angle = 10
            img = at.random_rotation(img, angle, row_axis=0, col_axis=1, channel_axis=2)
        if np.random.rand(1)[0] < args[1]:
            axs_0 = 21
            axs_1 = 6
            img = at.random_shift(img, axs_0, axs_1, row_axis=0, col_axis=1, channel_axis=2)
        return img               
    
    
    def _read_nii_file(self, file_name):
        """
        Takes as input a nifti file and return a numpy array file
        """
        return nb.load(file_name).get_fdata().astype('float32')
    
    
    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        return int(np.floor(len(self._metadata) / self.batch_size))

    def __data_generation(self, list_IDs_temp):
        """
        Generates data containing batch_size samples
        """ 
        # Initialization
        X = np.empty((self.batch_size, *self._dims, self._channels))
        y = np.empty((self.batch_size), dtype=int)
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
          # Store sample
          image = self._read_nii_file(self._prefix +self._metadata.iloc[ID].img_file)
          image = take_slice(image, self._view)
          if self._normalize:
              image = normalization_func(image)
          if self._transfo:
              image = self._transfo_img(image, self._args)
          X[i, ] = image
          # Store class
          y[i] = self._metadata.iloc[ID].noise  
        return (X, keras.utils.to_categorical(y, num_classes=self.n_classes)) 

    def __getitem__(self, index):
        """
        Generate one batch of data
        """
        if self._type == "balanced":
          idd_zero = np.random.choice(self.ind_zero ,size=self.batch_size//2, replace=False)
          idd_one = np.random.choice(self.ind_one ,size=self.batch_size//2, replace=self._replace)
          list_IDs_temp = (np.concatenate((idd_zero, idd_one)))
        elif self._type == "always_clean":
          list_IDs_temp = np.random.choice(self.ind_one ,size=self.batch_size, replace=True)
        elif self._type == "always_noised":
          list_IDs_temp = np.random.choice(self.ind_zero ,size=self.batch_size, replace=False)
        else:
          list_IDs_temp = np.random.choice(np.arange(self.n_samples) ,size=self.batch_size, replace=False)
        np.random.shuffle(list_IDs_temp)
        # Generate data
        X, y = self.__data_generation(list_IDs_temp) 
        return (X, y) 
    
    def on_epoch_end(self):
        if self._shuffle:
            np.random.shuffle(self.ind_zero)
            np.random.shuffle(self.ind_one)
        
    def display(self, name = "display.jpeg", orientation = "vertical", abs_pos = 20):
        """
        Generates a jpeg or png file with a batch of data generated by the generator and the corresponding class for eacy of them.
        """
        X, y = self.__getitem__(0)
        X = X[:, :, :, 0]
        if orientation == "horizontal":
            X = np.hstack(tuple([X[k].T for k in range(X.shape[0])]))
        elif orientation == "vertical":
            X = np.vstack(tuple([X[k].T for k in range(X.shape[0])]))
        #X = (255*X).astype(int)
        plt.imsave(name, X, origin = "lower", cmap = "Greys_r")
        img = Image.open(name)
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(font_path, 8)
        for k in range(len(y)):        
            draw.text((abs_pos, 5+182*k),"Class: "+str(np.argmax(y[k])), (255, 255, 255, 255), font=font)
        img.save(name)     
        return None
    
    
    @property
    def metadata(self):
      if self._metadata is None:
        self._create_metadata_()
      return self._metadata
    
    @metadata.setter
    def metadata(self):
      raise ValueError('metadata cannot be set')
    


        

