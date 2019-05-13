#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 17:03:03 2019

@author: dimitri.hamzaoui
"""

import numpy as np
import keras
import nibabel as nb
import os
import pandas as pd
import keras_preprocessing.image.affine_transformations as at
from scipy.ndimage import rotate, shift


class DataGenerator(keras.utils.Sequence):
    
    def __init__(self, img_root_dir, batch_size=32, n_classes=2, dims = (182, 182), channels = 1, shuffle=True,
                 transfo = False, normalize = False, view = "cor", args_transfo = (0.5, 0.5, 0, 0),
                 seed = None):
        self._img_root_dir = img_root_dir
        self._metadata = None
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self._dims = dims
        self._channels = channels
        self._transfo = transfo
        self._normalize = normalize
        self._view = view
        self._args = args_transfo
        self._seed = seed
        
        if self._seed is not None:
            np.random.seed(self._seed)

    def _create_metadata_(self):
        listim = []
        self._dirnames = [f for f in os.listdir(self._img_root_dir)
              if os.path.isdir(os.path.join(self._img_root_dir, f))]
        for dirc in self._dirnames:
             temp = os.listdir(os.path.join(self._img_root_dir, dirc))
             f = lambda title: os.path.join(self._img_root_dir, dirc, title)
             listim += list(map(f, temp))   
        noised = []
        for row in listim:
            if row[-2:] == "gz":
                noised.append(1)
            else:
                noised.append(0)
        self._metadata = pd.DataFrame(np.column_stack([listim, noised]),
                                              columns=['img_file' ,'noise'])
        self.n_samples = len(self._metadata)
    
    def _transfo_img(self, image, args): # Types of transformations and range inspired by Sujit 2019
        img = image         
        if np.random.rand(1)[0] < args[0]:
            angle = 20
            img = at.random_rotation(img, angle, row_axis=0, col_axis=1, channel_axis=2)
        if np.random.rand(1)[0] < args[1]:
            axs_0 = 21
            axs_1 = 6
            img = at.random_shift(img, axs_0, axs_1, row_axis=0, col_axis=1, channel_axis=2)
        if np.random.rand(1)[0] < args[2]:
            axs_0 = 0.2
            axs_1 = 0.2
            img = at.random_zoom(img, (axs_0, axs_1), row_axis=0, col_axis=1, channel_axis=2)                   
        if np.random.rand(1)[0] < args[3]:
            lim_inf= 0
            lim_sup = 1
            img = at.random_brightness(img, (lim_inf, lim_sup))
        return img               
       
    
    def _normalization_func(self, img):
        vmin, vmax = img.min(), img.max()
        im = (img - vmin)/(vmax - vmin)
        return im
    
    def _take_slice_img(self, img_3D):
        img_shape = img_3D.shape
        if self._view == "sag":
            slice_pos = np.random.randint(int(0.2*img_shape[0]), int(0.8*img_shape[0]))
            img_2D = img_3D[slice_pos, :, :]
        elif self._view == "cor":
            slice_pos = np.random.randint(int(0.2*img_shape[1]), int(0.8*img_shape[1]))
            img_2D = img_3D[:, slice_pos, :]
        else:
            slice_pos = np.random.randint(int(0.2*img_shape[2]), int(0.8*img_shape[2]))
            img_2D = img_3D[:, :, slice_pos]
        img_2D = np.expand_dims(img_2D, 2)        
        return img_2D
    
    def _read_nii_file(self, file_name):
        return nb.load(file_name).get_fdata().astype('float32')
    
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self._metadata) / self.batch_size))

    def __data_generation(self, list_IDs_temp):
      'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
      # Initialization
      X = np.empty((self.batch_size, *self._dims, self._channels))
      y = np.empty((self.batch_size), dtype=int)
    
      # Generate data
      for i, ID in enumerate(list_IDs_temp):
          # Store sample
        image = self._read_nii_file(self._metadata.iloc[ID].img_file)
        
#        if self._img_shape is None:
#            self._img_shape = image.shape
        
        image = self._take_slice_img(image)
        
        
        if self._normalize:
            image = self._normalization_func(image)

        if self._transfo:
            image = self._transfo_img(image, self._args)
            
        
        X[i, ] = image
    
          # Store class
        y[i] = self._metadata.iloc[ID].noise
    
      return (X, keras.utils.to_categorical(y, num_classes=self.n_classes)) 

    def __getitem__(self, index):
      'Generate one batch of data'
      # Generate indexes of the batch
    
      # Find list of IDs
      list_IDs_temp = np.random.choice(len(self._metadata), size=self.batch_size,replace=False)
    
      # Generate data
      X, y = self.__data_generation(list_IDs_temp)
    
      return (X, y) 
    
    @property
    def metadata(self):
      if self._metadata is None:
        self._create_metadata_()
      return self._metadata
    
    @metadata.setter
    def metadata(self):
      raise ValueError('metadata cannot be set')

class Balanced_DataGenerator(keras.utils.Sequence):
    
    def __init__(self, img_root_dir, batch_size=32, n_classes=2, dims = (182, 182), channels = 1, shuffle=True,
                 transfo = False, normalize = False, view = "cor", replace_option = False, args_transfo = (0.5, 0, 0, 0),
                 seed=None):
        self._img_root_dir = img_root_dir
        self._metadata = None
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self._dims = dims
        self._channels = channels
        self._transfo = transfo
        self._normalize = normalize
        self._view = view
        self.ind_over = None
        self.ind_under = None
        self.yy_train = None
        self._replace = replace_option
        self._args = args_transfo
        self._seed = seed
        
        if self._seed is not None:
            np.random.seed(self._seed)
        
    def _create_metadata_(self):
        listim = []
        self._dirnames = [f for f in os.listdir(self._img_root_dir)
              if os.path.isdir(os.path.join(self._img_root_dir, f))]
        for dirc in self._dirnames:
             temp = os.listdir(os.path.join(self._img_root_dir, dirc))
             f = lambda title: os.path.join(self._img_root_dir, dirc, title)
             listim += list(map(f, temp))   
        noised = []
        for row in listim:
            if row[-2:] == "gz":
                noised.append(1)
            else:
                noised.append(0)
        self._metadata = pd.DataFrame(np.column_stack([listim, noised]),
                                              columns=['img_file' ,'noise'])
        self.yy_train = self._metadata["noise"].values.astype(int)
        self.ind_over = list(np.where(self.yy_train == 0)[0])
        self.ind_under = list(np.where(self.yy_train == 1)[0])
        self.n_samples = len(self._metadata)
    
    def _transfo_img(self, image, args): # Types of transformations and range inspired by Sujit 2019
        img = image         
        if np.random.rand(1)[0] < args[0]:
            angle = 10
            img = at.random_rotation(img, angle, row_axis=0, col_axis=1, channel_axis=2)
        if np.random.rand(1)[0] < args[1]:
            axs_0 = 21
            axs_1 = 6
            img = at.random_shift(img, axs_0, axs_1, row_axis=0, col_axis=1, channel_axis=2)            
        if np.random.rand(1)[0] < args[2]:
            axs_0 = 0.8
            axs_1 = 0.8
            img = at.random_zoom(img, (axs_0, axs_1), row_axis=0, col_axis=1, channel_axis=2)
        if np.random.rand(1)[0] < args[3]:
            lim_inf= 0
            lim_sup = 1
            img = at.random_brightness(img, (lim_inf, lim_sup))
        return img               
    
    def _normalization_func(self, img):
        vmin, vmax = img.min(), img.max()
        im = (img - vmin)/(vmax - vmin)
        return im
    
    def _take_slice_img(self, img_3D):
        img_shape = img_3D.shape
        if self._view == "sag":
            slice_pos = np.random.randint(int(0.2*img_shape[0]), int(0.8*img_shape[0]))
            img_2D = img_3D[slice_pos, :, :]
        elif self._view == "cor":
            slice_pos = np.random.randint(int(0.2*img_shape[1]), int(0.8*img_shape[1]))
            img_2D = img_3D[:, slice_pos, :]
        else:
            slice_pos = np.random.randint(int(0.2*img_shape[2]), int(0.8*img_shape[2]))
            img_2D = img_3D[:, :, slice_pos]
        img_2D = np.expand_dims(img_2D, 2)        
        return img_2D
    
    def _read_nii_file(self, file_name):
        return nb.load(file_name).get_fdata().astype('float32')
    
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self._metadata) / self.batch_size))

    def __data_generation(self, list_IDs_temp):
      'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
      # Initialization
      X = np.empty((self.batch_size, *self._dims, self._channels))
      y = np.empty((self.batch_size), dtype=int)
    
      # Generate data
      for i, ID in enumerate(list_IDs_temp):
          # Store sample
        image = self._read_nii_file(self._metadata.iloc[ID].img_file)
        
#        if self._img_shape is None:
#            self._img_shape = image.shape
        
        image = self._take_slice_img(image)
        

        if self._normalize:
            image = self._normalization_func(image)

        if self._transfo:
            image = self._transfo_img(image, self._args)
            
        X[i, ] = image
    
          # Store class
        y[i] = self._metadata.iloc[ID].noise
    
      return (X, keras.utils.to_categorical(y, num_classes=self.n_classes)) 

    def __getitem__(self, index):
      'Generate one batch of data'
      # Generate indexes of the batch
    
      # Find list of IDs
        
      idd_over = np.random.choice(self.ind_over ,size=self.batch_size//2, replace=False)
      idd_under = np.random.choice(self.ind_under ,size=self.batch_size//2, replace=self._replace)
      list_IDs_temp = (np.concatenate((idd_over, idd_under)))
      np.random.shuffle(list_IDs_temp)
    
    
      # Generate data
      X, y = self.__data_generation(list_IDs_temp)
    
      return (X, y) 
    
    @property
    def metadata(self):
      if self._metadata is None:
        self._create_metadata_()
      return self._metadata
    
    @metadata.setter
    def metadata(self):
      raise ValueError('metadata cannot be set')

class Quadriview_DataGenerator(keras.utils.Sequence):
    
    def __init__(self, img_root_dir, batch_size=32, n_classes=2, dims = (400, 400), channels = 1, shuffle=True,
                 transfo = False, normalize = False, view = "cor", replace_option = False, args_transfo = (0.5, 0.5, 0.5),
                 seed=None):
        self._img_root_dir = img_root_dir
        self._metadata = None
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self._dims = dims
        self._channels = channels
        self._transfo = transfo
        self._normalize = normalize
        self._view = view
        self.ind_over = None
        self.ind_under = None
        self.yy_train = None
        self._replace = replace_option
        self._args = args_transfo
        self._seed = seed
        
        if self._seed is not None:
            np.random.seed(self._seed)
        
    def _create_metadata_(self):
        listim = []
        self._dirnames = [f for f in os.listdir(self._img_root_dir)
              if os.path.isdir(os.path.join(self._img_root_dir, f))]
        for dirc in self._dirnames:
             temp = os.listdir(os.path.join(self._img_root_dir, dirc))
             f = lambda title: os.path.join(self._img_root_dir, dirc, title)
             listim += list(map(f, temp))   
        noised = []
        for row in listim:
            if row[-2:] == "gz":
                noised.append(1)
            else:
                noised.append(0)
        self._metadata = pd.DataFrame(np.column_stack([listim, noised]),
                                              columns=['img_file' ,'noise'])
        self.yy_train = self._metadata["noise"].values.astype(int)
        self.ind_over = list(np.where(self.yy_train == 0)[0])
        self.ind_under = list(np.where(self.yy_train == 1)[0])
        self.n_samples = len(self._metadata)
    
    def _transfo_img(self, image, args): # Inspired by Sujit 2019
        img = image         
        if np.random.rand(1)[0] < args[0]:
            angle = np.random.randint(-10, 11)
            not_axs = np.random.randint(0, 3)
            axis_rot = tuple([k for k in range(3) if k != not_axs])
            img = rotate(img, angle, axes=axis_rot, reshape=False)
#        if np.random.rand(1)[0] < args[1]:
#            axs_0 = np.random.uniform(-0.2, 0.2)
#            axs_1 = np.random.uniform(-0.2, 0.2)
#            img = zoom(img, [axs_0, axs_1])
        if np.random.rand(1)[0] < args[2]:
            axs_0 = np.random.randint(0, 21)
            axs_1 = np.random.randint(-5, 6)
            axs_2 = np.random.randint(-5, 5)
            img = shift(img, [axs_0, axs_1, axs_2])
    
        return img               
    
    def _normalization_func(self, img):
        vmin, vmax = img.min(), img.max()
        im = (img - vmin)/(vmax - vmin)
        return im
    
    def _quadriview(self, nifti_image, slice_sag, slice_orth, 
                   slice_cor_1, slice_cor_2):
        view_1 = nifti_image[slice_sag, :, :]
        view_2 = nifti_image[:, slice_orth, :]
        view_3 = nifti_image[:, :, slice_cor_1]
        view_4 = nifti_image[:, :,  slice_cor_2]
        pad_lign = max(view_1.shape[0] +view_2.shape[0], view_3.shape[0] + view_4.shape[0])
        pad_col = max(view_1.shape[1] + view_3.shape[1], view_2.shape[1] + view_4.shape[1])
        pad = np.zeros((pad_lign, pad_col))
        pad[:view_1.shape[0],  :view_1.shape[1]] = view_1
        pad[-view_2.shape[0]:, :view_2.shape[1]] = view_2
        pad[:view_3.shape[0], -view_3.shape[1]:] = view_3
        pad[-view_4.shape[0]:,-view_4.shape[1]:] = view_4
        return pad
    
    def _read_nii_file(self, file_name):
        return nb.load(file_name).get_fdata().astype('float32')
    
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self._metadata) / self.batch_size))

    def __data_generation(self, list_IDs_temp):
      'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
      # Initialization
      X = np.empty((self.batch_size, *self._dims, self._channels))
      y = np.empty((self.batch_size), dtype=int)
    
      # Generate data
      for i, ID in enumerate(list_IDs_temp):
          # Store sample
        image = self._read_nii_file(self._metadata.iloc[ID].img_file)
        
#        if self._img_shape is None:
#            self._img_shape = image.shape
        
        if self._transfo:
            image = self._transfo_img(image, self._args)
            
        if self._normalize:
            image = self._normalization_func(image)
        
        slice_sag = np.random.randint(0.2*image.shape[0], 0.8*image.shape[0])
        slice_orth = np.random.randint(0.2*image.shape[1], 0.8*image.shape[1])
        slice_cor_1 = np.random.randint(0.2*image.shape[2], 0.5*image.shape[2])
        slice_cor_2 = np.random.randint(0.5*image.shape[2], 0.8*image.shape[2])
        image = self._quadriview(image, slice_sag,  slice_orth,
                                 slice_cor_1, slice_cor_2)
        image = np.expand_dims(image, 2)        
        X[i, ] = image
    
          # Store class
        y[i] = self._metadata.iloc[ID].noise
    
      return (X, keras.utils.to_categorical(y, num_classes=self.n_classes)) 

    def __getitem__(self, index):
      'Generate one batch of data'
      # Generate indexes of the batch
    
      # Find list of IDs
        
      idd_over = np.random.choice(self.ind_over ,size=self.batch_size//2, replace=False)
      idd_under = np.random.choice(self.ind_under ,size=self.batch_size//2, replace=self._replace)
      list_IDs_temp = (np.concatenate((idd_over, idd_under)))
      np.random.shuffle(list_IDs_temp)
    
    
      # Generate data
      X, y = self.__data_generation(list_IDs_temp)
    
      return (X, y) 
    
    @property
    def metadata(self):
      if self._metadata is None:
        self._create_metadata_()
      return self._metadata
    
    @metadata.setter
    def metadata(self):
      raise ValueError('metadata cannot be set')


class Unbalanced_Quadriview_DataGenerator(keras.utils.Sequence):
    
    def __init__(self, img_root_dir, batch_size=32, n_classes=2, dims = (400, 400), channels = 1, shuffle=True,
                 transfo = False, normalize = False, view = "cor", replace_option = False, args_transfo = (0.5, 0.5, 0.5),
                 seed=None):
        self._img_root_dir = img_root_dir
        self._metadata = None
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self._dims = dims
        self._channels = channels
        self._transfo = transfo
        self._normalize = normalize
        self._view = view
        self.ind_over = None
        self.ind_under = None
        self.yy_train = None
        self._replace = replace_option
        self._args = args_transfo
        self._seed = seed
        
        if self._seed is not None:
            np.random.seed(self._seed)
        
    def _create_metadata_(self):
        listim = []
        self._dirnames = [f for f in os.listdir(self._img_root_dir)
              if os.path.isdir(os.path.join(self._img_root_dir, f))]
        for dirc in self._dirnames:
             temp = os.listdir(os.path.join(self._img_root_dir, dirc))
             f = lambda title: os.path.join(self._img_root_dir, dirc, title)
             listim += list(map(f, temp))   
        noised = []
        for row in listim:
            if row[-2:] == "gz":
                noised.append(1)
            else:
                noised.append(0)
        self._metadata = pd.DataFrame(np.column_stack([listim, noised]),
                                              columns=['img_file' ,'noise'])
        self.yy_train = self._metadata["noise"].values.astype(int)
        self.ind_over = list(np.where(self.yy_train == 0)[0])
        self.ind_under = list(np.where(self.yy_train == 1)[0])
        self.n_samples = len(self._metadata)
    
    def _transfo_img(self, image, args): # Inspired by Sujit 2019
        img = image         
        if np.random.rand(1)[0] < args[0]:
            angle = np.random.randint(-10, 11)
            not_axs = np.random.randint(0, 3)
            axis_rot = tuple([k for k in range(3) if k != not_axs])
            img = rotate(img, angle, axes=axis_rot, reshape=False)
#        if np.random.rand(1)[0] < args[1]:
#            axs_0 = np.random.uniform(-0.2, 0.2)
#            axs_1 = np.random.uniform(-0.2, 0.2)
#            img = zoom(img, [axs_0, axs_1])
        if np.random.rand(1)[0] < args[2]:
            axs_0 = np.random.randint(0, 21)
            axs_1 = np.random.randint(-5, 6)
            axs_2 = np.random.randint(-5, 5)
            img = shift(img, [axs_0, axs_1, axs_2])
    
        return img               
    
    def _normalization_func(self, img):
        vmin, vmax = img.min(), img.max()
        im = (img - vmin)/(vmax - vmin)
        return im
    
    def _quadriview(self, nifti_image, slice_sag, slice_orth, 
                   slice_cor_1, slice_cor_2):
        view_1 = nifti_image[slice_sag, :, :]
        view_2 = nifti_image[:, slice_orth, :]
        view_3 = nifti_image[:, :, slice_cor_1]
        view_4 = nifti_image[:, :,  slice_cor_2]
        pad_lign = max(view_1.shape[0] +view_2.shape[0], view_3.shape[0] + view_4.shape[0])
        pad_col = max(view_1.shape[1] + view_3.shape[1], view_2.shape[1] + view_4.shape[1])
        pad = np.zeros((pad_lign, pad_col))
        pad[:view_1.shape[0],  :view_1.shape[1]] = view_1
        pad[-view_2.shape[0]:, :view_2.shape[1]] = view_2
        pad[:view_3.shape[0], -view_3.shape[1]:] = view_3
        pad[-view_4.shape[0]:,-view_4.shape[1]:] = view_4
        return pad
    
    def _read_nii_file(self, file_name):
        return nb.load(file_name).get_fdata().astype('float32')
    
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self._metadata) / self.batch_size))

    def __data_generation(self, list_IDs_temp):
      'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
      # Initialization
      X = np.empty((self.batch_size, *self._dims, self._channels))
      y = np.empty((self.batch_size), dtype=int)
    
      # Generate data
      for i, ID in enumerate(list_IDs_temp):
          # Store sample
        image = self._read_nii_file(self._metadata.iloc[ID].img_file)
        
#        if self._img_shape is None:
#            self._img_shape = image.shape
        
        
        
        if self._transfo:
            image = self._transfo_img(image, self._args)
            
        if self._normalize:
            image = self._normalization_func(image)
        
        slice_sag = np.random.randint(0.2*image.shape[0], 0.8*image.shape[0])
        slice_orth = np.random.randint(0.2*image.shape[1], 0.8*image.shape[1])
        slice_cor_1 = np.random.randint(0.2*image.shape[2], 0.5*image.shape[2])
        slice_cor_2 = np.random.randint(0.5*image.shape[2], 0.8*image.shape[2])
        image = self._quadriview(image, slice_sag,  slice_orth,
                                 slice_cor_1, slice_cor_2)
        X[i, ] = np.expand_dims(image, 2)
    
          # Store class
        y[i] = self._metadata.iloc[ID].noise
    
      return (X, keras.utils.to_categorical(y, num_classes=self.n_classes)) 

    def __getitem__(self, index):
      'Generate one batch of data'
      # Generate indexes of the batch
    
      # Find list of IDs
      list_IDs_temp = np.random.choice(len(self._metadata), size=self.batch_size,replace=False)
    
      # Generate data
      X, y = self.__data_generation(list_IDs_temp)
    
      return (X, y) 
    
    @property
    def metadata(self):
      if self._metadata is None:
        self._create_metadata_()
      return self._metadata
    
    @metadata.setter
    def metadata(self):
      raise ValueError('metadata cannot be set')


class Quadriview_always_true_DataGenerator(keras.utils.Sequence):
    
    def __init__(self, img_root_dir, batch_size=32, n_classes=2, dims = (400, 400), channels = 1, shuffle=True,
                 transfo = False, normalize = False, view = "cor", replace_option = False, args_transfo = (0.5, 0.5, 0.5),
                 seed=None):
        self._img_root_dir = img_root_dir
        self._metadata = None
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self._dims = dims
        self._channels = channels
        self._transfo = transfo
        self._normalize = normalize
        self._view = view
        self.ind_over = None
        self.ind_under = None
        self.yy_train = None
        self._replace = replace_option
        self._args = args_transfo
        self._seed = seed
        
        if self._seed is not None:
            np.random.seed(self._seed)
        
    def _create_metadata_(self):
        listim = []
        self._dirnames = [f for f in os.listdir(self._img_root_dir)
              if os.path.isdir(os.path.join(self._img_root_dir, f))]
        for dirc in self._dirnames:
             temp = os.listdir(os.path.join(self._img_root_dir, dirc))
             f = lambda title: os.path.join(self._img_root_dir, dirc, title)
             listim += list(map(f, temp))   
        noised = []
        for row in listim:
            if row[-2:] == "gz":
                noised.append(1)
            else:
                noised.append(0)
        self._metadata = pd.DataFrame(np.column_stack([listim, noised]),
                                              columns=['img_file' ,'noise'])
        self.yy_train = self._metadata["noise"].values.astype(int)
        self.ind_over = list(np.where(self.yy_train == 0)[0])
        self.ind_under = list(np.where(self.yy_train == 1)[0])
        self.n_samples = len(self._metadata)
    
    def _transfo_img(self, image, args): # Inspired by Sujit 2019
        img = image         
        if np.random.rand(1)[0] < args[0]:
            angle = np.random.randint(-10, 11)
            not_axs = np.random.randint(0, 3)
            axis_rot = tuple([k for k in range(3) if k != not_axs])
            img = rotate(img, angle, axes=axis_rot, reshape=False)
#        if np.random.rand(1)[0] < args[1]:
#            axs_0 = np.random.uniform(-0.2, 0.2)
#            axs_1 = np.random.uniform(-0.2, 0.2)
#            img = zoom(img, [axs_0, axs_1])
        if np.random.rand(1)[0] < args[2]:
            axs_0 = np.random.randint(0, 21)
            axs_1 = np.random.randint(-5, 6)
            axs_2 = np.random.randint(-5, 5)
            img = shift(img, [axs_0, axs_1, axs_2])
    
        return img               
    
    def _normalization_func(self, img):
        vmin, vmax = img.min(), img.max()
        im = (img - vmin)/(vmax - vmin)
        return im
    
    def _quadriview(self, nifti_image, slice_sag, slice_orth, 
                   slice_cor_1, slice_cor_2):
        view_1 = nifti_image[slice_sag, :, :]
        view_2 = nifti_image[:, slice_orth, :]
        view_3 = nifti_image[:, :, slice_cor_1]
        view_4 = nifti_image[:, :,  slice_cor_2]
        pad_lign = max(view_1.shape[0] +view_2.shape[0], view_3.shape[0] + view_4.shape[0])
        pad_col = max(view_1.shape[1] + view_3.shape[1], view_2.shape[1] + view_4.shape[1])
        pad = np.zeros((pad_lign, pad_col))
        pad[:view_1.shape[0],  :view_1.shape[1]] = view_1
        pad[-view_2.shape[0]:, :view_2.shape[1]] = view_2
        pad[:view_3.shape[0], -view_3.shape[1]:] = view_3
        pad[-view_4.shape[0]:,-view_4.shape[1]:] = view_4
        return pad
    
    def _read_nii_file(self, file_name):
        return nb.load(file_name).get_fdata().astype('float32')
    
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self._metadata) / self.batch_size))

    def __data_generation(self, list_IDs_temp):
      'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
      # Initialization
      X = np.empty((self.batch_size, *self._dims, self._channels))
      y = np.empty((self.batch_size), dtype=int)
    
      # Generate data
      for i, ID in enumerate(list_IDs_temp):
          # Store sample
        image = self._read_nii_file(self._metadata.iloc[ID].img_file)
        
#        if self._img_shape is None:
#            self._img_shape = image.shape
        
        
        
        if self._transfo:
            image = self._transfo_img(image, self._args)
            
        if self._normalize:
            image = self._normalization_func(image)
        
        slice_sag = np.random.randint(0.2*image.shape[0], 0.8*image.shape[0])
        slice_orth = np.random.randint(0.2*image.shape[1], 0.8*image.shape[1])
        slice_cor_1 = np.random.randint(0.2*image.shape[2], 0.5*image.shape[2])
        slice_cor_2 = np.random.randint(0.5*image.shape[2], 0.8*image.shape[2])
        image = self._quadriview(image, slice_sag,  slice_orth,
                                 slice_cor_1, slice_cor_2)
        image = np.expand_dims(image, 2)        
        X[i, ] = image
    
          # Store class
        y[i] = self._metadata.iloc[ID].noise
    
      return (X, keras.utils.to_categorical(y, num_classes=self.n_classes)) 

    def __getitem__(self, index):
      'Generate one batch of data'
      # Generate indexes of the batch
    
      # Find list of IDs
        
      #idd_over = np.random.choice(self.ind_over ,size=self.batch_size//2, replace=False)
      idd_under = np.random.choice(self.ind_under ,size=self.batch_size, replace=True)
      #list_IDs_temp = (np.concatenate((idd_over, idd_under)))
      #np.random.shuffle(list_IDs_temp)
    
    
      # Generate data
      X, y = self.__data_generation(idd_under)
    
      return (X, y) 
    
    @property
    def metadata(self):
      if self._metadata is None:
        self._create_metadata_()
      return self._metadata
    
    @metadata.setter
    def metadata(self):
      raise ValueError('metadata cannot be set')
