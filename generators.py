#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 17:56:29 2019

@author: dimitri.hamzaoui
"""
font_path = "/usr/share/fonts/truetype/freefont/FreeMono.ttf"

import numpy as np
import keras
from generate_df import generate_df
import pandas as pd
from utils import  apply_conditions_on_dataset, quadriview, take_slice, normalization_func
import nibabel as nb
import keras_preprocessing.image.affine_transformations as at
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
import abc
from scipy.ndimage import rotate, shift
import patches

class BasisGenerator(keras.utils.Sequence):
    __metaclass__  = abc.ABCMeta
    def __init__(self, dims, csv_file=None, img_root_dir=None, batch_size=64, n_classes=2, channels = 1, shuffle=True,
                 transfo = False, replace_option_clean = True, replace_option_noised = False, args_transfo = (0.5, 0),
                 seed=None, generator_type = "balanced", conditions = None, prefix = "", normalization = False):
        self._csv_file = csv_file
        self._img_root_dir = img_root_dir
        self._metadata = None
        self._batch_size = batch_size
        self._n_classes = n_classes
        self._dims = dims
        self._channels = channels
        self._transfo = transfo
        self._ind_zero = None
        self._ind_one = None
        self._yy_train = None
        self._replace_clean = replace_option_clean
        self._replace_noised = replace_option_noised
        self._args = args_transfo
        self._seed = seed
        self._type = generator_type
        self._conditions = conditions
        self._prefix = prefix
        self._shuffle = shuffle
        self._normalize = normalization
        
        if self._seed is not None:
            np.random.seed(self._seed)
            
        if self._type not in ["balanced", "always_clean", "always_noised", "fully_random", "fifty_fifty"]:
            raise ValueError("generator_type must be: balanced, always_clean, always_noised or fully_random")
        
        
    def _create_metadata_(self):
        """
        Creates the self.metadata used after, either by reading a csv_file or by creating the dataframe with generate_df, then apply the conditions
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
        self._yy_train = self._metadata["noise"].values.astype(int)
        self._ind_zero = list(np.where(self._yy_train == 0)[0])
        self._ind_one = list(np.where(self._yy_train == 1)[0])
        self._n_samples = len(self._metadata)
        
    
    def transfo_img(self, img):
        raise NotImplementedError("Please Implement this method")
    
    def _read_nii_file(self, file_name):
        return nb.load(file_name).get_fdata().astype('float32')
    
    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        return int(np.floor(len(self._metadata) / self._batch_size))

    def norm_func(self, img):
        return normalization_func(img)
    
    def _type_gestion(self, index):
        """
        Generate one batch of data
        """
        # Generate indexes of the batch 
        if self._type in ["balanced", "fifty_fifty"]:
            idd_zero = np.random.choice(self._ind_zero ,size= self._batch_size - self._batch_size//2, replace=self._replace_noised)
            idd_one = np.random.choice(self._ind_one ,size= self._batch_size//2, replace=self._replace_clean)
            list_IDs_temp = (np.concatenate((idd_zero, idd_one)))
        elif self._type == "always_clean":
            list_IDs_temp = np.random.choice(self._ind_one ,size=self._batch_size, replace=True)
        elif self._type == "always_noised":
            list_IDs_temp = np.random.choice(self._ind_zero ,size=self._batch_size, replace=False)
        else:
            list_IDs_temp = np.random.choice(np.arange(self._n_samples) ,size=self._batch_size, replace=False)
        if self._type != "fifty_fifty":
            np.random.shuffle(list_IDs_temp)
        return list_IDs_temp
    
    @abc.abstractmethod
    def __getitem__(self, index):
        'raise NotImplementedError("Please Implement this method")'
    
    
    
    @abc.abstractmethod
    def __data_generation(self):
        'raise NotImplementedError("Please Implement this method")'
    
    def on_epoch_end(self):
        if self._shuffle:
            np.random.shuffle(self._ind_zero)
            np.random.shuffle(self._ind_one)
    
    def display(self, abs_pos, name = "display.jpeg", orientation = "rect"):
        """
        Generates a jpeg or png file with a batch of data generated by the generator and the corresponding class for eacy of them.
        """
        X, y = self.__getitem__(0)
        X_return = X
        X = X[:, :, :, 0]
        if orientation == "horizontal":
            X = np.hstack(tuple([X[k].T for k in range(X.shape[0])]))
        elif orientation == "vertical":
            X = np.vstack(tuple([X[k].T for k in range(X.shape[0]-1, -1, -1)]))
        else:
            col = int(np.sqrt(self._batch_size))
            X = np.hstack(tuple([X[k].T for k in range(X.shape[0])]))
            if (X.shape[1]//self._dims[0])%col != 0:
                X = np.pad(X, ((0, 0), (0, col*((int((X.shape[1]//self._dims[0]))//col)+1)*self._dims[0] - X.shape[1])), 'constant', constant_values=(0, 0))
                clefable = [X[:, (self._dims[0]*col*k):(col*self._dims[0]*(k+1))] for k in range(((X.shape[1]//self._dims[0])//col)-1, -1, -1)]
                X = np.concatenate(clefable, axis = 0)
            else:
                X = np.concatenate([X[:, (self._dims[0]*col*k):(col*self._dims[0]*(k+1))] for k in range(((X.shape[1]//self._dims[0])//col)-1, -1, -1)], axis = 0)
        #X = (255*X).astype(int)
        plt.imsave(name, X, origin = "lower", cmap = "Greys_r")
        img = Image.open(name)
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(font_path, 8, encoding="unic")
        if orientation == "vertical":
            for k in range(len(y)):        
                draw.text((abs_pos, 5+self._dims[0]*k),"Class: "+str(np.argmax(y[k])), (255, 255, 255, 255), font=font)
        elif orientation == "horizontal":
            for k in range(len(y)):        
                draw.text((abs_pos+self._dims[1]*k, 5),"Class: "+str(np.argmax(y[k])), (255, 255, 255, 255), font=font)
        else:
            for k in range(len(y)):        
                draw.text((abs_pos+self._dims[0]*(k%col), 5 + self._dims[1]*(k//col)),"Class: "+str(np.argmax(y[k])), (255, 255, 255, 255), font=font)
        img.save(name)   
        return X_return, y
    
    def get_data(self):
        """
        Generates a jpeg or png file with a batch of data generated by the generator and the corresponding class for eacy of them.
        """
        X, y = self.__getitem__(0)
        return X, y


    @property
    def metadata(self):
      if self._metadata is None:
        self._create_metadata_()
      return self._metadata
    
    @metadata.setter
    def metadata(self):
      raise ValueError('metadata cannot be set')
      
class Generator_quadriview(BasisGenerator):
    def __init__(self, dims = (400, 400), **kwargs):
        super(Generator_quadriview, self).__init__(dims, **kwargs)
        
    def transfo_img(self, img):
        imge = img
        if np.random.rand() < self._args[0]:
           angle = 10
           axis = np.random.permutation([0, 1, 2])
           imge = at.random_rotation(imge, angle, row_axis=axis[0], col_axis=axis[1], channel_axis=axis[2])
        if np.random.rand() < self._args[1]:
           axs_0 = 5
           axs_1 = 5
           axis = np.random.permutation([0, 1, 2])
           imge = at.random_shift(imge, axs_0, axs_1, row_axis=axis[0], col_axis=axis[1], channel_axis=axis[2])
        return imge
    
    def __data_generation(self, list_IDs_temp):
        """
        Generates data containing batch_size samples
        """
        # Initialization
        X = np.empty((self._batch_size, *self._dims, self._channels))
        y = np.empty((self._batch_size), dtype=int)
        for i, ID in enumerate(list_IDs_temp):
            image = self._read_nii_file(self._prefix + self._metadata.iloc[ID].img_file)
            if self._transfo:
                image = self._transfo_img(image)
            if self._normalize:
                image = self.norm_func(image)
          

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
      
        return (X, keras.utils.to_categorical(y, num_classes=self._n_classes))
    
    def __getitem__(self, index):
    # Generate data
        list_IDs_temp = self._type_gestion(index)
        X, y = self.__data_generation(list_IDs_temp)
        return (X, y) 

    def display(self, abs_pos=185, **kwargs):
        return super(Generator_quadriview, self).display(abs_pos, **kwargs)
    
    
   

class Generator_slices(BasisGenerator):
    def __init__(self, dims = (182, 182), view = "cor", **kwargs):
        super(Generator_slices, self).__init__(dims, **kwargs)
        self._view = view
    
    def transfo_img(self, img):
        imge = img
        if np.random.rand() < self._args[0]:
           angle = np.random.randint(-10, 11)
           imge = rotate(imge, angle, reshape=False)
        if np.random.rand() < self._args[1]:
           axs_0 = 5
           axs_1 = 5
           imge = shift(imge, (axs_0, axs_1))
        return imge
    
    def __data_generation(self, list_IDs_temp):
        """
        Generates data containing batch_size samples
        """
        # Initialization
        X = np.empty((self._batch_size, *self._dims, self._channels))
        y = np.empty((self._batch_size), dtype=int)
        for i, ID in enumerate(list_IDs_temp):
            image = self._read_nii_file(self._prefix + self._metadata.iloc[ID].img_file)
            if self._transfo:
                image = self._transfo_img(image)
            if self._normalize:
                image = self.norm_func(image)

            
            image = take_slice(image, self._view)
            X[i, ] = image
            # Store class
            y[i] = self._metadata.iloc[ID].noise
      
        return (X, keras.utils.to_categorical(y, num_classes=self._n_classes))
    
    def __getitem__(self, index):
    # Generate data
        list_IDs_temp = self._type_gestion(index)
        X, y = self.__data_generation(list_IDs_temp)
        return (X, y) 
    
    def display(self, abs_pos=70, **kwargs):
        return super(Generator_slices, self).display(abs_pos, **kwargs)
    
class Generator_3D(BasisGenerator):
    def __init__(self, dims = (60, 60, 60), patches_per_vol = 5, **kwargs):
        super(Generator_3D, self).__init__(dims, **kwargs)
        self._patches_per_vol = patches_per_vol
    
    def transfo_img(self, img):
        imge = img
        if np.random.rand() < self._args[0]:
           angle = 10
           axis = np.random.permutation([0, 1, 2])
           imge = at.random_rotation(imge, angle, row_axis=axis[0], col_axis=axis[1], channel_axis=axis[2])
        if np.random.rand() < self._args[1]:
           axs_0 = 5
           axs_1 = 5
           axis = np.random.permutation([0, 1, 2])
           imge = at.random_shift(imge, axs_0, axs_1, row_axis=axis[0], col_axis=axis[1], channel_axis=axis[2])
        return imge
    
    def __data_generation(self, list_IDs_temp):
        """
        Generates data containing batch_size samples
        """
        # Initialization
        true_batch_size = self._batch_size*self._patches_per_vol
        X = np.empty((true_batch_size, *self._dims, self._channels))
        y = np.empty((true_batch_size), dtype=int)
        for i, ID in enumerate(list_IDs_temp):
            image = self._read_nii_file(self._prefix + self._metadata.iloc[ID].img_file)
            if self._transfo:
                image = self._transfo_img(image)
            if self._normalize:
                image = self.norm_func(image)
            temp_y = self._metadata.iloc[ID].noise
            # Store class
            patches_list = patches.make_random_3D_patches(image, self._dims, self._patches_per_vol)
            for j in range(self._patches_per_vol):
                X[i*self._patches_per_vol + j] = np.expand_dims(patches_list[j], 3)
                y[i*self._patches_per_vol + j] = temp_y
        p = np.random.permutation(X.shape[0])
        X = X[p]
        y = y[p]
      
        return (X, keras.utils.to_categorical(y, num_classes=self._n_classes))
    
    def __getitem__(self, index):
    # Generate data
        list_IDs_temp = self._type_gestion(index)
        X, y = self.__data_generation(list_IDs_temp)
        return (X, y)
    
    def display(self, name = "display.jpeg", orientation = "rect", abs_pos = 185):
        raise ValueError("Not adapted to this generator")
    
