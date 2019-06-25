#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 15:59:12 2019

@author: dimitri.hamzaoui
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 17:08:55 2019

@author: dimitri.hamzaoui
"""
import numpy as np
from sklearn.feature_extraction.image import extract_patches

#%% 
#def make_patches_step(img, patch_shape, extraction_step):
#    patches = extract_patches(img, patch_shape, extraction_step=extraction_step)
#    patches = patches.reshape((-1, )+patch_shape)
#    return patches
#
#def make_patches_overlay(img, patch_shape, overlay):
#    extraction_step = tuple(np.array(np.round([patch_shape[k]*(1-overlay) for k in range(len(patch_shape))])).astype(int))
#    patches = extract_patches(img, patch_shape, extraction_step=extraction_step)
#    patches = patches.reshape((-1, )+patch_shape)
#    return patches

def make_organized_patches(img, patch_shape, step_info):
    """
    About the dimension:
        If 2D: The first dimension is the weight, the secoind one is the height
        If 3D: The first dimension is the width, the second one the height
        and the third one the length
    About step_info:
        * If a float between 0 and 1, takes this as the overlay between two
        successives patches
        * If an int, choose a step with this int as step in every direction
        * If a vector, is the step chosen
        
    """
    
    arr_ndim = img.ndim
    if isinstance(patch_shape, int):
        patch_shape = tuple([patch_shape] *arr_ndim)
    
    if isinstance(step_info, float) and int(step_info) == 0:
        extraction_step =  tuple(np.array(np.round([patch_shape[k]*(1-step_info) for k in range(arr_ndim)])).astype(int))
    elif isinstance(step_info, int):
        extraction_step = tuple([step_info] * arr_ndim)
    else :
        extraction_step = step_info
    for k in range(arr_ndim):
        if (img.shape[k]-patch_shape[k])%extraction_step[k] !=0:
            print("Attention: Dimension {} -Pas non adapté à la partition voulue:".format(k)
                  +"certaines parties ne sont pas prises!")
            new_size = list(img.shape)
    patches = extract_patches(img, patch_shape, extraction_step=extraction_step)
    patches = patches.reshape((-1, )+patch_shape)
    return patches

def make_random_3D_patches(img, patch_shape, patch_number):
    """
    About the dimension:
        If 2D: The first dimension is the weight, the secoind one is the height
        If 3D: The first dimension is the width, the second one the height
        and the third one the length
    About step_info:
        * If a float between 0 and 1, takes this as the overlay between two
        successives patches
        * If an int, choose a step with this int as step in every direction
        * If a vector, is the step chosen
        
    """
    
    patches = []
    shape_im = img.shape
    arr_ndim = img.ndim
    if isinstance(patch_shape, int):
        patch_shape = tuple([patch_shape] *arr_ndim)
    for i in range(patch_number):
        border_0 = np.random.randint(0, shape_im[0] - patch_shape[0])
        border_1 = np.random.randint(0, shape_im[1] - patch_shape[1])
        border_2 = np.random.randint(0, shape_im[2] - patch_shape[2])
        patches.append(img[border_0:border_0 + patch_shape[0], border_1:border_1 + patch_shape[1], border_2:border_2 + patch_shape[2]])
    return patches



def reconstruct_images_2d(patches, size_image, step_info):
    """
    About the dimension:
        The first dimension is the the height, the secoind one is the length
    
    About step_info:
        * If a float between 0 and 1, takes this as the overlay between two
        successives patches
        * If an int, choose a step with this int as step in every direction
        * If a vector, is the step chosen
    
    """
    patch_shape = patches[0].shape
    if isinstance(size_image, int):
        size_image = tuple([size_image] *(len(patch_shape)))
    img = np.zeros(size_image)
    if isinstance(step_info, float) and int(step_info) == 0:
        step =  tuple(np.array(np.round([patch_shape[k]*(1-step_info) for k in range(len(size_image))]).astype(int)))
    elif isinstance(step_info, int):
        step = tuple([step_info] * len(patch_shape))
    else:
        step = step_info
    absc = 0
    ordo = 0
    for i in range(len(patches)):
        local_patch = patches[i]
        #print(local_patch)
        img[ordo: ordo + patch_shape[1], absc: absc + patch_shape[0]] = local_patch
        absc += step[0]
        if absc > size_image[0]- patch_shape[0]:
            absc = 0
            ordo += step[1]
    return img

def reconstruct_images_3d(patches, size_image, step_info):
    
    """
    About the dimension:
        If 2D: The first dimension is the weight, the secoind one is the height
        If 3D: The first dimension is the width, the second one the height
        and the third one the length
    About step_info:
        * If a float between 0 and 1, takes this as the overlay between two
        successives patches
        * If an int, choose a step with this int as step in every direction
        * If a vector, is the step chosen
        
    """
    patch_shape = patches[0].shape
    if isinstance(size_image, int):
        size_image = tuple([size_image] *(len(patch_shape)))
    img = np.zeros(size_image)
    if isinstance(step_info, float) and int(step_info) == 0:
        step =  tuple(np.array(np.round([patch_shape[k]*(1-step_info) for k in range(len(patch_shape))]).astype(int)))
    elif isinstance(step_info, int):
        step = tuple([step_info] * len(patch_shape))
    else:
        step = step_info
    x_abs = 0
    y_abs = 0
    z_abs = 0
    for i in range(len(patches)):
        local_patch = patches[i]
        #print(local_patch)
        img[y_abs: y_abs + local_patch.shape[0], z_abs: z_abs + local_patch.shape[1],
           x_abs: x_abs + local_patch.shape[2]] = local_patch
        x_abs += step[2]
        if x_abs > size_image[2]- patch_shape[2]:
            x_abs = 0
            z_abs += step[1]
        if z_abs > size_image[1]- patch_shape[1]:
            z_abs = 0
            y_abs += step[0]
    return img
