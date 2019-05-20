import operator
import numpy as np
import keras_preprocessing.image.affine_transformations as at
import scipy.ndimage as ndi

operator_dict = {
    '==': operator.eq,
    '>': operator.gt,
    '<': operator.lt,
    "|": operator.or_,
    "&": operator.and_
}

def apply_conditions_on_dataset(dataset, conditions):
    """
    Conditions of the form ((intermediate_op,) var, op, values).
    Takes one or several conditions (each condition must be in a 3-tuple or a 4-tuple, each block of conditions in a list)
    and a dataframe, and returns the part of the dataframe respecting the conditions.
    """
    for p_c in conditions:
      if type(p_c) == list:
        if len(p_c[0]) == 3:
          snap = apply_conditions_on_dataset(dataset, p_c)
        else:
          temp_var = p_c[0][0]
          p_c[0] = p_c[0][1:]
          temp_snap = apply_conditions_on_dataset(dataset, p_c)
          snap = operator_dict[temp_var](snap.copy(), temp_snap)              
      elif len(p_c) == 3:
        snap = operator_dict[p_c[1]](dataset.copy()[p_c[0]],p_c[2])
      elif len(p_c) == 4:
        snap = operator_dict[p_c[0]]((snap.copy()), (operator_dict[p_c[2]](dataset.copy()[p_c[1]],p_c[3])))
    return snap    

def quadriview(nifti_image, slice_sag, slice_orth, 
                   slice_cor_1, slice_cor_2):
    """
    Takes as input a 3D image and the 4 positions of slices, and returns the fabricated image (sag - cor/ ax1 - ax2) and return 
    """
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

def take_slice(img_3D, view):
   """
   Takes as input a 3D image and the wanted view, and returns a slice with the wanted view taken in a random point.
   """
   img_shape = img_3D.shape
   if view == "sag":
       slice_pos = np.random.randint(int(0.2*img_shape[0]), int(0.8*img_shape[0]))
       img_2D = img_3D[slice_pos, :, :]
   elif view == "cor":
       slice_pos = np.random.randint(int(0.2*img_shape[1]), int(0.8*img_shape[1]))
       img_2D = img_3D[:, slice_pos, :]
   else:
       slice_pos = np.random.randint(int(0.2*img_shape[2]), int(0.8*img_shape[2]))
       img_2D = img_3D[:, :, slice_pos]
   img_2D = np.expand_dims(img_2D, 2)        
   return img_2D

def transfo_imgs(image, args, mode): # Types of transformations and range inspired by Sujit 2019
    """
    Takes as input an image and transforms it (currently rotation and translation available)
    """  
    img = image
    if len(img.shape) == 3:         
        if np.random.rand(1)[0] < args[0]:
            angle = 10
            img = at.random_rotation(img, angle, row_axis=0, col_axis=1, channel_axis=2)
        if np.random.rand(1)[0] < args[1]:
            axs_0 = 21
            axs_1 = 6
            img = at.random_shift(img, axs_0, axs_1, row_axis=0, col_axis=1, channel_axis=2) 
    elif len(img.shape) == 4:
        img2 = np.zeros( (*img.shape[:-1], 0))
        if np.random.rand(1)[0] < args[0]:
            angle = 10
            axes = tuple(np.random.choice(range(3), 2))
            for k in range(img.shape[3]):
                img2 = np.concatenate((img2, ndi.rotate(img[k], angle, axes=axes, reshape=False)), axis = 3)
        img = img2
        if np.random.rand(1)[0] < args[1]:
            axs_0 = np.random.randint(0, 21)
            axs_1 = np.random.randint(-5, 6)
            axs_2 = np.random.randint(-5, 5)
            img = shift(img, [axs_0, axs_1, axs_2])
    return img
        
def normalization_func(img):
    vmin, vmax = img.min(), img.max()
    im = (img - vmin)/(vmax - vmin)
    return im


