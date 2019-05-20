import numpy as np
import nibabel as nb
import os
import pandas as pd
from tqdm import tqdm


def corrn(A, B):
    a_mean = np.mean(A)
    b_mean = np.mean(B)
    return (np.sum((A - a_mean)*(B - b_mean))/(np.sqrt(np.sum((A - a_mean)**2)\
                  *np.sum((B - b_mean)**2))))

def generate_df(dir_path, create_csv = False):
    """
    Takes as input a directory with files organised as follow: directory <- subdirectory <- files,
    and returns a pandas.Dataframe compsed of the following columns:
    path from pwd to the img_file/ if the img file is noised or not/ RMS / MSE (rounded to an int) between the image and the clean image of the subdirectory/ 
    Disp, swalF, swalM, sudF, sudM, MSE not rounded. 
    Noised images names must have the following format: xxx_RMS_2_Disp_5_swalF_0_swalM_5_sudF_0_sudM_5.nii
    Clean images must be in a format different than nii (like nii.gz) 
    If the correspnding option is true, a csv file is created from this dataframe
    """
    suj_listdir = []
    y_RMS, y_Disp, y_swalF, y_swalM, y_sudF, y_sudM = [], [], [], [], [], []
    y_MSE, noised, y_corr, y_MSE_All = [], [], [], []
    for direc in tqdm(os.listdir(dir_path)):
        temp = os.listdir(dir_path+direc)
        temp_to_add = []
        for filename in temp:
            
            temp_to_add.append(dir_path+direc+"/"+filename)
            if filename[-2:] == "gz":
                noised.append(1)
                clean_im = nb.load(dir_path+direc+"/"+filename).get_fdata()
            else:
                noised.append(0)
        suj_listdir += temp_to_add
        for st in temp_to_add:
            temp = [pos for pos, char in enumerate(st) if char == "_"]
            if st[-3:] == "nii":
                y_RMS.append(int(st[temp[-11]+1:temp[-10]]))
                y_Disp.append(int(st[temp[-9]+1:temp[-8]]))
                y_swalF.append(int(st[temp[-7]+1:temp[-6]]))
                y_swalM.append(int(st[temp[-5]+1:temp[-4]]))
                y_sudF.append(int(st[temp[-3]+1:temp[-2]]))
                y_sudM.append(int(st[temp[-1]+1:len(st)-4]))
            else:
                y_RMS.append(0)
                y_Disp.append(0)
                y_swalF.append(0)
                y_swalM.append(0)
                y_sudF.append(0)
                y_sudM.append(0)
            image = nb.load(st).get_fdata()
            y_MSE.append((np.sqrt(np.square(image-clean_im).mean())))
            y_corr.append((corrn(image, clean_im)))
    
    

    df = pd.DataFrame(np.column_stack([suj_listdir, noised, y_RMS, np.round(y_MSE, 0), y_corr, y_Disp, y_swalF, y_swalM, y_sudF, y_sudM, y_MSE]),
                                                  columns=['img_file' ,'noise', 'RMS', 'MSE', 'Corr', 'Disp', 'swalF', 'swalM', 'sudF', 'sudM', "MSE_All"])
    for col in df.columns:
        if col == "Corr":
          df[col] = df[col].apply(float)
        elif col != "img_file":
          df[col] = df[col].apply(int)
    
    if create_csv:
        df.to_csv(dir_path + "_csv", sep = ",")
    return df





