import generators
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
import numpy as np
import os
import pandas as pd
from utils import quadriview, normalization_func, normalization_mask, apply_conditions_on_dataset, normalization_brain
import nibabel as nb
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
from generate_df import generate_df
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import roc_curve, auc

from vis.utils import utils
from keras import activations
import matplotlib.cm as cm
from vis.visualization import visualize_saliency, overlay
from vis.visualization import visualize_cam




def summary_model_training(model, prefix):
    history = model.history.history
    fig = plt.figure(figsize = (15, 15))
    plt.subplot(211)
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.subplot(212)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    model_name = model.name
    if not os.path.isdir(prefix + "Checks/Check_"+model_name):
        os.mkdir(prefix + "Checks/Check_"+model_name)
    file_path = prefix + "Checks/Check_"+model_name+"/"
    plt.savefig(file_path+model_name+"_learning_curves.png")
    print("Learning curves generated!")

def model_eval(model, dataset_test, prefix="", steps=60, **kwargs):
    model_name = model.name
    if not os.path.isdir(prefix+"Checks/Check_"+model_name):
        os.mkdir(prefix+"Checks/Check_"+model_name)
    file_path = prefix+"Checks/Check_"+model_name+"/"
    if dataset_test[-3:] == "csv":
        gen1 = generators.Quadriview_DataGenerator(csv_file = dataset_test, generator_type = "balanced", prefix=prefix, **kwargs)
        gen2 = generators.Quadriview_DataGenerator(csv_file = dataset_test, generator_type = "always_noised", prefix=prefix, **kwargs)
        gen3 = generators.Quadriview_DataGenerator(csv_file = dataset_test, generator_type = "always_clean", prefix=prefix, **kwargs)
    else:
        gen1 = generators.Quadriview_DataGenerator(csv_file = dataset_test, generator_type = "balanced", prefix=prefix, **kwargs)
        gen2 = generators.Quadriview_DataGenerator(csv_file = dataset_test, generator_type = "always_noised", prefix=prefix, **kwargs)
        gen3 = generators.Quadriview_DataGenerator(csv_file = dataset_test, generator_type = "always_clean", prefix=prefix, **kwargs)
    gen1.metadata;
    gen2.metadata;
    gen3.metadata;
    acc_global = model.evaluate_generator(gen1, steps = steps, verbose = 1, use_multiprocessing = True, workers = 20)
    print("Global accuracy: Done!")
    acc_noised = model.evaluate_generator(gen2, steps = steps, verbose = 1, use_multiprocessing = True, workers = 20)
    print("Accuracy on noised images: Done!")
    acc_clean =  model.evaluate_generator(gen3, steps = steps, verbose = 1, use_multiprocessing = True, workers = 20)
    print("Accuracy on clean images: Done!")
    with open(file_path+model_name+"_summary.txt","w+") as f:
        f.write("Global accuracy of "+model_name+" : {}  \n".format(np.round(acc_global[1], 3)))
        f.write("Accuracy on noised images of "+model_name+" : {}  \n".format(np.round(acc_noised[1], 3)))
        f.write("Accuracy on clean images of "+model_name+" : {}  \n".format(np.round(acc_clean[1], 3)))
        f.close()
    print("Model evaluation finished")

def model_eval_test(model, size,  prefix="", **kwargs):
    model_name = model.name
    if not os.path.isdir(prefix+"Checks/Check_"+model_name):
        os.mkdir(prefix+"Checks/Check_"+model_name)
    file_path = prefix+"Checks/Check_"+model_name+"/"
    gen1 = generators.Quadriview_DataGenerator(generator_type = "balanced", prefix=prefix, batch_size = size, **kwargs)
    gen1.metadata;
    X, y = gen1.get_data()
    y_pred = model.predict(X, verbose = 1)
    y = np.argmax(y, axis = 1)
    y_pred = np.argmax(y_pred, axis = 1)
    acc_global = (np.sum(y == y_pred)/len(y))
    acc_noised = np.sum((y == 0)*(y_pred == 0))/np.sum((y==0))
    acc_clean = np.sum((y == 1)*(y_pred == 1))/np.sum((y==1)) 
    with open(file_path+model_name+"_summary.txt","w+") as f:
        f.write("Global accuracy of "+model_name+" : {}  \n".format(np.round(acc_global, 3)))
        f.write("Accuracy on noised images of "+model_name+" : {}  \n".format(np.round(acc_noised, 3)))
        f.write("Accuracy on clean images of "+model_name+" : {}  \n".format(np.round(acc_clean, 3)))
        f.close()
    print("Model evaluation finished")

def model_display(model,  name = "output.jpg", orientation = "rect", font_path = "/usr/share/fonts/truetype/freefont/FreeMono.ttf", prefix = "", **kwargs):
    model_name = model.name
    if not os.path.isdir(prefix + "Checks/Check_"+model_name):
        os.mkdir(prefix + "Checks/Check_"+model_name)
    file_path = prefix + "Checks/Check_"+model_name+"/"
    pc_test = generators.Quadriview_DataGenerator(prefix = prefix, **kwargs)
    pc_test.metadata;
    X, y = pc_test.display(name = name, orientation = orientation, abs_pos = 155)
    y = np.argmax(y, axis = 1)
    y_pred = model.predict(X)
    img = Image.open(name)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, 8, encoding="unic")
    if orientation == "vertical":
        for k in range(len(y_pred)):
            cl = np.argmax(y_pred[k])
            pba =  np.round(y_pred[k, cl], 3)
            draw.text((205, 5+400*k),"Pred: {} Prob:".format(cl) + str(pba), (255, 255, 255, 255), font=font)
    elif orientation == "horizontal":
        for k in range(len(y_pred)):
            cl = np.argmax(y_pred[k])
            pba =  np.round(y_pred[k, cl], 3)
            draw.text((205+400*k, 5),"Pred: {} Prob:".format(cl) + str(pba), (255, 255, 255, 255), font=font)
    else:
        col = int(np.sqrt(pc_test.batch_size))
        for k in range(len(y_pred)):     
            cl = np.argmax(y_pred[k])
            pba =  np.round(y_pred[k, cl], 3)   
            draw.text((205+400*(k%col), 5 + 400*(k//col)),"Pred: {} Prob:".format(cl) + str(pba), (255, 255, 255, 255), font=font)
        
    img.save(file_path + model_name+"_"+name)
    os.remove(name)
    acc = np.mean(np.argmax(y_pred, axis = 1) == y)
    t_p = np.sum(np.argmax(y_pred, axis = 1) * y)/np.sum(y)
    t_n = np.sum((1-np.argmax(y_pred, axis = 1)) * (1-y))/np.sum(1-y)
    print("Accuracy: {}, True_positive_rate: {},  True_negative_rate: {}".format(np.round(acc, 3), np.round(t_p, 3), np.round(t_n, 3)))
    print("Creation of images finished")
    return acc, t_p, t_n

def test_noise_characs(model, dataset_test, measure = ["RMS", "MSE", "Corr"], prefix = "", normalization = None, conditions = None):
    if dataset_test[-3:] == "csv":
        tab = pd.read_csv(prefix + dataset_test, sep = ",", index_col = 0)
    else:
        tab = generate_df(prefix + dataset_test)
    
    if conditions is not None:
            tab = tab[apply_conditions_on_dataset(tab, conditions)]
    length = len(tab)
    XX = []
    model_name = model.name
    if not os.path.isdir(prefix + "Checks/Check_"+model_name):
        os.mkdir(prefix + "Checks/Check_"+model_name)
    file_path = prefix +"Checks/Check_"+model_name+"/"
    if type(measure) == str:
        measure = [measure]
    for k in tqdm(range(length)):
        image = nb.load(prefix + tab.iloc[k].img_file).get_fdata()
        if normalization == "global":
            image = normalization_func(image)
        elif normalization == "local":
            temp1 = [pos for pos, char in enumerate(tab.iloc[k].img_file) if char == "/"][0]
            temp2 = [pos for pos, char in enumerate(tab.iloc[k].img_file) if tab.iloc[k].img_file[pos:pos+4]==".nii"][-1]
            mask = nb.load(prefix + "masks"+tab.iloc[k].img_file[temp1:temp2]+"_mask.nii.gz").get_fdata()
            #temp = [pos for pos, char in enumerate(tab.iloc[k].img_file) if char == "_"]
            image = normalization_mask(image, mask)
        elif normalization == "brain":
            temp1 = [pos for pos, char in enumerate(tab.iloc[k].img_file) if char == "/"][0]
            temp2 = [pos for pos, char in enumerate(tab.iloc[k].img_file) if tab.iloc[k].img_file[pos:pos+4]==".nii"][-1]
            mask = nb.load(prefix + "masks"+tab.iloc[k].img_file[temp1:temp2]+"_mask.nii.gz").get_fdata()
            #temp = [pos for pos, char in enumerate(tab.iloc[k].img_file) if char == "_"]
            image = normalization_brain(image, mask)
        slice_sag = int(0.5*image.shape[0])
        slice_orth = int(0.5*image.shape[1])
        slice_cor_1 = int(0.4*image.shape[2])
        slice_cor_2 = int(0.6*image.shape[2])
        image_2 = quadriview(image, slice_sag,  slice_orth,
                                        slice_cor_1, slice_cor_2)
        image_2 = np.expand_dims(image_2, 2)        
        XX.append(image_2)

    XX = np.array(XX)
    prob_X = model.predict(XX, verbose = 1)
    
    for meas in measure:
        y = tab[meas].values
        meas_zero = y[tab.noise == 0]
        meas_one = y[tab.noise == 1]
        
        prob_zero = prob_X[:,1][tab.noise == 0]
        prob_one = prob_X[:,1][tab.noise == 1]
     
        fig = plt.figure(figsize = (15, 15))
        plt.subplot(311)
        
        plt.semilogy(meas_zero, prob_zero,  "r.", label = "Noised images")
        plt.semilogy(meas_one, prob_one,  "b.", label = "Clean images")
        plt.semilogy(y, [0.5 for k in y],  "g", label = "Threshold")
        plt.xlabel(meas)
        plt.ylabel("Proba of classification 'without noise' ")
        plt.title("Impact of " + meas + " on classification - Raw")
        plt.legend()
        

        measure_set = list(set(meas_zero))
        mean_prob = []
        std_prob = []
        
        for measured in measure_set:
            temp = [prob_zero[k] for k in range(len(prob_zero)) if meas_zero[k] == measured ]
            mean_prob.append(np.mean(temp))
            std_prob.append(np.std(temp))
        measure_set = list(set(meas_one)) + measure_set
        mean_prob = [np.mean(prob_one)] + mean_prob
        std_prob = [np.std(prob_one)] + std_prob

        measure_set, mean_prob, std_prob = (list(t) for t in zip(*sorted(zip(measure_set, mean_prob, std_prob))))
        plt.subplot(312)
        plt.plot(measure_set, mean_prob, "r")
        plt.plot(measure_set, [0.5 for k in measure_set],  "g", label = "Threshold")
        plt.yscale('log')
        plt.xlabel(meas)
        plt.ylabel("Mean of the proba of classification 'without noise'")
        plt.title("Impact of " + meas+" on Misclassification Pba - without std")

        plt.subplot(313)
        plt.errorbar(measure_set, mean_prob, yerr = std_prob, color = "red", ecolor = "blue")
        plt.plot(measure_set, [0.5 for k in measure_set],  "g", label = "Threshold")
        plt.xlabel(meas)
        plt.ylabel("Mean of the proba of classification 'without noise'")
        plt.title("Impact of " + meas+" on Misclassification Pba - with std")

        plt.savefig(file_path+model_name+"_"+meas+".png")
    print("Noise evaluation finished")

def test_slice(model, dataset_test, num_choices = 40, normalization = None, prefix = "", conditions=None, measure = "RMS", threshold = (10, 20)):
    if dataset_test[-3:] == "csv":
        csv = pd.read_csv(prefix + dataset_test, sep = ",", index_col = 0)
    else:
        csv = generate_df(prefix + dataset_test)
    if conditions is not None:
        csv = csv[apply_conditions_on_dataset(csv, conditions)]
    ind_not_noised = np.random.choice(np.where(csv.noise.values == 1)[0], num_choices)
    ind_small = np.random.choice(np.where((csv.noise.values == 0) & (csv[measure].values < threshold[0]))[0], num_choices)
    ind_med = np.random.choice(np.where((csv.noise.values == 0) & (csv[measure].values > threshold[0]) & (csv.RMS.values < threshold[1]))[0], num_choices)
    ind_big = np.random.choice(np.where((csv.noise.values == 0) & (csv[measure].values > threshold[1]))[0], num_choices)
    
    if normalization == "global":
        not_noised_images = np.array([normalization_func(nb.load(prefix +csv.iloc[k].img_file).get_fdata()) for k in tqdm(ind_not_noised)])
        small_images = np.array([normalization_func(nb.load(prefix +csv.iloc[k].img_file).get_fdata()) for k in tqdm(ind_small)])
        med_images = np.array([normalization_func(nb.load(prefix +csv.iloc[k].img_file).get_fdata()) for k in tqdm(ind_med)])
        big_images = np.array([normalization_func(nb.load(prefix +csv.iloc[k].img_file).get_fdata()) for k in tqdm(ind_big)])

    elif normalization == "local":
        temp1 = [[pos for pos, char in enumerate(csv.iloc[k].img_file) if char == "/"][0] for k in (ind_not_noised)]
        temp2 = [[pos for pos, char in enumerate(csv.iloc[k].img_file) if csv.iloc[k].img_file[pos:pos+4]==".nii"][-1] for k in (ind_not_noised)]
        not_noised_images =  np.array([normalization_mask(nb.load(prefix +csv.iloc[k].img_file).get_fdata(), nb.load(prefix + "masks"+csv.iloc[k].img_file[temp1[j]:temp2[j]]+"_mask.nii.gz").get_fdata())\
 for j, k in tqdm(enumerate(ind_not_noised))]) 
        temp1 = [[pos for pos, char in enumerate(csv.iloc[k].img_file) if char == "/"][0] for k in (ind_small)]
        temp2 = [[pos for pos, char in enumerate(csv.iloc[k].img_file) if csv.iloc[k].img_file[pos:pos+4]==".nii"][-1] for k in (ind_small)]
        small_images =  np.array([normalization_mask(nb.load(prefix +csv.iloc[k].img_file).get_fdata(), nb.load(prefix + "masks"+csv.iloc[k].img_file[temp1[j]:temp2[j]]+"_mask.nii.gz").get_fdata())\
 for j, k in tqdm(enumerate(ind_small))]) 
        temp1 = [[pos for pos, char in enumerate(csv.iloc[k].img_file) if char == "/"][0] for k in (ind_med)]
        temp2 = [[pos for pos, char in enumerate(csv.iloc[k].img_file) if csv.iloc[k].img_file[pos:pos+4]==".nii"][-1] for k in (ind_med)]
        med_images =  np.array([normalization_mask(nb.load(prefix +csv.iloc[k].img_file).get_fdata(), nb.load(prefix + "masks"+csv.iloc[k].img_file[temp1[j]:temp2[j]]+"_mask.nii.gz").get_fdata())\
 for j, k in tqdm(enumerate(ind_med))]) 
        temp1 = [[pos for pos, char in enumerate(csv.iloc[k].img_file) if char == "/"][0] for k in (ind_big)]
        temp2 = [[pos for pos, char in enumerate(csv.iloc[k].img_file) if csv.iloc[k].img_file[pos:pos+4]==".nii"][-1] for k in (ind_big)]
        big_images =  np.array([normalization_mask(nb.load(prefix +csv.iloc[k].img_file).get_fdata(), nb.load(prefix + "masks"+csv.iloc[k].img_file[temp1[j]:temp2[j]]+"_mask.nii.gz").get_fdata())\
 for j, k in tqdm(enumerate(ind_big))]) 
     
    elif normalization == "brain":
        temp1 = [[pos for pos, char in enumerate(csv.iloc[k].img_file) if char == "/"][0] for k in (ind_not_noised)]
        temp2 = [[pos for pos, char in enumerate(csv.iloc[k].img_file) if csv.iloc[k].img_file[pos:pos+4]==".nii"][-1] for k in (ind_not_noised)]
        not_noised_images =  np.array([normalization_brain(nb.load(prefix +csv.iloc[k].img_file).get_fdata(), nb.load(prefix + "masks"+csv.iloc[k].img_file[temp1[j]:temp2[j]]+"_mask.nii.gz").get_fdata())\
 for j, k in tqdm(enumerate(ind_not_noised))]) 
        temp1 = [[pos for pos, char in enumerate(csv.iloc[k].img_file) if char == "/"][0] for k in (ind_small)]
        temp2 = [[pos for pos, char in enumerate(csv.iloc[k].img_file) if csv.iloc[k].img_file[pos:pos+4]==".nii"][-1] for k in (ind_small)]
        small_images =  np.array([normalization_brain(nb.load(prefix +csv.iloc[k].img_file).get_fdata(), nb.load(prefix + "masks"+csv.iloc[k].img_file[temp1[j]:temp2[j]]+"_mask.nii.gz").get_fdata())\
 for j, k in tqdm(enumerate(ind_small))]) 
        temp1 = [[pos for pos, char in enumerate(csv.iloc[k].img_file) if char == "/"][0] for k in (ind_med)]
        temp2 = [[pos for pos, char in enumerate(csv.iloc[k].img_file) if csv.iloc[k].img_file[pos:pos+4]==".nii"][-1] for k in (ind_med)]
        med_images =  np.array([normalization_brain(nb.load(prefix +csv.iloc[k].img_file).get_fdata(), nb.load(prefix + "masks"+csv.iloc[k].img_file[temp1[j]:temp2[j]]+"_mask.nii.gz").get_fdata())\
 for j, k in tqdm(enumerate(ind_med))]) 
        temp1 = [[pos for pos, char in enumerate(csv.iloc[k].img_file) if char == "/"][0] for k in (ind_big)]
        temp2 = [[pos for pos, char in enumerate(csv.iloc[k].img_file) if csv.iloc[k].img_file[pos:pos+4]==".nii"][-1] for k in (ind_big)]
        big_images =  np.array([normalization_brain(nb.load(prefix +csv.iloc[k].img_file).get_fdata(), nb.load(prefix + "masks"+csv.iloc[k].img_file[temp1[j]:temp2[j]]+"_mask.nii.gz").get_fdata())\
 for j, k in tqdm(enumerate(ind_big))]) 

    else:
        not_noised_images = np.array([nb.load(prefix + csv.iloc[k].img_file).get_fdata() for k in tqdm(ind_not_noised)])
        small_images = np.array([nb.load(prefix +csv.iloc[k].img_file).get_fdata() for k in tqdm(ind_small)])
        med_images = np.array([nb.load(prefix +csv.iloc[k].img_file).get_fdata() for k in tqdm(ind_med)])
        big_images = np.array([nb.load(prefix +csv.iloc[k].img_file).get_fdata() for k in tqdm(ind_big)])
    
    shape_im = not_noised_images[0].shape
    sag_slice = int(0.5*shape_im[0])    
    cor_slice = int(0.5*shape_im[1])
    ax1_slice = int(0.4*shape_im[2])
    ax2_slice = int(0.6*shape_im[2])
    
    not_noised_predict = np.zeros((0, 2))
    small_predict = np.zeros((0, 2))
    med_predict = np.zeros((0, 2))
    big_predict = np.zeros((0, 2))
    for k in tqdm(range(num_choices)):
        XX_not_noised = []
        XX_small = []
        XX_med = []
        XX_big = []
        for sliice in (range(int(0.2*shape_im[0]), int(0.8*shape_im[0]))):
            XX_not_noised.append(np.expand_dims(quadriview(not_noised_images[k], sliice, cor_slice, ax1_slice, ax2_slice) ,2))
            XX_small.append(np.expand_dims(quadriview(small_images[k], sliice, cor_slice, ax1_slice, ax2_slice) ,2))
            XX_med.append(np.expand_dims(quadriview(med_images[k], sliice, cor_slice, ax1_slice, ax2_slice) ,2))
            XX_big.append(np.expand_dims(quadriview(big_images[k], sliice, cor_slice, ax1_slice, ax2_slice) ,2))
        XX_not_noised = np.array(XX_not_noised)
        XX_small = np.array(XX_small)
        XX_med = np.array(XX_med)
        XX_big = np.array(XX_big)
        not_noised_predict = np.concatenate((not_noised_predict, model.predict(XX_not_noised)), axis = 0)
        small_predict = np.concatenate((small_predict, model.predict(XX_small)), axis = 0)
        med_predict = np.concatenate((med_predict, model.predict(XX_med)), axis = 0)
        big_predict = np.concatenate((big_predict, model.predict(XX_big)), axis = 0)
    
    fig = plt.figure(figsize = (15, 15))
    gs = gridspec.GridSpec(nrows=3, ncols=2, figure=fig)

    ax0 = fig.add_subplot(gs[0, :])
    plt.plot(np.array(range(int(0.2*shape_im[0]), int(0.8*shape_im[0])))/shape_im[0], np.mean(not_noised_predict[:,0].reshape((num_choices, -1)), axis = 0), label = "Not noised image")
    plt.plot(np.array(range(int(0.2*shape_im[0]), int(0.8*shape_im[0])))/shape_im[0], np.mean(small_predict[:,1].reshape((num_choices, -1)), axis = 0), label = "Weakly Noised image")
    plt.plot(np.array(range(int(0.2*shape_im[0]), int(0.8*shape_im[0])))/shape_im[0], np.mean(med_predict[:,1].reshape((num_choices, -1)), axis = 0), label = "Mildly noised image")
    plt.plot(np.array(range(int(0.2*shape_im[0]), int(0.8*shape_im[0])))/shape_im[0], np.mean(big_predict[:,1].reshape((num_choices, -1)), axis = 0), label = "Seriously noised image")
    plt.xlabel("Slice position")
    plt.title("Sagittal slice")
    plt.ylabel("Misclassification proba")
    plt.legend()
    

    not_noised_predict = np.zeros((0, 2))
    small_predict = np.zeros((0, 2))
    med_predict = np.zeros((0, 2))
    big_predict = np.zeros((0, 2))
    for k in tqdm(range(num_choices)):
        XX_not_noised = []
        XX_small = []
        XX_med = []
        XX_big = []
        for sliice in (range(int(0.2*shape_im[1]), int(0.8*shape_im[1]))):
            XX_not_noised.append(np.expand_dims(quadriview(not_noised_images[k], sag_slice, sliice, ax1_slice, ax2_slice) ,2))
            XX_small.append(np.expand_dims(quadriview(small_images[k], sag_slice, sliice, ax1_slice, ax2_slice) ,2))
            XX_med.append(np.expand_dims(quadriview(med_images[k], sag_slice, sliice, ax1_slice, ax2_slice) ,2))
            XX_big.append(np.expand_dims(quadriview(big_images[k], sag_slice, sliice, ax1_slice, ax2_slice) ,2))
        XX_not_noised = np.array(XX_not_noised)
        XX_small = np.array(XX_small)
        XX_med = np.array(XX_med)
        XX_big = np.array(XX_big)
        not_noised_predict = np.concatenate((not_noised_predict, model.predict(XX_not_noised)), axis = 0)
        small_predict = np.concatenate((small_predict, model.predict(XX_small)), axis = 0)
        med_predict = np.concatenate((med_predict, model.predict(XX_med)), axis = 0)
        big_predict = np.concatenate((big_predict, model.predict(XX_big)), axis = 0)
    
    ax1 = fig.add_subplot(gs[1, :])
    plt.plot(np.array(range(int(0.2*shape_im[1]), int(0.8*shape_im[1])))/shape_im[1], np.mean(not_noised_predict[:,0].reshape((num_choices, -1)), axis = 0), label = "Not noised image")
    plt.plot(np.array(range(int(0.2*shape_im[1]), int(0.8*shape_im[1])))/shape_im[1], np.mean(small_predict[:,1].reshape((num_choices, -1)), axis = 0), label = "Weakly Noised image")
    plt.plot(np.array(range(int(0.2*shape_im[1]), int(0.8*shape_im[1])))/shape_im[1], np.mean(med_predict[:,1].reshape((num_choices, -1)), axis = 0), label = "Mildly noised image")
    plt.plot(np.array(range(int(0.2*shape_im[1]), int(0.8*shape_im[1])))/shape_im[1], np.mean(big_predict[:,1].reshape((num_choices, -1)), axis = 0), label = "Seriously noised image")
    plt.xlabel("Slice position")
    plt.ylabel("Misclassification proba")
    plt.title("Coronal slice")
    plt.legend()

    

    not_noised_predict = np.zeros((0, 2))
    small_predict = np.zeros((0, 2))
    med_predict = np.zeros((0, 2))
    big_predict = np.zeros((0, 2))

    for k in tqdm(range(num_choices)):
        XX_not_noised = []
        XX_small = []
        XX_med = []
        XX_big = []
        for sliice in (range(int(0.2*shape_im[2]), int(0.5*shape_im[2]))):
            XX_not_noised.append(np.expand_dims(quadriview(not_noised_images[k], sag_slice, cor_slice, sliice, ax2_slice) ,2))
            XX_small.append(np.expand_dims(quadriview(small_images[k], sag_slice, cor_slice, sliice, ax2_slice) ,2))
            XX_med.append(np.expand_dims(quadriview(med_images[k], sag_slice, cor_slice, sliice, ax2_slice) ,2))
            XX_big.append(np.expand_dims(quadriview(big_images[k], sag_slice, cor_slice, sliice, ax2_slice) ,2))
        XX_not_noised = np.array(XX_not_noised)
        XX_small = np.array(XX_small)
        XX_med = np.array(XX_med)
        XX_big = np.array(XX_big)
        not_noised_predict = np.concatenate((not_noised_predict, model.predict(XX_not_noised)), axis = 0)
        small_predict = np.concatenate((small_predict, model.predict(XX_small)), axis = 0)
        med_predict = np.concatenate((med_predict, model.predict(XX_med)), axis = 0)
        big_predict = np.concatenate((big_predict, model.predict(XX_big)), axis = 0)
    
    ax2 = fig.add_subplot(gs[2, 0])
    plt.plot(np.array(range(int(0.2*shape_im[2]), int(0.5*shape_im[2])))/shape_im[2], np.mean(not_noised_predict[:,0].reshape((num_choices, -1)), axis = 0), label = "Not noised image")
    plt.plot(np.array(range(int(0.2*shape_im[2]), int(0.5*shape_im[2])))/shape_im[2], np.mean(small_predict[:,1].reshape((num_choices, -1)), axis = 0), label = "Small RMS image")
    plt.plot(np.array(range(int(0.2*shape_im[2]), int(0.5*shape_im[2])))/shape_im[2], np.mean(med_predict[:,1].reshape((num_choices, -1)), axis = 0), label = "Middle-range RMS image")
    plt.plot(np.array(range(int(0.2*shape_im[2]), int(0.5*shape_im[2])))/shape_im[2], np.mean(big_predict[:,1].reshape((num_choices, -1)), axis = 0), label = "High RMS image")
    plt.xlabel("Slice position")
    plt.ylabel("Misclassification proba")
    plt.title("Axial slice (down)")
    plt.legend()

    
    
    not_noised_predict = np.zeros((0, 2))
    small_predict = np.zeros((0, 2))
    med_predict = np.zeros((0, 2))
    big_predict = np.zeros((0, 2))

    for k in tqdm(range(num_choices)):
        XX_not_noised = []
        XX_small = []
        XX_med = []
        XX_big = []

        for sliice in (range(int(0.5*shape_im[2]), int(0.8*shape_im[2]))):
            XX_not_noised.append(np.expand_dims(quadriview(not_noised_images[k], sag_slice, cor_slice, ax1_slice, sliice) ,2))
            XX_small.append(np.expand_dims(quadriview(small_images[k], sag_slice, cor_slice, ax1_slice, sliice) ,2))
            XX_med.append(np.expand_dims(quadriview(med_images[k], sag_slice, cor_slice, ax1_slice, sliice) ,2))
            XX_big.append(np.expand_dims(quadriview(big_images[k], sag_slice, cor_slice, ax1_slice, sliice) ,2))
        XX_not_noised = np.array(XX_not_noised)
        XX_small = np.array(XX_small)
        XX_med = np.array(XX_med)
        XX_big = np.array(XX_big)
        not_noised_predict = np.concatenate((not_noised_predict, model.predict(XX_not_noised)), axis = 0)
        small_predict = np.concatenate((small_predict, model.predict(XX_small)), axis = 0)
        med_predict = np.concatenate((med_predict, model.predict(XX_med)), axis = 0)
        big_predict = np.concatenate((big_predict, model.predict(XX_big)), axis = 0)
        
    ax3 = fig.add_subplot(gs[2, 1])
    plt.plot(np.array(range(int(0.5*shape_im[2]), int(0.8*shape_im[2])))/shape_im[2], np.mean(not_noised_predict[:,0].reshape((num_choices, -1)), axis = 0), label = "Not noised image")
    plt.plot(np.array(range(int(0.5*shape_im[2]), int(0.8*shape_im[2])))/shape_im[2], np.mean(small_predict[:,1].reshape((num_choices, -1)), axis = 0), label = "Small RMS image")
    plt.plot(np.array(range(int(0.5*shape_im[2]), int(0.8*shape_im[2])))/shape_im[2], np.mean(med_predict[:,1].reshape((num_choices, -1)), axis = 0), label = "Middle-range RMS image")
    plt.plot(np.array(range(int(0.5*shape_im[2]), int(0.8*shape_im[2])))/shape_im[2], np.mean(big_predict[:,1].reshape((num_choices, -1)), axis = 0), label = "High RMS image")
    plt.xlabel("Slice position")
    plt.ylabel("Misclassification proba")
    plt.title("Axial slice (up)")
    plt.legend()


    model_name = model.name
    if not os.path.isdir(prefix + "Checks/Check_"+model_name):
        os.mkdir(prefix + "Checks/Check_"+model_name)
    file_path = prefix + "Checks/Check_"+model_name+"/"
    plt.savefig(file_path+model_name+"_slice_pos_impact.jpg")
    print("Test with regards to slice position finished")
    

def ROC_curve(model, seed = None, prefix = "", **kwargs):
    model_name = model.name
    if not os.path.isdir(prefix + "Checks/Check_"+model_name):
        os.mkdir(prefix + "Checks/Check_"+model_name)
    if seed is None:
        seed = np.random.randint(0, 1000)
    file_path = prefix + "Checks/Check_"+model_name+"/"    
    pc_test = generators.Quadriview_DataGenerator(prefix = prefix, seed = seed, **kwargs)
    pc_test.metadata;
    X, y_true = pc_test.get_data()
    y_pred = model.predict(X, verbose = 1)
    
    
    fpr, tpr, thresholds = roc_curve(np.argmax(y_true, axis = 1), y_pred[:,1])
    auc_rate = auc(fpr, tpr)
    fig = plt.figure(figsize = (15, 15))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='Area = {:.3f}'.format(auc_rate))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig(file_path+ "ROC_"+model_name+".jpg")
    print("ROC Curve generated!")


def features_heatmap(model, prefix = "", batch_size = 16, **kwargs):
    pc_test = generators.Quadriview_DataGenerator(batch_size = batch_size, prefix = prefix, **kwargs)
    pc_test.metadata;
    X, y = pc_test.get_data()
    y_pred = model.predict(X, verbose = 1)
    
    model_name = model.name
    if not os.path.isdir(prefix + "Checks/Check_"+model_name):
        os.mkdir(prefix + "Checks/Check_"+model_name)
    file_path = prefix + "Checks/Check_"+model_name+"/"    
    model_2 = model
    layer_idx = utils.find_layer_idx(model_2, "dense_2")
    # Utility to search for layer index by name. 
    # Alternatively we can specify this as -1 since it corresponds to the last layer.

    model_2.layers[layer_idx].activation = activations.linear
    model_2 = utils.apply_modifications(model_2)

    penultimate_layer = utils.find_layer_idx(model_2, 'max_pooling2d_6')

    # Swap softmax with linear

    
    plt.axis("off");
    X_0 = X[np.where((np.argmax(y, axis = 1) == 0) * (np.argmax(y_pred, axis = 1) == 0))]
    X_1 = X[np.where((np.argmax(y, axis = 1) == 1) * (np.argmax(y_pred, axis = 1) == 1))]
    X_diff = X[np.where(np.argmax(y, axis = 1) != np.argmax(y_pred, axis = 1))]
    y_diff =  y[np.where(np.argmax(y, axis = 1) != np.argmax(y_pred, axis = 1))]
    y_p_diff = y_pred[np.where(np.argmax(y, axis = 1) != np.argmax(y_pred, axis = 1))]
    fig0, axs0 = plt.subplots(nrows = X_0.shape[0], ncols = 3, figsize = (3*10, X_0.shape[0]*10), squeeze=False);
    fig1, axs1 = plt.subplots(nrows = X_1.shape[0], ncols = 3, figsize = (3*10, X_1.shape[0]*10), squeeze=False);
    figd, axsd = plt.subplots(nrows = X_diff.shape[0], ncols = 3, figsize = (3*10, X_diff.shape[0]*10), squeeze=False);
    
    for k in tqdm(range(X_0.shape[0])):
        grads = visualize_saliency(model_2, layer_idx, filter_indices=0, seed_input = np.expand_dims(X_0[k], 0),
                                  backprop_modifier='guided')
        # Plot with 'jet' colormap to visualize as a heatmap.
        axs0[k, 0].imshow(X_0[k,:,:,0].T, cmap='Greys_r', origin = "lower")
        plt.axis("off")
        axs0[k, 0].get_xaxis().set_visible(False)
        axs0[k, 0].get_yaxis().set_visible(False)
        axs0[k, 1].imshow(grads.T, cmap='jet', origin = "lower")
        axs0[k, 1].set_title("Class: {}, Pred: {}".format(0, 0), fontdict = {'fontsize': 'medium',
 'fontweight' : rcParams['axes.titleweight'], 'verticalalignment': 'baseline', 'horizontalalignment': 'center'})

        plt.axis("off")
        #axs[k, 1].get_xaxis().set_visible(False)
        #axs[k, 1].get_yaxis().set_visible(False)
        grads_map = visualize_cam(model_2, layer_idx, filter_indices=0,
                                  seed_input=np.expand_dims(X_0[k,:,:], 0), penultimate_layer_idx=penultimate_layer,
                                  backprop_modifier='guided')        
            # Lets overlay the heatmap onto original image.    
        jet_heatmap = np.uint8(cm.jet(grads_map)[..., :3] * 255)
        img = np.uint8(cm.Greys_r(X_0[k,:,:, 0])[..., :3] * 255)
        axs0[k, 2].imshow((overlay(jet_heatmap, img)).transpose((1, 0, 2)), origin = "lower")
        plt.axis("off")
        axs0[k, 2].get_xaxis().set_visible(False)
        axs0[k, 2].get_yaxis().set_visible(False)

    fig0.savefig(file_path+"heat_map_0_"+model_name+".png")
    
    
    for k in tqdm(range(X_1.shape[0])):
        grads = visualize_saliency(model_2, layer_idx, filter_indices=1, seed_input = np.expand_dims(X_1[k], 0),
                                  backprop_modifier='guided')
        # Plot with 'jet' colormap to visualize as a heatmap.
        axs1[k, 0].imshow(X_1[k,:,:,0].T, cmap='Greys_r', origin = "lower")
        plt.axis("off")
        axs1[k, 0].get_xaxis().set_visible(False)
        axs1[k, 0].get_yaxis().set_visible(False)
        axs1[k, 1].imshow(grads.T, cmap='jet', origin = "lower")
        axs1[k, 1].set_title("Class: {}, Pred: {}".format(1, 1), fontdict = {'fontsize': 'medium',
 'fontweight' : rcParams['axes.titleweight'], 'verticalalignment': 'baseline', 'horizontalalignment': 'center'})

        plt.axis("off")
        #axs[k, 1].get_xaxis().set_visible(False)
        #axs[k, 1].get_yaxis().set_visible(False)
        grads_map = visualize_cam(model_2, layer_idx, filter_indices=1,
                                  seed_input=np.expand_dims(X_1[k,:,:], 0), penultimate_layer_idx=penultimate_layer,
                                  backprop_modifier='guided')        
            # Lets overlay the heatmap onto original image.    
        jet_heatmap = np.uint8(cm.jet(grads_map)[..., :3] * 255)
        img = np.uint8(cm.Greys_r(X_1[k,:,:, 0])[..., :3] * 255)
        axs1[k, 2].imshow((overlay(jet_heatmap, img)).transpose((1, 0, 2)), origin = "lower")
        plt.axis("off")
        axs1[k, 2].get_xaxis().set_visible(False)
        axs1[k, 2].get_yaxis().set_visible(False)

    fig1.savefig(file_path+"heat_map_1_"+model_name+".png")
    
    
    for k in tqdm(range(X_diff.shape[0])):
        grads = visualize_saliency(model_2, layer_idx, filter_indices=np.argmax(y_p_diff[k]), seed_input = np.expand_dims(X_diff[k], 0),
                                  backprop_modifier='guided')
        # Plot with 'jet' colormap to visualize as a heatmap.
        axsd[k, 0].imshow(X_diff[k,:,:,0].T, cmap='Greys_r', origin = "lower")
        plt.axis("off")
        axsd[k, 0].get_xaxis().set_visible(False)
        axsd[k, 0].get_yaxis().set_visible(False)
        axsd[k, 1].imshow(grads.T, cmap='jet', origin = "lower")
        axsd[k, 1].set_title("Class: {}, Pred: {}, Prob: {}".format(np.argmax(y_diff[k]), np.argmax(y_p_diff[k]), np.max(y_p_diff[k])), fontdict = {'fontsize': 'medium',
 'fontweight' : rcParams['axes.titleweight'], 'verticalalignment': 'baseline', 'horizontalalignment': 'center'})

        plt.axis("off")
        #axs[k, 1].get_xaxis().set_visible(False)
        #axs[k, 1].get_yaxis().set_visible(False)
        grads_map = visualize_cam(model_2, layer_idx, filter_indices=np.argmax(y_p_diff[k]),
                                  seed_input=np.expand_dims(X_diff[k,:,:], 0), penultimate_layer_idx=penultimate_layer,
                                  backprop_modifier='guided')        
            # Lets overlay the heatmap onto original image.    
        jet_heatmap = np.uint8(cm.jet(grads_map)[..., :3] * 255)
        img = np.uint8(cm.Greys_r(X_diff[k,:,:, 0])[..., :3] * 255)
        axsd[k, 2].imshow((overlay(jet_heatmap, img)).transpose((1, 0, 2)), origin = "lower")
        plt.axis("off")
        axsd[k, 2].get_xaxis().set_visible(False)
        axsd[k, 2].get_yaxis().set_visible(False)

    figd.savefig(file_path+"heat_map_diff_"+model_name+".png")

    
    
    
    
    
    
    
    
    
