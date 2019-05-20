import generators
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
import numpy as np
import os
import pandas as pd
from utils import quadriview, normalization_func
import nibabel as nb
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm
import matplotlib.gridspec as gridspec
from generate_df import generate_df



def summary_model_training(model):
    history = cor_model.history.history
    fig = plt.figure(figsize = (15, 15))
    plt.subplots(211)
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.subplots(212)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    model_name = model.name
    if not os.path.isdir("./Check_"+model_name):
        os.mkdir("./Check_"+model_name)
    file_path = "./Check_"+model_name+"/"
    plt.savefig(file_path+model_name+"_learning_curves.png")

def model_eval(model, dataset_test, steps=50, **kwargs):
    model_name = model.name
    if not os.path.isdir("./Check_"+model_name):
        os.mkdir("./Check_"+model_name)
    file_path = "./Check_"+model_name+"/"
    if dataset_test[-3:] == "csv":
        gen1 = generators.Quadriview_DataGenerator(csv_file = dataset_test, generator_type = "balanced", **kwargs)
        gen2 = generators.Quadriview_DataGenerator(csv_file = dataset_test, generator_type = "always_noised", **kwargs)
        gen3 = generators.Quadriview_DataGenerator(csv_file = dataset_test, generator_type = "always_clean",**kwargs)
    else:
        gen1 = generators.Quadriview_DataGenerator(csv_file = dataset_test, generator_type = "balanced", **kwargs)
        gen2 = generators.Quadriview_DataGenerator(csv_file = dataset_test, generator_type = "always_noised", **kwargs)
        gen3 = generators.Quadriview_DataGenerator(csv_file = dataset_test, generator_type = "always_clean",**kwargs)
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

def model_apply(model, dataset_test, name = "output.jpg", orientation = "vertical", font_path = "/usr/share/fonts/truetype/freefont/FreeMono.ttf", **kwargs):
    model_name = model.name
    if not os.path.isdir("./Check_"+model_name):
        os.mkdir("./Check_"+model_name)
    file_path = "./Check_"+model_name+"/"
    if dataset_test[-3:] == "csv":
        pc_test = generators.Quadriview_DataGenerator(csv_file = dataset_test, **kwargs)
    else:
        pc_test = generators.Quadriview_DataGenerator(img_root_dir = dataset_test, **kwargs)
    pc_test.metadata;
    y_pred = model.predict_generator(pc_test, steps = 1, verbose = 1, use_multiprocessing=True, workers = 20)
    print(y_pred.dtype)
    if dataset_test[-3:] == "csv":
        pc_test = generators.Quadriview_DataGenerator(csv_file = dataset_test, **kwargs)
    else:
        pc_test = generators.Quadriview_DataGenerator(img_root_dir = dataset_test, **kwargs)
    pc_test.metadata;
    y = pc_test.display(name = name, orientation = orientation, abs_pos = 155)
    y = np.argmax(y, axis = 1)
    img = Image.open(name)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, 8, encoding="unic")
    if orientation == "vertical":
        for k in range(len(y_pred)):
            cl = np.argmax(y_pred[k])
            pba =  np.round(y_pred[k, cl], 3)
            draw.text((205, 5+400*k),"Pred: {} Prob:".format(cl) + str(pba), (255, 255, 255, 255), font=font)
    else:
        for k in range(len(y_pred)):
            cl = np.argmax(y_pred[k])
            pba =  np.round(y_pred[k, cl], 3)
            draw.text((205+400*k, 5),"Pred: {} Prob:".format(cl) + str(pba), (255, 255, 255, 255), font=font)
        
    img.save(file_path + model_name+"_"+name)
    os.remove(name)
    acc = np.mean(np.argmax(y_pred, axis = 1) == y)
    t_p = np.sum(np.argmax(y_pred, axis = 1) * y)/np.sum(y)
    t_n = np.sum((1-np.argmax(y_pred, axis = 1)) * (1-y))/np.sum(1-y)
    print("Accuracy: {}, True_positive_rate: {},  True_negative_rate: {}".format(np.round(acc, 3), np.round(t_p, 3), np.round(t_n, 3)))
    return acc, t_p, t_n

def test_noise_characs(model, dataset_test, measure = ["RMS", "MSE", "Corr"], prefix = "", normalization = True):
    if dataset_test[-3:] == "csv":
        tab = pd.read_csv(dataset_test, sep = ",", index_col = 0)
    else:
        tab = generate_df(dataset_test)
    
    length = len(tab)
    XX = []
    model_name = model.name
    if not os.path.isdir("./Check_"+model_name):
        os.mkdir("./Check_"+model_name)
    file_path = "./Check_"+model_name+"/"
    if type(measure) == str:
        measure = [measure]
    for k in tqdm(range(length)):
        image = nb.load(prefix + tab.iloc[k].img_file).get_fdata()
        if normalization:
            image = normalization_func(image)
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

def test_slice(model, dataset_test, num_choices = 40, normalization = True):
    if dataset_test[-3:] == "csv":
        csv = pd.read_csv(dataset_test, sep = ",", index_col = 0)
    else:
        csv = generate_df(dataset_test)
    ind_not_noised = np.random.choice(np.where(csv.noise.values == 1)[0], num_choices)
    ind_small_RMS = np.random.choice(np.where((csv.noise.values == 0) & (csv.RMS.values < 10))[0], num_choices)
    ind_med_RMS = np.random.choice(np.where((csv.noise.values == 0) & (csv.RMS.values > 10) & (csv.RMS.values < 20))[0], num_choices)
    ind_big_RMS = np.random.choice(np.where((csv.noise.values == 0) & (csv.RMS.values > 20))[0], num_choices)
    if normalization:
        not_noised_images = np.array([normalization_func(nb.load(csv.iloc[k].img_file).get_fdata()) for k in tqdm(ind_not_noised)])
        small_RMS_images = np.array([normalization_func(nb.load(csv.iloc[k].img_file).get_fdata()) for k in tqdm(ind_small_RMS)])
        med_RMS_images = np.array([normalization_func(nb.load(csv.iloc[k].img_file).get_fdata()) for k in tqdm(ind_med_RMS)])
        big_RMS_images = np.array([normalization_func(nb.load(csv.iloc[k].img_file).get_fdata()) for k in tqdm(ind_big_RMS)])
    else:
        not_noised_images = np.array([nb.load(csv.iloc[k].img_file).get_fdata() for k in tqdm(ind_not_noised)])
        small_RMS_images = np.array([nb.load(csv.iloc[k].img_file).get_fdata() for k in tqdm(ind_small_RMS)])
        med_RMS_images = np.array([nb.load(csv.iloc[k].img_file).get_fdata() for k in tqdm(ind_med_RMS)])
        big_RMS_images = np.array([nb.load(csv.iloc[k].img_file).get_fdata() for k in tqdm(ind_big_RMS)])
    shape_im = not_noised_images[0].shape
    sag_slice = int(0.5*shape_im[0])    
    cor_slice = int(0.5*shape_im[1])
    ax1_slice = int(0.4*shape_im[2])
    ax2_slice = int(0.6*shape_im[2])
    
    not_noised_predict = np.zeros((0, 2))
    small_RMS_predict = np.zeros((0, 2))
    med_RMS_predict = np.zeros((0, 2))
    big_RMS_predict = np.zeros((0, 2))
    for k in tqdm(range(num_choices)):
        XX_not_noised = []
        XX_small_RMS = []
        XX_med_RMS = []
        XX_big_RMS = []
        for sliice in (range(int(0.2*shape_im[0]), int(0.8*shape_im[0]))):
            XX_not_noised.append(np.expand_dims(quadriview(not_noised_images[k], sliice, cor_slice, ax1_slice, ax2_slice) ,2))
            XX_small_RMS.append(np.expand_dims(quadriview(small_RMS_images[k], sliice, cor_slice, ax1_slice, ax2_slice) ,2))
            XX_med_RMS.append(np.expand_dims(quadriview(med_RMS_images[k], sliice, cor_slice, ax1_slice, ax2_slice) ,2))
            XX_big_RMS.append(np.expand_dims(quadriview(big_RMS_images[k], sliice, cor_slice, ax1_slice, ax2_slice) ,2))
        XX_not_noised = np.array(XX_not_noised)
        XX_small_RMS = np.array(XX_small_RMS)
        XX_med_RMS = np.array(XX_med_RMS)
        XX_big_RMS = np.array(XX_big_RMS)
        not_noised_predict = np.concatenate((not_noised_predict, model.predict(XX_not_noised)), axis = 0)
        small_RMS_predict = np.concatenate((small_RMS_predict, model.predict(XX_small_RMS)), axis = 0)
        med_RMS_predict = np.concatenate((med_RMS_predict, model.predict(XX_med_RMS)), axis = 0)
        big_RMS_predict = np.concatenate((big_RMS_predict, model.predict(XX_big_RMS)), axis = 0)
    
    fig = plt.figure(figsize = (15, 15))
    gs = gridspec.GridSpec(nrows=3, ncols=2, figure=fig)

    ax0 = fig.add_subplot(gs[0, :])
    plt.plot(np.array(range(int(0.2*shape_im[0]), int(0.8*shape_im[0])))/shape_im[0], np.mean(not_noised_predict[:,0].reshape((num_choices, -1)), axis = 0), label = "Not noised image")
    plt.plot(np.array(range(int(0.2*shape_im[0]), int(0.8*shape_im[0])))/shape_im[0], np.mean(small_RMS_predict[:,1].reshape((num_choices, -1)), axis = 0), label = "Small RMS image")
    plt.plot(np.array(range(int(0.2*shape_im[0]), int(0.8*shape_im[0])))/shape_im[0], np.mean(med_RMS_predict[:,1].reshape((num_choices, -1)), axis = 0), label = "Middle-range RMS image")
    plt.plot(np.array(range(int(0.2*shape_im[0]), int(0.8*shape_im[0])))/shape_im[0], np.mean(big_RMS_predict[:,1].reshape((num_choices, -1)), axis = 0), label = "High RMS image")
    plt.xlabel("Slice position")
    plt.title("Sagittal slice")
    plt.ylabel("Misclassification proba")
    plt.legend()
    

    not_noised_predict = np.zeros((0, 2))
    small_RMS_predict = np.zeros((0, 2))
    med_RMS_predict = np.zeros((0, 2))
    big_RMS_predict = np.zeros((0, 2))
    for k in tqdm(range(num_choices)):
        XX_not_noised = []
        XX_small_RMS = []
        XX_med_RMS = []
        XX_big_RMS = []
        for sliice in (range(int(0.2*shape_im[1]), int(0.8*shape_im[1]))):
            XX_not_noised.append(np.expand_dims(quadriview(not_noised_images[k], sag_slice, sliice, ax1_slice, ax2_slice) ,2))
            XX_small_RMS.append(np.expand_dims(quadriview(small_RMS_images[k], sag_slice, sliice, ax1_slice, ax2_slice) ,2))
            XX_med_RMS.append(np.expand_dims(quadriview(med_RMS_images[k], sag_slice, sliice, ax1_slice, ax2_slice) ,2))
            XX_big_RMS.append(np.expand_dims(quadriview(big_RMS_images[k], sag_slice, sliice, ax1_slice, ax2_slice) ,2))
        XX_not_noised = np.array(XX_not_noised)
        XX_small_RMS = np.array(XX_small_RMS)
        XX_med_RMS = np.array(XX_med_RMS)
        XX_big_RMS = np.array(XX_big_RMS)
        not_noised_predict = np.concatenate((not_noised_predict, model.predict(XX_not_noised)), axis = 0)
        small_RMS_predict = np.concatenate((small_RMS_predict, model.predict(XX_small_RMS)), axis = 0)
        med_RMS_predict = np.concatenate((med_RMS_predict, model.predict(XX_med_RMS)), axis = 0)
        big_RMS_predict = np.concatenate((big_RMS_predict, model.predict(XX_big_RMS)), axis = 0)
    
    ax1 = fig.add_subplot(gs[1, :])
    plt.plot(np.array(range(int(0.2*shape_im[1]), int(0.8*shape_im[1])))/shape_im[1], np.mean(not_noised_predict[:,0].reshape((num_choices, -1)), axis = 0), label = "Not noised image")
    plt.plot(np.array(range(int(0.2*shape_im[1]), int(0.8*shape_im[1])))/shape_im[1], np.mean(small_RMS_predict[:,1].reshape((num_choices, -1)), axis = 0), label = "Small RMS image")
    plt.plot(np.array(range(int(0.2*shape_im[1]), int(0.8*shape_im[1])))/shape_im[1], np.mean(med_RMS_predict[:,1].reshape((num_choices, -1)), axis = 0), label = "Middle-range RMS image")
    plt.plot(np.array(range(int(0.2*shape_im[1]), int(0.8*shape_im[1])))/shape_im[1], np.mean(big_RMS_predict[:,1].reshape((num_choices, -1)), axis = 0), label = "High RMS image")
    plt.xlabel("Slice position")
    plt.ylabel("Misclassification proba")
    plt.title("Coronal slice")
    plt.legend()

    

    not_noised_predict = np.zeros((0, 2))
    small_RMS_predict = np.zeros((0, 2))
    med_RMS_predict = np.zeros((0, 2))
    big_RMS_predict = np.zeros((0, 2))

    for k in tqdm(range(num_choices)):
        XX_not_noised = []
        XX_small_RMS = []
        XX_med_RMS = []
        XX_big_RMS = []
        for sliice in (range(int(0.2*shape_im[2]), int(0.5*shape_im[2]))):
            XX_not_noised.append(np.expand_dims(quadriview(not_noised_images[k], sag_slice, cor_slice, sliice, ax2_slice) ,2))
            XX_small_RMS.append(np.expand_dims(quadriview(small_RMS_images[k], sag_slice, cor_slice, sliice, ax2_slice) ,2))
            XX_med_RMS.append(np.expand_dims(quadriview(med_RMS_images[k], sag_slice, cor_slice, sliice, ax2_slice) ,2))
            XX_big_RMS.append(np.expand_dims(quadriview(big_RMS_images[k], sag_slice, cor_slice, sliice, ax2_slice) ,2))
        XX_not_noised = np.array(XX_not_noised)
        XX_small_RMS = np.array(XX_small_RMS)
        XX_med_RMS = np.array(XX_med_RMS)
        XX_big_RMS = np.array(XX_big_RMS)
        not_noised_predict = np.concatenate((not_noised_predict, model.predict(XX_not_noised)), axis = 0)
        small_RMS_predict = np.concatenate((small_RMS_predict, model.predict(XX_small_RMS)), axis = 0)
        med_RMS_predict = np.concatenate((med_RMS_predict, model.predict(XX_med_RMS)), axis = 0)
        big_RMS_predict = np.concatenate((big_RMS_predict, model.predict(XX_big_RMS)), axis = 0)
    
    ax2 = fig.add_subplot(gs[2, 0])
    plt.plot(np.array(range(int(0.2*shape_im[2]), int(0.5*shape_im[2])))/shape_im[2], np.mean(not_noised_predict[:,0].reshape((num_choices, -1)), axis = 0), label = "Not noised image")
    plt.plot(np.array(range(int(0.2*shape_im[2]), int(0.5*shape_im[2])))/shape_im[2], np.mean(small_RMS_predict[:,1].reshape((num_choices, -1)), axis = 0), label = "Small RMS image")
    plt.plot(np.array(range(int(0.2*shape_im[2]), int(0.5*shape_im[2])))/shape_im[2], np.mean(med_RMS_predict[:,1].reshape((num_choices, -1)), axis = 0), label = "Middle-range RMS image")
    plt.plot(np.array(range(int(0.2*shape_im[2]), int(0.5*shape_im[2])))/shape_im[2], np.mean(big_RMS_predict[:,1].reshape((num_choices, -1)), axis = 0), label = "High RMS image")
    plt.xlabel("Slice position")
    plt.ylabel("Misclassification proba")
    plt.title("Axial slice (down)")
    plt.legend()

    
    
    not_noised_predict = np.zeros((0, 2))
    small_RMS_predict = np.zeros((0, 2))
    med_RMS_predict = np.zeros((0, 2))
    big_RMS_predict = np.zeros((0, 2))

    for k in tqdm(range(num_choices)):
        XX_not_noised = []
        XX_small_RMS = []
        XX_med_RMS = []
        XX_big_RMS = []

        for sliice in (range(int(0.5*shape_im[2]), int(0.8*shape_im[2]))):
            XX_not_noised.append(np.expand_dims(quadriview(not_noised_images[k], sag_slice, cor_slice, ax1_slice, sliice) ,2))
            XX_small_RMS.append(np.expand_dims(quadriview(small_RMS_images[k], sag_slice, cor_slice, ax1_slice, sliice) ,2))
            XX_med_RMS.append(np.expand_dims(quadriview(med_RMS_images[k], sag_slice, cor_slice, ax1_slice, sliice) ,2))
            XX_big_RMS.append(np.expand_dims(quadriview(big_RMS_images[k], sag_slice, cor_slice, ax1_slice, sliice) ,2))
        XX_not_noised = np.array(XX_not_noised)
        XX_small_RMS = np.array(XX_small_RMS)
        XX_med_RMS = np.array(XX_med_RMS)
        XX_big_RMS = np.array(XX_big_RMS)
        not_noised_predict = np.concatenate((not_noised_predict, model.predict(XX_not_noised)), axis = 0)
        small_RMS_predict = np.concatenate((small_RMS_predict, model.predict(XX_small_RMS)), axis = 0)
        med_RMS_predict = np.concatenate((med_RMS_predict, model.predict(XX_med_RMS)), axis = 0)
        big_RMS_predict = np.concatenate((big_RMS_predict, model.predict(XX_big_RMS)), axis = 0)
        
    ax3 = fig.add_subplot(gs[2, 1])
    plt.plot(np.array(range(int(0.5*shape_im[2]), int(0.8*shape_im[2])))/shape_im[2], np.mean(not_noised_predict[:,0].reshape((num_choices, -1)), axis = 0), label = "Not noised image")
    plt.plot(np.array(range(int(0.5*shape_im[2]), int(0.8*shape_im[2])))/shape_im[2], np.mean(small_RMS_predict[:,1].reshape((num_choices, -1)), axis = 0), label = "Small RMS image")
    plt.plot(np.array(range(int(0.5*shape_im[2]), int(0.8*shape_im[2])))/shape_im[2], np.mean(med_RMS_predict[:,1].reshape((num_choices, -1)), axis = 0), label = "Middle-range RMS image")
    plt.plot(np.array(range(int(0.5*shape_im[2]), int(0.8*shape_im[2])))/shape_im[2], np.mean(big_RMS_predict[:,1].reshape((num_choices, -1)), axis = 0), label = "High RMS image")
    plt.xlabel("Slice position")
    plt.ylabel("Misclassification proba")
    plt.title("Axial slice (up)")
    plt.legend()


    model_name = model.name
    if not os.path.isdir("./Check_"+model_name):
        os.mkdir("./Check_"+model_name)
    file_path = "./Check_"+model_name+"/"
    plt.savefig(file_path+model_name+"_slice_pos_impact.jpg")
    
    
    
    
    
    
