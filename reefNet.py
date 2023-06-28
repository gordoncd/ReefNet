## local implementation of ReefNet Pacific Atoll segmentation using UNet architecture
## colab implementation: https://colab.research.google.com/drive/1NeM0HsvqyCEFnhzhlucrCa5E3RlR4_hC?usp=sharing
##author: Gordon Doore, Drew Hinton, Sameer Khan
##01/22/2023

#imports and dependencies
import model as mod
import os
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from keras import backend as K
from sklearn.utils import class_weight
from keras.metrics import MeanIoU

X_SIZE = 448
Y_SIZE = 448


def getModel(numclasses=3, imgheight=448, imgwidth=448 , channels=3):
    return mod.multi_unet_model(numclasses, imgheight, imgwidth, channels )

def loadData(imageDir, maskDir = None):
    #from imageDir reads all .npy files in directory and returns them as array
    #masks is param which lets one only return those with a mask in 
    #directory masks
    #masks are jpg and imgs are .npy array
    list = []
    masks = []
    dirs = sorted(os.listdir(imageDir))
    masks = sorted(os.listdir(maskDir))
    for object in dirs:
        for mask in maskDir:
            if masks[:-7] in object and mask.endswith("l64.jpg"):
                img = np.load(imageDir+object)
                img = cv2.resize(img, (X_SIZE, Y_SIZE), interpolation= cv2.INTER_CUBIC)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
                truth = np.load(imageDir+object)
                truth = cv2.resize(truth, (X_SIZE, Y_SIZE), interpolation= cv2.INTER_CUBIC)                    
                list.append(img) 
                masks.append(truth)   
    return np.array(list), np.array(masks)

def fixImageRanges(images):
    #masks have unfixed pixel vals, so this thresholds them to represent a single value
    toReturn= []
    for imgIdx in range(images.shape[0]):
        img = cv2.cvtColor(images[imgIdx], cv2.COLOR_BGR2GRAY)
        ret,ocean = cv2.threshold(img,50,255,cv2.THRESH_BINARY_INV)
        ret,reef = cv2.threshold(img,200,255,cv2.THRESH_BINARY)
        ret,reefLand = cv2.threshold(img,50,255,cv2.THRESH_BINARY)
        land = reefLand - reef
        label = []
        label.append(reef)
        label.append(land)
        label.append(ocean)
        label = np.array(label)
        label = np.swapaxes(label, 0,1)
        label = np.swapaxes(label, 1,2)
        label = label/255
        toReturn.append(label)
    return np.array(toReturn)

def augmentWithRotation(images, masks):
    #rotates images and masks and adds them original set
    #masks and images must have same shape
    
    imagesRot = images
    if images.shape[0] == masks.shape[0]:
        for i in range(imagesRot.shape[0]):
            img = imagesRot[i]
            rotate = np.swapaxes(img,1,2)
            mask = masks[i]
            maskRotate = np.swapaxes(mask,1,2)
            images.append(rotate)
            masks.append(maskRotate)
    return images, masks
        
def splitData(images, masks, test_size_ratio = .1):
    #splits data into train and test groups
    X1, X_test, y1, y_test = train_test_split(images, masks, test_size = test_size_ratio, random_state = 0)

    #Further split training data t0 a smaller subset for quick testing of models
    X_train, X_do_not_use, y_train, y_do_not_use = train_test_split(X1, y1, test_size = 2*test_size_ratio, random_state = 0)

    print("Class values in the dataset are ... ", np.unique(y_train))  # 0 is the background/few unlabeled 

    return X1, X_test, y1, y_test, X_train, X_do_not_use, y_train, y_do_not_use


def recall_m(y_true, y_pred):
    #manually calculate recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    #manually calculate precision
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    #manually calculate f1 score
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def trainModel():
    #load data
    
    images, masks = loadData("NIR/", "JPG_Labeled/")
    masks = fixImageRanges(masks)
    images, masks = augmentWithRotation(images, masks)
    #now need to split data
    X1, X_test, y1, y_test, X_train, X_do_not_use, y_train, y_do_not_use = splitData(images, masks)
    model = getModel() 
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[f1_m])
    model.summary()

    history = model.fit(X_train, y_train, 
                    batch_size = 4, 
                    verbose=1, 
                    epochs=50, 
                    validation_data=(X_test, y_test), 
                    #class_weight=class_weights,
                    shuffle=False)
    
    model.save('aiZoom-k-means-augment-5.hdf5')
    return history, model, X_test, y_test

def evaluateModel(pathToModel, model, X_test, y_test):

    model = model.load_weights(pathToModel)
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis = 3)

    n_classes = 3
    testIndex = 3
    IOU_keras = MeanIoU(num_classes=n_classes)  
    iou1 = IOU_keras.update_state(y_test[:,:,:,0], np.around(y_pred[:,:,:,0],decimals=0))
    res1 = IOU_keras.result().numpy()

    iou2 = IOU_keras.update_state(y_test[:,:,:,1], np.around(y_pred[:,:,:,1],decimals=0))
    res2 = IOU_keras.result().numpy()

    iou3 = IOU_keras.update_state(y_test[:,:,:,2], np.around(y_pred[:,:,:,2],decimals=0))
    res3 = IOU_keras.result().numpy()

    v_IoU = res1
    r_IoU = res2
    o_IoU = res3
    m_IoU = (res1+res2+res3)/3


    #F1 score
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc',f1_m,precision_m, recall_m])
    loss, accuracy, f1_score, precision, recall = model.evaluate(y_test, y_pred, verbose=0)
    


    return m_IoU, v_IoU, r_IoU, o_IoU, f1_score, loss, accuracy, precision, recall #maybe add class specific f1 as well?
    