## Local Implementation of ReefNet Pacific Atoll Segmentation using UNet Architecture
## Colab Implementation: [Link to Colab Notebook]
## UNet is based on the work by Olaf Ronneberger et al.
## Authors: Gordon Doore, Drew Hinton, Sameer Khan
## Date: 01/22/2023

# Importing dependencies
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

# Constants
X_SIZE = 448
Y_SIZE = 448

def getModel(numclasses=3, imgheight=448, imgwidth=448, channels=3):
    """
    Function to get the UNet model for image segmentation.

    Args:
    - numclasses (int): Number of output classes.
    - imgheight (int): Height of the input images.
    - imgwidth (int): Width of the input images.
    - channels (int): Number of color channels in the input images.

    Returns:
    - model: The UNet model.
    """
    return mod.multi_unet_model(numclasses, imgheight, imgwidth, channels)

def loadData(imageDir, maskDir=None):
    """
    Function to load image and mask data from directories.

    Args:
    - imageDir (str): Directory path containing image files.
    - maskDir (str): Directory path containing mask files.

    Returns:
    - tuple: Tuple containing image and mask arrays.
    """
    list = []
    masks = []
    dirs = sorted(os.listdir(imageDir))
    masks = sorted(os.listdir(maskDir))
    for object in dirs:
        for mask in maskDir:
            if masks[:-7] in object and mask.endswith("l64.jpg"):
                img = np.load(imageDir + object)
                img = cv2.resize(img, (X_SIZE, Y_SIZE), interpolation=cv2.INTER_CUBIC)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                truth = np.load(imageDir + object)
                truth = cv2.resize(truth, (X_SIZE, Y_SIZE), interpolation=cv2.INTER_CUBIC)
                list.append(img)
                masks.append(truth)
    return np.array(list), np.array(masks)

def fixImageRanges(images):
    """
    Function to fix pixel value ranges of mask images.

    Args:
    - images (ndarray): Array of mask images.

    Returns:
    - ndarray: Array of fixed mask images.
    """
    toReturn = []
    for imgIdx in range(images.shape[0]):
        img = cv2.cvtColor(images[imgIdx], cv2.COLOR_BGR2GRAY)
        ret, ocean = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY_INV)
        ret, reef = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
        ret, reefLand = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
        land = reefLand - reef
        label = []
        label.append(reef)
        label.append(land)
        label.append(ocean)
        label = np.array(label)
        label = np.swapaxes(label, 0, 1)
        label = np.swapaxes(label, 1, 2)
        label = label / 255
        toReturn.append(label)
    return np.array(toReturn)

def augmentWithRotation(images, masks):
    """
    Function to augment images and masks by rotation.

    Args:
    - images (ndarray): Array of input images.
    - masks (ndarray): Array of mask images.

    Returns:
    - tuple: Tuple containing augmented image and mask arrays.
    """
    imagesRot = images
    if images.shape[0] == masks.shape[0]:
        for i in range(imagesRot.shape[0]):
            img = imagesRot[i]
            rotate = np.swapaxes(img, 1, 2)
            mask = masks[i]
            maskRotate = np.swapaxes(mask, 1, 2)
            images.append(rotate)
            masks.append(maskRotate)
    return images, masks

def splitData(images, masks, test_size_ratio=.1):
    """
    Function to split data into train and test sets.

    Args:
    - images (ndarray): Array of input images.
    - masks (ndarray): Array of mask images.
    - test_size_ratio (float): Ratio of test data to total data.

    Returns:
    - tuple: Tuple containing train and test data arrays.
    """
    X1, X_test, y1, y_test = train_test_split(images, masks, test_size=test_size_ratio, random_state=0)

    # Further split training data to a smaller subset for quick testing of models
    X_train, X_do_not_use, y_train, y_do_not_use = train_test_split(X1, y1, test_size=2 * test_size_ratio, random_state=0)

    print("Class values in the dataset are...", np.unique(y_train))  # 0 is the background/few unlabeled 

    return X1, X_test, y1, y_test, X_train, X_do_not_use, y_train, y_do_not_use

def recall_m(y_true, y_pred):
    """
    Custom metric function to calculate recall.

    Args:
    - y_true (tensor): True labels.
    - y_pred (tensor): Predicted labels.

    Returns:
    - tensor: Recall value.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    """
    Custom metric function to calculate precision.

    Args:
    - y_true (tensor): True labels.
    - y_pred (tensor): Predicted labels.

    Returns:
    - tensor: Precision value.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    """
    Custom metric function to calculate F1 score.

    Args:
    - y_true (tensor): True labels.
    - y_pred (tensor): Predicted labels.

    Returns:
    - tensor: F1 score value.
    """
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

def trainModel():
    """
    Function to train the model.

    Returns:
    - tuple: Tuple containing the training history, trained model, and test data.
    """
    # Load data
    images, masks = loadData("NIR/", "JPG_Labeled/")
    masks = fixImageRanges(masks)
    images, masks = augmentWithRotation(images, masks)
    
    # Split data
    X1, X_test, y1, y_test, X_train, X_do_not_use, y_train, y_do_not_use = splitData(images, masks)
    
    # Get the model
    model = getModel()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[f1_m])
    model.summary()

    # Train the model
    history = model.fit(X_train, y_train,
                        batch_size=4,
                        verbose=1,
                        epochs=50,
                        validation_data=(X_test, y_test),
                        shuffle=False)

    model.save('aiZoom-k-means-augment-5.hdf5')
    return history, model, X_test, y_test

def evaluateModel(pathToModel, model, X_test, y_test):
    """
    Function to evaluate the model.

    Args:
    - pathToModel (str): Path to the trained model.
    - model: The trained model.
    - X_test (ndarray): Test data.
    - y_test (ndarray): Test labels.

    Returns:
    - tuple: Tuple containing evaluation metrics.
    """
    model = model.load_weights(pathToModel)
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=3)

    n_classes = 3
    testIndex = 3
    IOU_keras = MeanIoU(num_classes=n_classes)
    iou1 = IOU_keras.update_state(y_test[:, :, :, 0], np.around(y_pred[:, :, :, 0], decimals=0))
    res1 = IOU_keras.result().numpy()

    iou2 = IOU_keras.update_state(y_test[:, :, :, 1], np.around(y_pred[:, :, :, 1], decimals=0))
    res2 = IOU_keras.result().numpy()

    iou3 = IOU_keras.update_state(y_test[:, :, :, 2], np.around(y_pred[:, :, :, 2], decimals=0))
    res3 = IOU_keras.result().numpy()

    v_IoU = res1
    r_IoU = res2
    o_IoU = res3
    m_IoU = (res1 + res2 + res3) / 3

    # F1 score
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc', f1_m, precision_m, recall_m])
    loss, accuracy, f1_score, precision, recall = model.evaluate(y_test, y_pred, verbose=0)

    return m_IoU, v_IoU, r_IoU, o_IoU, f1_score, loss, accuracy, precision, recall  # Maybe add class-specific F1 score as well?
