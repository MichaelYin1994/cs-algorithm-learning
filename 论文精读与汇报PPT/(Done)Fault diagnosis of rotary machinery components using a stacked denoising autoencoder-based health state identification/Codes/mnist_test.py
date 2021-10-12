#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 15:25:52 2019

@author: yinzhuo
"""

import pandas as pd
import numpy as np
import gc

from StackedDenoisingAE import StackedDenoisingAE
from UTILS import simple_lgb_clf

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
###############################################################################
###############################################################################
def load_mnist(nrows=None):
    path = "..//Data//Mnist//"
    trainData = pd.read_csv(path + "mnist_train.csv", nrows=nrows)
    testData = pd.read_csv(path + "mnist_test.csv", nrows=nrows)
    return trainData, testData

# Type 1: SDA with 2 layer network, no input noise, with dropout
def type_1(trainUsedNums=1000, testUsedNums=500):
    # Step 1: Parameters setting
    img_rows, img_cols = 28, 28
    trainData, testData = load_mnist(nrows=None)
    X_train, X_test, y_train, y_test = trainData.drop("label", axis=1).values, testData.drop("label", axis=1).values, trainData["label"].values, testData["label"].values
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # Step 2: Select train instances
    X_train, y_train = X_train[:trainUsedNums, :], y_train[:trainUsedNums]
    X_test, y_test = X_test[:testUsedNums, :], y_test[:testUsedNums]
    print("Number of training and testing samples :{} and {}.".format(trainUsedNums, testUsedNums))

    # Step 3: Denoising AE training
    # 80% -- 70%
    n_hid = [int(trainData.shape[1] * 0.8), int(trainData.shape[1] * 0.7)]
    sdae = StackedDenoisingAE(n_layers=2, n_hid=n_hid, dropout=[0.02], nb_epoch=200,
                              enc_act=["tanh", "relu"], dec_act=["linear", "linear"], batch_size=64)
    model, (dense_train, dense_val, dense_test), recon_mse = sdae.get_pretrained_sda(X_train, X_test, X_test, dir_out='..//Models//', write_model=False)
    lgb_dense = simple_lgb_clf(X_train=dense_train, X_valid=dense_test, X_test=dense_test, y_train=y_train, y_valid=y_test, y_test=y_test, name="sda_2_layer_dropout_dense")
    gc.collect()
    return lgb_dense

if __name__ == "__main__":
    # Specify the input parameters
    trainUsedNums = 2000
    testUsedNums = 500
    type_1(trainUsedNums=trainUsedNums, testUsedNums=testUsedNums)
    flag = 1