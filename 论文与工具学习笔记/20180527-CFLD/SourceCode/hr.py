# -*- coding: utf-8 -*-
"""
Created on Sat May 26 12:06:23 2018

@author: XPS13
"""
import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno
from matplotlib import rcParams
rcParams['patch.force_edgecolor'] = True
rcParams['patch.facecolor'] = 'b'

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

from sklearn.linear_model import LogisticRegression

np.random.seed(0)
###############################################################################
def load_data():
    data = pd.read_csv("..//Data//HumanResource.csv")
    return data

def visualize_data():
    data = load_data()
    featureName = list(data.columns)
    
    for i in featureName[0:5]:
        nums = np.random.randint(0, 3000)
        index = np.random.randint(0, len(data), nums)
        data[i].iloc[index] = np.nan

    msno.bar(data, fontsize=15, figsize=(14, 9))
    plt.savefig("..//Plots//MissingValuesBar.pdf", dpi=500, bbox_inches='tight')
    msno.matrix(data, fontsize=15)
    plt.savefig("..//Plots//MissingValuesMatrix.pdf", dpi=500, bbox_inches='tight')
    
    f, ax = plt.subplots(figsize=(5, 5))
    plt.hist(data["left"].values, bins=3, facecolor='b', alpha=0.5)
    plt.xlabel("Left or not")
    plt.ylabel("Number of samples")
    plt.savefig("..//Plots//LeftCountplot.pdf", dpi=500, bbox_inches='tight')

def data_processing():
    data = load_data()
    
    lbl = LabelEncoder()
    data["sales"] = lbl.fit_transform(data["sales"].values)
    data["salary"] = lbl.fit_transform(data["salary"].values)
    X = data.drop(["left"], axis=1).values
    y = data["left"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    return X_train, X_test, y_train, y_test
    
def logistic_regression(X_train, y_train):
    lr = LogisticRegression(fit_intercept=True, max_iter=2000, penalty='l2')
    C = np.arange(0.00001, 10, 0.2)
    solver = ['lbfgs', 'saga']
    random_state = [1, 2, 3, 4, 5]
    param = {
            "C":C,
            "solver":solver,
            "random_state":random_state
            }
    print("===================Training Logistic Regression===================")
    clf = RandomizedSearchCV(estimator=lr,
                             param_distributions=param,
                             n_iter=100,
                             cv=5,
                             verbose=1,
                             n_jobs=-1)
    clf.fit(X_train, y_train)
    print("==================================================================")
    lr_results = {}
    lr_results["bestEstimator"] = clf.best_estimator_
    lr_results["bestCvScore"] = clf.best_score_
    lr_results["bestParam"] = clf.best_params_
    
    return lr_results

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = data_processing()