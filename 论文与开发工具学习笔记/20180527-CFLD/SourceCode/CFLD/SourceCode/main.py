# -*- coding: utf-8 -*-
"""
Created on Sat May 26 12:06:23 2018

@author: XPS13
"""
import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC    
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

sns.set(style='ticks', font_scale=1.2, palette='deep', color_codes=True)
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
    '''
    msno.bar(data, fontsize=15, figsize=(14, 9))
    plt.savefig("..//..//Plots//MissingValuesBar.png", dpi=500, bbox_inches='tight')
    msno.matrix(data, fontsize=15)
    plt.savefig("..//..//Plots//MissingValuesMatrix.png", dpi=500, bbox_inches='tight')
    '''
    
    f, ax = plt.subplots(figsize=(12, 10))
    sns.countplot(data["left"].values, palette="Blues_d")
    plt.xlabel("Left or not")
    plt.ylabel("Number of samples")
    plt.savefig("..//..//Plots//LeftCountplot.png", dpi=500, bbox_inches='tight')

def data_processing():
    data = load_data()
    
    lbl = LabelEncoder()
    data["sales"] = lbl.fit_transform(data["sales"].values)
    data["salary"] = lbl.fit_transform(data["salary"].values)
    
    categoryList = ['number_project', 'sales']
    for featureName in categoryList:
        dummyTmp = pd.get_dummies(data[featureName], prefix=featureName)
        data.drop([featureName], axis=1, inplace=True)
        data = data.join(dummyTmp)
    
    X = data.drop(["left"], axis=1).values
    y = data["left"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    return X_train, X_test, y_train, y_test
    
def logistic_regression(X_train, y_train):
    lr = LogisticRegression(fit_intercept=True, max_iter=5000, penalty='l2', solver='lbfgs')
    C = np.arange(0.00001, 5, 0.1)
    random_state = [1, 2, 3, 4, 5]
    param = {
            "C":C,
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

def support_vector_mechine(X_train, y_train):
    svm = SVC(kernel='rbf')
    C = np.arange(0.00001, 10, 0.1)
    random_state = [1, 2, 3, 4, 5]
    gamma = np.arange(0.1, 5, 0.1)
    param = {
            "C":C,
            "random_state":random_state,
            'gamma':gamma
            }
    print("===================Training SV Classifier===================")
    clf = RandomizedSearchCV(estimator=svm,
                             param_distributions=param,
                             n_iter=1,
                             cv=3,
                             verbose=1,
                             n_jobs=-1)
    clf.fit(X_train, y_train)
    print("==================================================================")
    svm_results = {}
    svm_results["bestEstimator"] = clf.best_estimator_
    svm_results["bestCvScore"] = clf.best_score_
    svm_results["bestParam"] = clf.best_params_
    
    return svm_results

def adaboost(X_train, y_train):
    ada = AdaBoostClassifier()
    n_estimators = [i for i in range(500, 1000)]
    algorithm = ["SAMME", "SAMME.R"]
    learning_rate = np.arange(0.1, 2, 0.1)
    random_state = [1, 2, 3, 4, 5]
    param = {
            "n_estimators":n_estimators,
            "algorithm":algorithm,
            'learning_rate':learning_rate,
            "random_state":random_state,
            }
    print("===================Training Adaboost Classifier===================")
    clf = RandomizedSearchCV(estimator=ada,
                             param_distributions=param,
                             n_iter=200,
                             cv=5,
                             verbose=1,
                             n_jobs=-1)
    clf.fit(X_train, y_train)
    print("==================================================================")
    ada_results = {}
    ada_results["bestEstimator"] = clf.best_estimator_
    ada_results["bestCvScore"] = clf.best_score_
    ada_results["bestParam"] = clf.best_params_
    
    return ada_results

def GBDT(X_train, y_train):
    gbdt = GradientBoostingClassifier()
    n_estimators = [i for i in range(100, 1000)]
    learning_rate = np.arange(0.01, 1, 0.1)
    max_depth = [i for i in range(3, 50)]
    min_samples_split = [i for i in range(2, 30)]
    min_samples_leaf = [i for i in range(1, 30)]
    subsample = np.arange(0.8, 1, 0.02)
    max_features = ['sqrt', 'log2', None]
    random_state = [1, 20, 300, 400, 500]
    
    param = {
            "n_estimators":n_estimators,
            'learning_rate':learning_rate,
            "max_depth":max_depth,
            'min_samples_split':min_samples_split,
            'min_samples_leaf':min_samples_leaf,
            'subsample':subsample,
            'max_features':max_features,
            "random_state":random_state
            }
    print("===================Training GBDT Classifier===================")
    clf = RandomizedSearchCV(estimator=gbdt,
                             param_distributions=param,
                             n_iter=100,
                             cv=5,
                             verbose=1,
                             n_jobs=-1)
    clf.fit(X_train, y_train)
    print("==================================================================")
    gbdt_results = {}
    gbdt_results["bestEstimator"] = clf.best_estimator_
    gbdt_results["bestCvScore"] = clf.best_score_
    gbdt_results["bestParam"] = clf.best_params_
    
    return gbdt_results

def random_forest(X, y, searchMethod='RandomSearch'): 
    rf = RandomForestClassifier()
    n_estimators = [i for i in range(100, 1000)]
    max_depth = [int(x) for x in range(5, 200, 5)]
    max_features = ('auto', 'sqrt', 'log2', None)
    min_samples_split = [int(x) for x in range(2, 20)]
    min_samples_leaf = [int(x) for x in range(1, 20)]
    random_state = [int(x) for x in range(1, 500, 2)]
    param = {
            "n_estimators": n_estimators,
            "max_features": max_features,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_depth": max_depth,
            "random_state": random_state
            }
    print("------------Training Random Forest--------------")
    clf = RandomizedSearchCV(estimator=rf,
                             param_distributions=param,
                             n_iter=5,
                             cv=5,
                             verbose=1,
                             n_jobs=1,
                             )
    clf.fit(X, y)
    print("------------------------------------------------")
    rf_results = {}
    rf_results["bestEstimator"] = clf.best_estimator_
    rf_results["bestCvScore"] = clf.best_score_
    rf_results["bestParam"] = clf.best_params_
    
    return rf_results

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = data_processing()
    
    #lr_results = logistic_regression(X_train, y_train)
    #lr_results["testResults"] = lr_results["bestEstimator"].score(X_test, y_test)
    
    svm_results = support_vector_mechine(X_train, y_train)
    svm_results["testResults"] = svm_results["bestEstimator"].score(X_test, y_test)
    
    #gbdt_results = GBDT(X_train, y_train)
    #gbdt_results["testResults"] = gbdt_results["bestEstimator"].score(X_test, y_test)
    
    #rf_results = random_forest(X_train, y_train)
    #rf_results["testResults"] = rf_results["bestEstimator"].score(X_test, y_test)