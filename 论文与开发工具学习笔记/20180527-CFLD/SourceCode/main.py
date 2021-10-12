# -*- coding: utf-8 -*-
"""
Created on Sun May 27 01:44:51 2018

@author: XPS13
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

h = 0.02  # step size in the mesh
np.random.seed(0)
###############################################################################
def plot_sigmoid():
    x = np.arange(-5, 5, 0.01)
    y = 1/(1+np.exp(-x))
    
    plt.plot(x, y, 'k-')
    #plt.grid(True)
    plt.xlim(x.min(), x.max())
    plt.ylim(y.min(), y.max())
    plt.title("Sigmoid function")
    plt.savefig("..//Plots//sigmoidFcn.pdf", bbox_inches='tight')
    
def load_data():
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    linearly_separable = (X, y)
    
    datasets = [make_moons(noise=0.3, random_state=0),
                make_circles(noise=0.2, factor=0.5, random_state=1),
                linearly_separable]
    
    return datasets

def visualize_data():
    data = load_data()
    datasetName = ["MoonDataset", "CircleDataset", "LinearlyDataset"]
    for d, name in zip(data, datasetName):
        print(name)
        X = d[0]
        y = d[1]
        f, ax = plt.subplots(figsize=(6, 5))
        plt.plot(X[:, 0][y==1], X[:, 1][y==1], 'bo', markerfacecolor='none', markersize=5)
        plt.plot(X[:, 0][y==0], X[:, 1][y==0], 'rx', markersize=5)
        plt.legend(["Positive sample", "Negitive sample"])
        plt.grid(False)
        plt.xlabel("Feature X")
        plt.ylabel("Feature Y")
        plt.title(name)
        plt.savefig("..//Plots//"+name+".pdf", bbox_inches='tight')
     
def lr_plot():
    data = load_data()
    datasetName = ["MoonDataset", "CircleDataset", "LinearlyDataset"]
    
    fig = plt.figure(figsize=(15,4))
    for ind, ds in enumerate(data):
        X, y = ds
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)
        clf = LogisticRegression(penalty='l2', C=0.1, solver='lbfgs', max_iter=200)
        
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        
        ax = plt.subplot(1, len(data), ind+1)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print("lr score is:{}".format(score))
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
            # Put the result into a color plot
            
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=0.8)

        # Plot also the training points
        ax.scatter(X_train[:, 0][y_train==1], X_train[:, 1][y_train==1], c='b', cmap=cm_bright, marker='o',
                   edgecolors='k')
        ax.scatter(X_train[:, 0][y_train==0], X_train[:, 1][y_train==0], c=y_train[y_train==0], cmap=cm_bright, marker='x',
                   edgecolors='k')
        # and testing points
        ax.scatter(X_test[:, 0][y_test==1], X_test[:, 1][y_test==1], c='b', cmap=cm_bright, marker='o',
                   edgecolors='k', alpha=0.6)
        ax.scatter(X_test[:, 0][y_test==0], X_test[:, 1][y_test==0], c=y_test[y_test==0], cmap=cm_bright, marker='x',
                   edgecolors='k', alpha=0.6)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        plt.xlabel(datasetName[ind])
    plt.savefig("..//Plots//LogisticRegression.pdf", dpi=600)

def svm_plot():
    data = load_data()
    datasetName = ["MoonDataset", "CircleDataset", "LinearlyDataset"]
    
    fig = plt.figure(figsize=(15,4))
    for ind, ds in enumerate(data):
        X, y = ds
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)
        clf = SVC(C=1, gamma=2)
        
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        
        ax = plt.subplot(1, len(data), ind+1)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print("svm score is:{}".format(score))
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
            # Put the result into a color plot
            
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=0.8)

        # Plot also the training points
        ax.scatter(X_train[:, 0][y_train==1], X_train[:, 1][y_train==1], c='b', cmap=cm_bright, marker='o',
                   edgecolors='k')
        ax.scatter(X_train[:, 0][y_train==0], X_train[:, 1][y_train==0], c=y_train[y_train==0], cmap=cm_bright, marker='x',
                   edgecolors='k')
        # and testing points
        ax.scatter(X_test[:, 0][y_test==1], X_test[:, 1][y_test==1], c='b', cmap=cm_bright, marker='o',
                   edgecolors='k', alpha=0.6)
        ax.scatter(X_test[:, 0][y_test==0], X_test[:, 1][y_test==0], c=y_test[y_test==0], cmap=cm_bright, marker='x',
                   edgecolors='k', alpha=0.6)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        plt.xlabel(datasetName[ind])
    plt.savefig("..//Plots//svm.pdf")
    

def adaboost_plot():
    data = load_data()
    datasetName = ["MoonDataset", "CircleDataset", "LinearlyDataset"]
    
    fig = plt.figure(figsize=(15,4))
    for ind, ds in enumerate(data):
        X, y = ds
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)
        clf = AdaBoostClassifier(n_estimators=100)
        
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        
        ax = plt.subplot(1, len(data), ind+1)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print("adaboost score is:{}".format(score))
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
            # Put the result into a color plot
            
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=0.8)

        # Plot also the training points
        ax.scatter(X_train[:, 0][y_train==1], X_train[:, 1][y_train==1], c='b', cmap=cm_bright, marker='o',
                   edgecolors='k')
        ax.scatter(X_train[:, 0][y_train==0], X_train[:, 1][y_train==0], c=y_train[y_train==0], cmap=cm_bright, marker='x',
                   edgecolors='k')
        # and testing points
        ax.scatter(X_test[:, 0][y_test==1], X_test[:, 1][y_test==1], c='b', cmap=cm_bright, marker='o',
                   edgecolors='k', alpha=0.6)
        ax.scatter(X_test[:, 0][y_test==0], X_test[:, 1][y_test==0], c=y_test[y_test==0], cmap=cm_bright, marker='x',
                   edgecolors='k', alpha=0.6)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        plt.xlabel(datasetName[ind])
    plt.savefig("..//Plots//adaboost.pdf")

def gbdt_plot():
    data = load_data()
    datasetName = ["MoonDataset", "CircleDataset", "LinearlyDataset"]
    
    fig = plt.figure(figsize=(15,4))
    for ind, ds in enumerate(data):
        X, y = ds
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)
        clf = GradientBoostingClassifier(n_estimators=900)
        
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        
        ax = plt.subplot(1, len(data), ind+1)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print("GBDT score is:{}".format(score))
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
            # Put the result into a color plot
            
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=0.8)

        # Plot also the training points
        ax.scatter(X_train[:, 0][y_train==1], X_train[:, 1][y_train==1], c='b', cmap=cm_bright, marker='o',
                   edgecolors='k')
        ax.scatter(X_train[:, 0][y_train==0], X_train[:, 1][y_train==0], c=y_train[y_train==0], cmap=cm_bright, marker='x',
                   edgecolors='k')
        # and testing points
        ax.scatter(X_test[:, 0][y_test==1], X_test[:, 1][y_test==1], c='b', cmap=cm_bright, marker='o',
                   edgecolors='k', alpha=0.6)
        ax.scatter(X_test[:, 0][y_test==0], X_test[:, 1][y_test==0], c=y_test[y_test==0], cmap=cm_bright, marker='x',
                   edgecolors='k', alpha=0.6)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        plt.xlabel(datasetName[ind])
    plt.savefig("..//Plots//gbdt.pdf")
    
def random_forest_plot():
    data = load_data()
    datasetName = ["MoonDataset", "CircleDataset", "LinearlyDataset"]
    
    fig = plt.figure(figsize=(15,4))
    for ind, ds in enumerate(data):
        X, y = ds
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)
        clf = RandomForestClassifier(n_estimators=900)
        
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        
        ax = plt.subplot(1, len(data), ind+1)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print("Random forest score is:{}".format(score))
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
            # Put the result into a color plot
            
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=0.8)

        # Plot also the training points
        ax.scatter(X_train[:, 0][y_train==1], X_train[:, 1][y_train==1], c='b', cmap=cm_bright, marker='o',
                   edgecolors='k')
        ax.scatter(X_train[:, 0][y_train==0], X_train[:, 1][y_train==0], c=y_train[y_train==0], cmap=cm_bright, marker='x',
                   edgecolors='k')
        # and testing points
        ax.scatter(X_test[:, 0][y_test==1], X_test[:, 1][y_test==1], c='b', cmap=cm_bright, marker='o',
                   edgecolors='k', alpha=0.6)
        ax.scatter(X_test[:, 0][y_test==0], X_test[:, 1][y_test==0], c=y_test[y_test==0], cmap=cm_bright, marker='x',
                   edgecolors='k', alpha=0.6)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        plt.xlabel(datasetName[ind])
    plt.savefig("..//Plots//rf.pdf")
if __name__ == "__main__":
    #data = load_data()
    #visualize_data()
    #svm_plot()
    #adaboost_plot()
    #gbdt_plot()