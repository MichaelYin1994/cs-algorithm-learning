# -*- coding: utf-8 -*-
"""
Created on Wed May 16 10:34:09 2018

@author: XPS13
"""
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice

np.random.seed(0)
sns.set(style='dark', font_scale=1.2, palette='deep', color_codes=True)
###############################################################################
def make_dataset():

    n_samples = 1500
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,noise=.05)
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
    blobs = datasets.make_blobs(n_samples=n_samples,
                                 cluster_std=[1.0, 2.5, 0.5],
                                 random_state=170)
    dataset = {"noisy_circles":noisy_circles,
               "noisy_moons":noisy_moons,
               "blobs":blobs}
    param = {"param_1":[0.2, 15],
             "param_2":[0.2, 20],
             "param_3":[0.15, 6]}
    return dataset, param

dataset, param = make_dataset()
nameList = ["noisy_circle", "noisy_moon", "blobs"]
for dataName, paramName, name in zip(dataset.keys(), param.keys(), nameList):
    X = dataset[dataName][0]
    labels = dataset[dataName][1]
    
    X_sc = StandardScaler()
    X = X_sc.fit_transform(X)
    '''
    plt.figure()
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=6)
    plt.title(name + " dataset")
    plt.savefig("..//Plots//" + name + "Dataset.pdf", dpi=800, bbox_inches='tight')
    plt.close("all")
    '''
    db = DBSCAN(eps=param[paramName][0],
                min_samples=param[paramName][1])
    y_pred = db.fit_predict(X)
    
    sc = SpectralClustering(n_clusters=3, affinity="ne")
    
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                         '#f781bf', '#a65628', '#984ea3',
                                         '#999999', '#e41a1c', '#dede00']),
                                      int(max(y_pred) + 1))))
    plt.figure()
    color = colors[y_pred]
    for i in range(X.shape[0]):
        plt.plot(X[i, 0], X[i, 1], '.', color=str(color[i]))
    plt.title(name + " dataset clustering result Eps={}, minPts={}".format(param[paramName][0], param[paramName][1]))
    plt.savefig("..//Plots//" + name + "ClusteringResults.pdf", bbox_inches='tight')