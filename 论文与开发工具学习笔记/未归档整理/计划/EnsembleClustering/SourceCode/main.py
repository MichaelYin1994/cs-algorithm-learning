# -*- coding: utf-8 -*-
"""
Created on Mon May 20 17:18:47 2019

@author: XPS13
"""
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from itertools import cycle, islice
sns.set(style='dark', font_scale=1.2, palette='deep', color_codes=True)
np.random.seed(2019)
###############################################################################
def make_dataset(n_samples=2000):
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

###############################################################################
class kmeans(object):
    def __init__(self, numberOfCluster=4, seed=None, max_iter=200, distance="pre_computed"):
        self.numberOfCluster = numberOfCluster
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)
        self.max_iter = max_iter
        self.distance = distance
        
    def initialize_centroids(self, data):
        initial_centroids = np.random.permutation(data.shape[0])[:self.numberOfCluster]
        self.centroids = data[initial_centroids]
        return self.centroids
    
    def assign_clusters(self, data):
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        if self.distance == "pre_computed":
            dist_to_centroid = None
        
        dist_to_centroid =  pairwise_distances(data, self.centroids, metric='euclidean')
        self.cluster_labels = np.argmin(dist_to_centroid, axis = 1)
        
        return  self.cluster_labels
    
    def update_centroids(self, data):
        self.centroids = np.array([data[self.cluster_labels == i].mean(axis = 0) for i in range(self.numberOfCluster)])
        return self.centroids

    def predict(self, data):
        return self.assign_clusters(data)
    
    def fit_kmeans(self, data):
        self.centroids = self.initialize_centroids(data)
        
        # Main kmeans loop
        for currentIter in range(self.max_iter):
            self.cluster_labels = self.assign_clusters(data)
            self.centroids = self.update_centroids(data)          
            if iter % 100 == 0:
                print("Running Model Iteration %d " %currentIter)
        print("Model finished running")
        return self       

###############################################################################
class ClusterEnsemble():
    def __init__(self, clusterRes={}, numberOfCluster=4, theta=1):
        self._clusterRes = clusterRes
        self._numberOfCluster = numberOfCluster
        self._theta = theta
    
    def get_entropy(self, clusterDict={}, samples=None, theta=1.695):
        '''
        clusterDict: dict-like, containing the clustering results.
        samples: int-like, the number of samples.
        theta: the ECI parameter.
        '''
        if len(clusterDict) == 0:
            return None
        
        # Determine the total number of the base clusters
        uniqueLabels = {}
        for key in clusterDict.keys():
            uniqueLabels[key] = sorted(np.unique(clusterDict[key]))
        nodeIndex = np.arange(0, samples)
        
        H, ECI = {}, {}
        
        # 遍历C_{1} ... c_{n^{c}}
        sortedUniqueLabelsKeys = sorted(uniqueLabels.keys())
        for clusterLabel in sortedUniqueLabelsKeys:
            H[clusterLabel], ECI[clusterLabel] = [], []
            for classLabel in sortedUniqueLabelsKeys:
                classEntropy = []
                classNodes = nodeIndex[classLabel == clusterDict[clusterLabel]]
                for i in sortedUniqueLabelsKeys:
                    entropyEachCluster = []
                    for j in sortedUniqueLabelsKeys:
                        classNodesCompare = nodeIndex[j == clusterDict[i]]
                        intersectNodes = np.intersect1d(classNodes, classNodesCompare)
                        tmp = len(intersectNodes) / len(classNodes)
                        if tmp == 0:
                            continue
                        else:
                            entropyEachCluster.append(-tmp * np.log2(tmp))
                    classEntropy.append(sum(entropyEachCluster))
                H[clusterLabel].append(sum(classEntropy))
                ECI[clusterLabel].append(np.exp(- sum(classEntropy) / (theta * len(clusterDict) )))
        return H, ECI
    
    def get_weighted_co_association_matrix(self, clusterDict={}, samples=None, theta=1.695):
        if len(clusterDict) == 0:
            return None
    
        # Calculating weights
        self._H, self._ECI = self.get_entropy(clusterDict=clusterDict, samples=samples, theta=theta)
        
        # Constructing the co_association matrix
        coMat = np.zeros((samples, samples))
        nodeIndex = np.arange(0, samples)
        for node in range(samples):
            for clusterLabel in sorted(clusterDict.keys()):
                nodeClassLabel = clusterDict[clusterLabel][node]
                connectedNode = nodeIndex[clusterDict[clusterLabel] == nodeClassLabel]
                for j in connectedNode:
                    coMat[node, j] += 1 * self._ECI[clusterLabel][nodeClassLabel]
        return coMat/len(clusterDict)
    
    def ensemble(self):
        if len(self._clusterRes) == 0:
            return None
        
        # Get the number of clusters
        ind = list(self._clusterRes.keys())
        numberOfSamples = len(self._clusterRes[ind[0]])
        
        # Get the co-mat
        self._coMat = self.get_weighted_co_association_matrix(clusterDict=self._clusterRes,
                                                              samples=numberOfSamples, 
                                                              theta=self._theta)
        
        # KMeans clustering
        kmeans = KMeans(n_clusters=self._numberOfCluster)
        kmeans.fit()
    
###############################################################################
if __name__ == "__main__":
    dataset, params = make_dataset()
    nameList = ["noisy_circle", "noisy_moon", "blobs"]
    
    for dataName, name in zip(dataset.keys(), nameList):
        X = dataset[dataName][0]
        labels = dataset[dataName][1]
        
        X_sc = StandardScaler()
        X = X_sc.fit_transform(X)
    
        clf = kmeans(numberOfCluster=3, seed=None, max_iter=200, distance="pre_computed")
        clf.fit_kmeans(X)
        y_pred = clf.predict(X)
        
#        # Plot the clustering results
#        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
#                                         '#f781bf', '#a65628', '#984ea3',
#                                         '#999999', '#e41a1c', '#dede00']),
#                                      int(max(y_pred) + 1))))
#        plt.figure()
#        color = colors[y_pred]
#        for i in range(X.shape[0]):
#            plt.plot(X[i, 0], X[i, 1], '.', color=str(color[i]))
#        plt.title(name)