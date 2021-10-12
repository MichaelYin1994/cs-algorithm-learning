# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 14:55:11 2019

@author: XPS13
"""
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import SpectralClustering
from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice

np.random.seed(0)
sns.set(style='dark', font_scale=1.2, palette='deep', color_codes=True)
###############################################################################
#def make_dataset():
#
#    n_samples = 1500
#    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,noise=.05)
#    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
#    blobs = datasets.make_blobs(n_samples=n_samples,
#                                 cluster_std=[1.0, 2.5, 0.5],
#                                 random_state=170)
#    dataset = {"noisy_circles":noisy_circles,
#               "noisy_moons":noisy_moons,
#               "blobs":blobs}
#    param = {"param_1":[0.2, 15],
#             "param_2":[0.2, 20],
#             "param_3":[0.15, 6]}
#    return dataset, param
#
#if __name__ == "__main__":
#    dataset, param = make_dataset()
#    nameList = ["noisy_circle", "noisy_moon", "blobs"]
#    for dataName, paramName, name in zip(dataset.keys(), param.keys(), nameList):
#        X = dataset[dataName][0]
#        labels = dataset[dataName][1]
#        
#        X_sc = StandardScaler()
#        X = X_sc.fit_transform(X)
#        
#        plt.figure()
#        plt.plot(X[:, 0], X[:, 1], 'k.', markersize=6)
#        plt.title(name + " dataset")
#        plt.grid(True)
#        plt.savefig("..//Plots//" + name + "Dataset.pdf", dpi=800, bbox_inches='tight')
#        plt.close("all")

def get_entropy(basePartition={}, sampleNums=None, theta=0.5):
    '''
    依据文献[1]提出的算法，计算每一组基聚类的每一个聚类的熵，并返回计算结果。
    

    Parameters:
    -----------
    basePartition: dict-like
            基聚类结果。键(key)代表基聚类的编号，值(value)代表基聚类的聚类结果。
    
    sampleNums: int-like
            聚类样本的个数。
    
    theta: int-like
            控制熵分布的范围。
    
    H: dict-like
            保存计算出来的每一个聚类熵的值。
    
    ECI: dict-like
            保存计算出来的每一个聚类的ECI值。
    -----------
    References:
    [1] Huang D, Wang C D, Lai J H. Locally weighted ensemble clustering[J]. IEEE transactions on cybernetics, 2017, 48(5): 1460-1473.
    '''
    
    # 异常输入检测，若无基聚类结果，返回None。
    if len(basePartition) == 0:
        return None
    
    # 检查每一个基聚类具有多少种聚类标签，并存入uniqueLabels字典
    # uniqueLabels字典的键是基聚类编号，值是排序后的不同的标签列表
    # nodeIndex
    uniqueLabels = {}
    for key in basePartition.keys():
        uniqueLabels[key] = np.sort(np.unique(basePartition[key]))
    nodeIndex = np.arange(0, sampleNums)
    
    # 计算每一组基聚类的每一个聚类的熵
    H, ECI, partitionIndexSorted = {}, {}, sorted(list(basePartition.keys()))
    
    # 扫描每一个basePartition
    for basePartitionLabel in partitionIndexSorted:
        H[basePartitionLabel], ECI[basePartitionLabel] = [], []
        
        # 扫描该basePartition的每一个类标签
        for clusterLabel in uniqueLabels[basePartitionLabel]:
            
            # partitionEntropy保存该cluster与每一个partition的熵
            partitionEntropy = []
            
            # 先找到所有属于clusterLabel的样本的编号
            clusterNodes = nodeIndex[clusterLabel == basePartition[basePartitionLabel]]
            print("No.{} partition, cluster {}, Total clusters {}, samples {}".format(basePartitionLabel, clusterLabel, uniqueLabels[basePartitionLabel].max(), len(clusterNodes)))
            
            # 扫描每一个basePartition
            for i in partitionIndexSorted:
                # clusterLabel与该partition的每一个cluter的熵
                entropyEachCluster = []
                for j in uniqueLabels[i]:
                    # 找到该cluter具有的样本的编号
                    # WARNING:此处可被优化，可在开始前使用一个dict记录每一个聚类的样本编号
                    clusterNodesCompare = nodeIndex[j == basePartition[i]]
                    
                    # 计算cluterLabel与该cluster的样本的相交的个数，并计算比例
                    # WARNING:此处也能被优化，可用一矩阵记录cluterLabel与第i个cluster的交集的
                    # 个数，避免重复计算
                    intersectNodes = np.intersect1d(clusterNodes, clusterNodesCompare)
                    tmp = len(intersectNodes) / len(clusterNodes)
                    
                    # 记录与该partition的每一个聚类的熵
                    if tmp == 0:
                        continue
                    else:
                        entropyEachCluster.append(-tmp * np.log2(tmp))
                partitionEntropy.append(sum(entropyEachCluster))
            # 计算cluterLabel聚类的熵
            H[basePartitionLabel].append(sum(partitionEntropy))
            ECI[basePartitionLabel].append(np.exp(-sum(partitionEntropy)/(theta * sampleNums)))
    return H, ECI

if __name__ == "__main__":
    basePartition = {0: [0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 2, 0, 2, 2, 2, 2],
                     1: [0, 0, 0, 0, 1, 1, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2], 
                     3: [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2]}
    # Real theta is 0.32, not 0.5
    H, ECI = get_entropy(basePartition=basePartition, sampleNums=16, theta=0.325)
        