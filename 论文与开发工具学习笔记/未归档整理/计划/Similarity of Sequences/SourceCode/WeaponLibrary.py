# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 18:02:16 2018

@author: Administrator
"""
import pandas as pd
from pandas import DataFrame
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from functools import wraps
import time
from sklearn.metrics import roc_curve, auc
###############################################################################
def ROC_curve(y_real, y_prob, colorCode='b'):
    fpr, tpr, _ = roc_curve(y_real, y_prob)
    rocAuc = auc(fpr, tpr)
    plt.plot(fpr, tpr)
    
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve and auc = {:4f}".format(rocAuc))

class LoadSave(object):
    def __init__(self, fileName=None):
        self._fileName = fileName
    
    def save_data(self, data):
        assert self._fileName != None, "Invaild file path !"
        self.__save_data(data)
    
    def load_data(self):
        assert self._fileName != None, "Invaild file path !"
        return self.__load_data()
        
    def __save_data(self, data):
        print("--------------Start saving--------------")
        print("Save data to path {}.".format(self._fileName))
        f = open(self._fileName, "wb")
        pickle.dump(data, f)
        f.close()
        print("--------------Saving successed !--------------\n")
        
    def __load_data(self):
        assert self._fileName != None, "Invaild file path !"
        print("--------------Start loading--------------")
        print("Load from path {}.".format(self._fileName))
        f = open(self._fileName, 'rb')
        data = pickle.load(f)
        f.close()
        print("--------------loading successed !--------------\n")
        return data

def save_data(data, fileName):
    f = open(fileName, "wb")
    pickle.dump(data, f)
    f.close()

def load_data(fileName):
    f = open(fileName, 'rb')
    data = pickle.load(f)
    f.close()
    return data

def dataProcess():
    PATH = "..//Data//line2"
    trajData = {}
    fileNameList = os.listdir(PATH)
    fileNameList.sort(key = lambda x:int(x[:-4]))
    
    j = 0
    for ind, fileName in enumerate(fileNameList):
        trajData[j] = pd.read_table(PATH+'//'+fileName, sep=',', index_col=False)
        if len(trajData[j]) < 6:
            print("Not enough pts! Total is {}, ind is {}.".format(len(fileNameList), ind))
        else:
            trajData[j].rename(columns={"time":"TIME", "x":"X", "y":"Y", "v":"V", 'BBoxSize':'boundingBoxSize', 'in':'IN'}, inplace=True)
            trajData[j]["TIME"] = pd.to_datetime(trajData[j]["TIME"])
            trajData[j]["unixTime"] = trajData[j]["TIME"].copy()
            trajData[j]["TIME"] = trajData[j]["TIME"] - trajData[j]["TIME"].iloc[0]
            trajData[j]["TIME"] = trajData[j]["TIME"].dt.seconds + trajData[j]["TIME"].dt.microseconds/1000000
            print("Current is {}, total is {}, ind is {}.".format(j, len(fileNameList), ind))
            j += 1
            
    # 不明原因无法检测长度
    trajData.pop(10566)
    save_data(trajData, "..//Data//TrajectoryDataXianOriginal.pkl")

###############################################################################
def timefn(fn):
    @wraps(fn)
    def measure_time(*args, **kwargs):
        start = time.time()
        result = fn(*args, **kwargs)
        end = time.time()
        print("@timefn: " + fn.__name__ + " took " + str(end-start) + " seconds")
        return result
    return measure_time
    
def plot_random_n_traj(n, trajData):
    # 接受的trajData数据是一个字典，键是轨迹编号，值是轨迹DataFrame
    trajNums = len(trajData)
    trajInd = list(trajData.keys())
    ind = np.random.randint(0, trajNums, n)
    plt.figure(figsize=(30,300))
    plt.imshow(background)
    for index, item in enumerate(ind):
        i = trajInd[item]
        plt.plot(trajData[i]["X"], trajData[i]["Y"], "r-o", markersize=3)
    plt.axis('off')

def plot_list_traj(index, trajData):
    # 接受的trajData数据是一个字典，键是轨迹编号，值是轨迹DataFrame
    plt.figure(figsize=(30,300))
    plt.imshow(background)
    for item in index:
        plt.plot(trajData[item]["X"], trajData[item]["Y"], "ro", markersize=3)
    plt.axis('off')
    
###############################################################################
def calc_traj_complexity(traj):
    # 计算总距离的Array数组
    distArray = np.sqrt(np.sum(np.square(traj[["X", "Y"]].values[:-1] - traj[["X", "Y"]].values[1:]), axis=1))
    distCumSum = distArray.cumsum()
    actualDistance = distCumSum[-1]
    directDistance = np.sqrt(np.sum(np.square(traj[["X", "Y"]].values[-1, :] - traj[["X", "Y"]].values[0, :])))
    
    # 计算轨迹的复杂程度
    trajComplexity = directDistance/actualDistance
    return trajComplexity, directDistance, distArray

def sampling_compression(traj, distArray, samplingNums=50):
    # 接受一条轨迹作为函数的输入，输入类型是DataFrame
    # 输出为对应traj长度的索引，索引为轨迹停止点的索引
    trajCompressed = traj
    
    # 显示采样是否正常，flag=1代表成功；flag=-1代表失败
    flag = 1
    distCumSum = 0

    if len(traj) < samplingNums:
        print("Too less points ! Want {} but only get {} !".format(samplingNums, len(traj)))
        flag = -1
        return trajCompressed, flag
    
    distCumSum = distArray.cumsum()
    distCumSum = np.insert(distCumSum, 0, 0)
    totalDistance = distCumSum[-1]
    samplingSequence = np.linspace(0, totalDistance, int(samplingNums+1))
    trajCompressedVal = np.zeros((samplingNums, 2))
    
    for i in range(1, len(samplingSequence)):
        cond_1 = (distCumSum >= samplingSequence[i-1])
        cond_2 = (distCumSum < samplingSequence[i])
        cond = cond_1 & cond_2
        if cond.sum() == 0:
            tmp_front = traj[cond_2].iloc[0].values
            tmp_behind = traj[cond_1].iloc[0].values
            trajCompressedVal[i-1, :] = (tmp_front + tmp_behind)/2
        else:
            trajCompressedVal[i-1, :] = traj[cond].values.mean(axis=0)
    trajCompressed = DataFrame(trajCompressedVal, columns=["X", "Y"])
    
    return trajCompressed, flag
###############################################################################
class CalculateAdjacencyMat(object):
    def __init__(self, trajData=None, trajDataFeatuers=None, trajUsedNums=500, load=True):
        self._trajData = trajData
        self._trajDataFeatures = trajDataFeatuers
        self._load = load
        self._trajUsedNums = trajUsedNums
        
        # 设置初始化参数设置
        self._minDistCond = 40
        self._minPts = 10
        self._eta = 0.1
        
    def set_param(self, minDistCond, minPts, eta):
        self._minDistCond = minDistCond
        self._minPts = minPts
        self._eta = eta
        
    def save_data(self, data, fileName):
        f = open(fileName, "wb")
        pickle.dump(data, f)
        f.close()
    
    def select_trajData(self):
        if self._trajUsedNums == 'all':
            pass
        else:
            self._trajUsedIndex = list(self._trajData.keys())[:self._trajUsedNums]
            self._trajUsedData = dict([(index, self._trajData[index]) for index in self._trajUsedIndex])
            self._trajUsedFeatures = self._trajDataFeatures.iloc[0:self._trajUsedNums]

    def calc_adjacency_mat(self):
        start = time.clock()
        if self._load == True:
            self.select_trajData()
            self._adjacencyMat = load_data("..//Data//AdjacencyMatrix_" + str(self._trajUsedNums) + "_" + str(self._minDistCond) + "_" + str(self._minPts) + '.pkl')
        else:
            self.select_trajData()
            self._adjacencyMat = create_lcss_similar_matrix(self._trajUsedData, minDistCond=self._minDistCond, minPts=self._minPts, eta=self._eta)
            self.save_data(self._adjacencyMat, "..//Data//AdjacencyMatrix_" + str(self._trajUsedNums) + "_" + str(self._minDistCond) + "_" + str(self._minPts) + '.pkl')
        end = time.clock()
        print("@calc_adjacency_mat took {}".format(end-start))
        return self._trajUsedData, self._trajUsedFeatures, self._adjacencyMat

###############################################################################
# 实现并查集算法，返回的是对应的array数组
class UnionFind(object):
    def __init__(self, arrayLength, connected):
        self._arrayLength = arrayLength
        self._connected = connected
        self._results = np.arange(0, self._arrayLength)
        self._results = np.tile(self._results, reps=2).reshape((2, self._arrayLength))
    
    def find(self, pos):
        return self._results[1, pos]
    
    def connected(self, pos_1, pos_2):
        return self.find(pos_1) == self.find(pos_2)
    
    def union(self, pos_1, pos_2):
        id_1 = self.find(pos_1)
        id_2 = self.find(pos_2)
        
        if id_1 == id_2:
            return
        else:
            idTmp = min(id_1, id_2)
            self._results[1, :][self._results[1, :]==id_1] = idTmp
            self._results[1, :][self._results[1, :]==id_2] = idTmp
            
    def fit(self):
        for i in self._connected:
            pos_1 = i[0]
            pos_2 = i[1]
            if self.connected(pos_1, pos_2):
                continue
            else:
                self.union(pos_1, pos_2)
                
        return self._results