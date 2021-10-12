# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 18:02:16 2018

@author: Administrator
"""
import pandas as pd
from pandas import DataFrame, Series
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from numba import jit
from functools import wraps
import time

background = plt.imread("backgroundXian.bmp")
def save_data(data, fileName):
    f = open(fileName, "wb")
    pickle.dump(data, f)
    f.close()

def load_data(fileName):
    f = open(fileName, 'rb')
    data = pickle.load(f)
    f.close()
    return data
    
def dataReadProcess():
    PATH = ["..//Data//0700-1000",
            "..//Data//1100-1200",
            "..//Data//1700-2100"]
    trkData = {}
    j = 0
    for path in PATH:
        fileNameList = os.listdir(path)
        fileNameList.sort(key=lambda x:int(x[:-4]))
        for ind, fileName in enumerate(fileNameList):
            trkData[j] = pd.read_table(path+'//'+fileName, sep=',', index_col=False)
            trkData[j].rename(columns={"time":"TIME", "x":"X", "y":"Y", "v":"V"}, inplace=True)
            trkData[j]["TIME"] = pd.to_datetime(trkData[j]["TIME"])
            print("Current is {}.".format(j))
            j+=1
            
    save_data(trkData, "..//Data//trkDataXian.pkl")
    return trkData
    
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
    
def plot_random_n_trk(n, trkData):
    # 接受的trkData数据是一个字典，键是轨迹编号，值是轨迹DataFrame
    trkNums = len(trkData)
    trkInd = list(trkData.keys())
    ind = np.random.randint(0, trkNums, n)
    plt.figure(figsize=(30,300))
    plt.imshow(background)
    for index, item in enumerate(ind):
        i = trkInd[item]
        plt.plot(trkData[i]["X"], trkData[i]["Y"], "ro", markersize=3)
    plt.axis('off')

def plot_list_trk(index, trkData):
    # 接受的trkData数据是一个字典，键是轨迹编号，值是轨迹DataFrame
    plt.figure(figsize=(30,300))
    plt.imshow(background)
    for item in index:
        plt.plot(trkData[item]["X"], trkData[item]["Y"], "ro", markersize=5)
    plt.axis('off')
    
###############################################################################
def get_poi(trkData):
    trkNums = len(trkData)
    startPoints = np.zeros((trkNums, 2))
    endPoints = np.zeros((trkNums, 2))
    trkInd = list(trkData.keys())
    for ind, item in enumerate(trkInd):
        trk = trkData[item]
        maxLength = len(trk["X"]) - 1
        startPoints[ind, 0], startPoints[ind, 1] = trk["X"][0], trk["Y"][0]
        endPoints[ind, 0], endPoints[ind, 1] = trk["X"][maxLength], trk["Y"][maxLength]
    return startPoints, endPoints

def calc_trk_complexity(trk):
    distArray = np.sqrt(np.sum(np.square(trk[["X", "Y"]].values[:-1] - trk[["X", "Y"]].values[1:]), axis=1))
    distCumSum = distArray.cumsum()
    actualDistance = distCumSum[-1]
    directDistance = np.sqrt(np.sum(np.square(trk[["X", "Y"]].values[-1, :] - trk[["X", "Y"]].values[0, :])))
    trkComplexity = directDistance/actualDistance
    return trkComplexity, directDistance, distArray

@jit
def sampling_compression(trk, distArray, samplingPercent=0.1):
    # 接受一条轨迹作为函数的输入，输入类型是DataFrame
    # 输出为对应trk长度的索引，索引为轨迹停止点的索引
    trkLength = len(trk)
    trkCompressed = trk
    
    # 显示采样是否正常，1代表成功；-1代表失败
    flag = 1
    distCumSum = 0
    samplingNums = samplingPercent * trkLength
    samplingNums = 80

    if len(trk) < samplingNums:
        print("Too less points ! Want {} but only get {} !".format(samplingNums, len(trk)))
        flag = -1
        return trkCompressed, flag
    
    distCumSum = distArray.cumsum()
    distCumSum = np.insert(distCumSum, 0, 0)
    totalDistance = distCumSum[-1]
    samplingSequence = np.linspace(0, totalDistance, int(samplingNums+1))
    trkCompressedVal = np.zeros((samplingNums, 2))
    
    for i in range(1, len(samplingSequence)):
        cond = (distCumSum > samplingSequence[i-1]) & (distCumSum < samplingSequence[i])
        if cond.sum() == 0:
            trkCompressedVal[i-1, :] = trk[distCumSum > samplingSequence[i-1]][["X", "Y"]].iloc[0].values
        else:
            trkCompressedVal[i-1, :] = trk[cond].values.mean(axis=0)
    trkCompressed = DataFrame(trkCompressedVal, columns=["X", "Y"])
    
    return trkCompressed, flag

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