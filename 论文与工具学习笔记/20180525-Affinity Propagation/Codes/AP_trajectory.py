# -*- coding: utf-8 -*-
"""
Created on Tue May  8 23:10:08 2018

@author: XPS13
"""

from sklearn.cluster import AffinityPropagation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from WeaponLibrary import sampling_compression
from WeaponLibrary import load_data

sns.set(style='dark', font_scale=1.2, palette='deep', color_codes=True)
background = plt.imread("backgroundXian.bmp")
trkData = load_data("trkDataXianAfterFiltering.pkl")
similarMatrix = load_data("trkDataXianSimilarMat.pkl")
distArray = load_data("trkDataXianDistArray.pkl")
###############################################################################

trkDataCompressed = {}
start = time.clock()
for ind in list(trkData.keys()):
    trkCompressedTmp, flag = sampling_compression(trkData[ind][['X', 'Y']], distArray[ind]['distArray'], samplingPercent=0.07)
    if flag == -1:
        continue
    else:
        trkDataCompressed[ind] = trkCompressedTmp
end = time.clock()
print("Compression time is {}".format(end-start))

trkUseNums = 1000
trkInd = list(trkDataCompressed.keys())
compressedInd = trkInd[0:(trkUseNums)]
compressedTrk = dict([(index, trkDataCompressed[index]) for index in compressedInd])

ap = AffinityPropagation(preference=-40,
                         affinity='precomputed')
labels = ap.fit_predict(similarMatrix)


eachPlotTrkNums = 4
colorCode = ['b', 'g', 'r', 'y']
flag = 0
plt.figure()
plt.imshow(background)

'''
def plot_all_clustering_results(self):
        eachPlotTrkNums = 4
        colorCode = ['b', 'g', 'r', 'y']
        flag = 0
        plt.figure()
        plt.imshow(background)
        for currentLabel in list(range(self._clusteringNums)):
            if flag <= (eachPlotTrkNums-1):
                for ind, item in enumerate(self._trkDataCompressed.keys()):
                    if self._results["labels"][ind] == currentLabel:
                        plt.plot(self._trkDataCompressed[item]["X"].values,
                                 self._trkDataCompressed[item]["Y"].values,
                                 colorCode[flag]+"o",
                                 markersize=3)
                flag += 1
            else:
                plt.savefig("..//Plots//allClusteringResults_" + str(currentLabel) + '.pdf', dpi=800, bbox_inches='tight')
                plt.figure()
                plt.imshow(background)
                flag = 0
                for ind, item in enumerate(self._trkDataCompressed.keys()):
                    if self._results["labels"][ind] == currentLabel:
                        plt.plot(self._trkDataCompressed[item]["X"].values,
                                 self._trkDataCompressed[item]["Y"].values,
                                 colorCode[flag]+"o",
                                 markersize=3)
                flag += 1
'''