# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 23:17:47 2018

@author: XPS13
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
from scipy import linalg
import seaborn as sns

sns.set(color_codes=True, style='ticks', font_scale=1.1)
###############################################################################
def plot_ocsvm(filePath=None, nu=0.1, gamma=0.1):
    np.random.seed(1)
    
    xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
    # 生成训练数据
    X_1 = 0.4 * np.random.randn(100, 2)
    X_1 = np.r_[X_1 + 2, X_1 - 4]
    X_2 = 0.2 * np.random.randn(100, 2)
    X_2 = np.r_[X_2 + 3, X_2 - 1]
    X_train = np.vstack((X_1, X_2))
    print(X_train.shape)
    # 离群点数据生成
    X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
    
    # OCSVM学习
    clf = svm.OneClassSVM(nu=nu, kernel="rbf", gamma=gamma)
    clf.fit(X_train)
    y_pred_train = clf.predict(X_train)
    y_pred_outliers = clf.predict(X_outliers)
    n_error_train = y_pred_train[y_pred_train == -1].size
    n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size
    
    # plot the line, the points, and the nearest vectors to the plane
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
#    plt.title(r"$\nu$: {}; $\eta$: {}; trainErr: {}/400; abnormalErr: {}/40".format(nu, gamma, n_error_train, n_error_outliers), fontsize=12)
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
    a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
    plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')
    
    s = 40
    b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=s, edgecolors='k')
    c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red', marker='x', s=s,
                    edgecolors='k')
    plt.axis('tight')
    plt.xlim((-5, 5))
    plt.ylim((-5, 5))
#    plt.legend([a.collections[0], b1, c],
#               ["boundary", "training", "abnormal"],
#               loc="upper left",
#               prop=matplotlib.font_manager.FontProperties(size=11))
    
#    plt.show()
    
def ocsvm_test(filePath=None, nu=0.1, gamma=0.1):
    xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
    
    # 生成训练数据
    X_1 = 0.4 * np.random.randn(100, 2)
    X_1 = np.r_[X_1 + 2, X_1 - 4]
    
    X_2 = 0.2 * np.random.randn(100, 2)
    X_2 = np.r_[X_2 + 3, X_2 - 1]
    
    X_train = np.vstack((X_1, X_2))
    # 离群点数据生成
    X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
    X_train = np.vstack((X_train, X_outliers))
    trainSize = len(X_train)
    
    # OCSVM学习
    clf = svm.OneClassSVM(nu=nu, kernel="rbf", gamma=gamma)
    clf.fit(X_train)
    y_pred_train = clf.predict(X_train)
    y_pred_outliers = clf.predict(X_outliers)
    n_error_train = y_pred_train[y_pred_train == -1].size
    n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size
    
    # plot the line, the points, and the nearest vectors to the plane
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.title("Nu: {}; \eta: {} \n trainErr: {}/400 ; abnormalErr: {}/40".format(nu, gamma, n_error_train, n_error_outliers), fontsize=12)
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
    a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
    plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')
    
    s = 40
    b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=s, edgecolors='k')
    plt.axis('tight')
    plt.xlim((-5, 5))
    plt.ylim((-5, 5))
    plt.legend([a.collections[0], b1],
               ["boundary", "training"],
               loc="upper left",
               prop=matplotlib.font_manager.FontProperties(size=11))
    
    alpha = clf.dual_coef_ / (trainSize * nu)
    
    rbfMat = rbf_kernel(X_train, X_train, gamma=gamma)
    kernelConst = np.sum(rbfMat) ** (0.5)
    
    theta = []
    for i in range(trainSize):
        tmp = np.arccos(rbfMat[i, :].sum()/kernelConst)
        theta.append(tmp)
    theta = np.array(theta)
    theta = np.sort(theta)
    
    return theta
    
if __name__ == "__main__":
    '''
    # 演示\eta变化的时候，决策边界的变化大小。
    plt.figure()
    plot_ocsvm(nu=0.0001, gamma=0.3)
    plt.savefig("..//Plots//etaChange_1.pdf", bbox_inches='tight')
    
    plt.figure()
    plot_ocsvm(nu=0.5, gamma=0.00001)
    plt.savefig("..//Plots//etaChange_2.pdf", bbox_inches='tight')
    '''
    # 九宫格图
    plt.figure(figsize=(16,12))
    plt.subplots_adjust(left=0.3, bottom=0.2, top=1.2, right=1.1)
    
    plt.subplot(3, 3, 1)
    plot_ocsvm(nu=0.01, gamma=0.05)
    
    plt.subplot(3, 3, 2)
    plot_ocsvm(nu=0.01, gamma=0.3)
    
    plt.subplot(3, 3, 3)
    plot_ocsvm(nu=0.01, gamma=1)
    
    plt.subplot(3, 3, 4)
    plot_ocsvm(nu=0.2, gamma=0.05)
    
    plt.subplot(3, 3, 5)
    plot_ocsvm(nu=0.2, gamma=0.3)
    
    plt.subplot(3, 3, 6)
    plot_ocsvm(nu=0.2, gamma=1)
    
    plt.subplot(3, 3, 7)
    plot_ocsvm(nu=0.7, gamma=0.05)
    
    plt.subplot(3, 3, 8)
    plot_ocsvm(nu=0.7, gamma=0.3)
    
    plt.subplot(3, 3, 9)
    plot_ocsvm(nu=0.7, gamma=1)
    plt.savefig("..//Plots//OCSVM.pdf", bbox_inches='tight')
