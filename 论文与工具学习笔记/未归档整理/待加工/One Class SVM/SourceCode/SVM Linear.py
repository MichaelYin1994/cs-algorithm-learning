# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 23:01:01 2018

@author: XPS13
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs
import seaborn as sns

sns.set(style="ticks", font_scale=1.2, palette='deep', color_codes=True)

# 创建40个样本点，作为训练数据
X, y = make_blobs(n_samples=40, centers=2, random_state=3)

# 利用训练数据训练模型，是的罚参数很大，等价于没有正则
clf = svm.SVC(kernel='linear', C=1000)
clf.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=40, marker='s', cmap=plt.cm.Paired)
plt.title("The Linear Separable Data")
# 画出决策边界
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# 创建网格
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# 画出决策边界与其裕度
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
# 画出支持向量
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none')
plt.savefig('..//Plots//LinearSeparable.pdf', bbox_inches='tight')
plt.show()
