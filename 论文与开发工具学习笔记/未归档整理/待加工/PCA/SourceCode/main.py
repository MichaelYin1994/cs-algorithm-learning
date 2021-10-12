# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 16:08:29 2018

@author: XPS13
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
import seaborn as sns

from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler

np.random.seed(0)
sns.set(style='darkgrid', font_scale=1.2, palette='deep', color_codes=True)
###############################################################################
X, y = make_circles(n_samples=400, factor=0.3, noise=0.05)
sc = StandardScaler(with_std=False)
X = sc.fit_transform(X)

kpca = KernelPCA(n_components=1, kernel='rbf', gamma=10, fit_inverse_transform=True)
X_new = kpca.fit_transform(X)
X_reconstruct = kpca.inverse_transform(X_new)

plt.figure()
for i in range(X.shape[0]):
    plt.plot(X[i, 0], X[i, 1], '.', color="b" if y[i] == 1 else "r")
plt.title('Circles Dataset', fontweight='bold')

plt.figure()
for i in range(X.shape[0]):
    plt.plot(X_reconstruct[i, 0], X_reconstruct[i, 1], '.', color="g" if y[i] == 1 else "k")
plt.title('Circles Dataset Reconstructed', fontweight='bold')


