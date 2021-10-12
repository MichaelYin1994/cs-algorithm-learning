# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 21:35:08 2019

@author: XPS13
"""
from sklearn.datasets import make_moons
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(0)
sns.set(style="ticks", font_scale=1.5, color_codes=True)

###############################################################################
class MLP():
    def __init__(self):
        pass
    
    def calculate_loss(self):
        pass

###############################################################################
if __name__ == "__main__":
    # Generate a dataset and plot it
    X, y = make_moons(200, noise=0.20)
    plt.title("Moon dataset")
    plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)