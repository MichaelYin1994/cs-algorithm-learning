# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 17:28:49 2020

@author: XPS13
"""

from datetime import datetime
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import mean_absolute_error, mean_squared_error

import pywt
from scipy.signal import savgol_filter
from statsmodels.tsa.ar_model import AR

warnings.filterwarnings('ignore')
sns.set(style="ticks", font_scale=1.2, palette='deep', color_codes=True)
colors = ["C" + str(i) for i in range(0, 9+1)]
PLOT_SAVE_PATH = ".//plots//"
###############################################################################
###############################################################################
def load_data(path=None, nrows=100):
    """Loading the demo data from the local path."""
    ts = pd.read_csv(path + "demo.csv", nrows=nrows)
    return ts


def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


def denoise_signal(x, wavelet='db4', level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1/0.6745) * maddest(coeff[-level])

    uthresh = sigma * np.sqrt(2*np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])

    return pywt.waverec(coeff, wavelet, mode='per')


def signal_mean_filter(signal_vals=None, file_path=None,
                       window_length_list=None,
                       mode="flat", save_fig=False):
    """Smooth the data using a window with requested size."""
    if file_path is None:
        file_path = "..//plots//mean_smoothed_signal.png"
    if signal_vals.ndim != 1:
        raise ValueError("Smooth only accepts 1 dimension arrays!")
    if signal_vals.size < min(window_length_list):
        raise ValueError("Input vector needs to be bigger than window size.")
    if max(window_length_list) < 3:
        return signal_vals
    if mode not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    div_res = [divmod(item, 2)[1] == 1 for item in window_length_list]
    if sum(div_res) != len(div_res):
        raise ValueError("The window size nust be odd!")

    params, filtered_signals_window = window_length_list, []
    for ind, window_len in enumerate(params):
        signal_tmp = np.r_[signal_vals[(window_len//2):0:-1],
                           signal_vals,
                           signal_vals[-2:-(window_len//2)-2:-1]]
        if mode == 'flat':
            w = np.ones(window_len, 'd')
        else:
            w = eval('np.' + mode + '(window_len)')
        filtered_signals_window.append(np.convolve(w / w.sum(), signal_tmp,
                                                   mode='valid'))

    plt.close("all")
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(signal_vals, color="k", 
        linewidth=1.5, label="Original Signal")
    for ind, item in enumerate(window_length_list):
        label_name = "window_length_{}".format(item)
        ax.plot(filtered_signals_window[ind], color=colors[ind],
               linewidth=1.5, label=label_name)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_xlim(0, len(signal_vals))
    ax.legend(fontsize=12)
    ax.grid(True)
    plt.tight_layout()

    if save_fig:
        plt.savefig(file_path, bbox_inches="tight", dpi=700)


def signal_savgol_filter(signal_vals=None, file_path=None,
                         window_length_list=None, poly_order_list=None,
                         mode="mirror", save_fig=False):
    """Signal noise filtering."""
    plt.close("all")
    if file_path is None:
        file_path = "..//plots//s_g_smoothed_signal.png"

    if window_length_list is None:
        window_length_list = [7, 13, 21]
    if poly_order_list is None:
        poly_order_list = [2, 6]

    params, filtered_signals_window = window_length_list, []
    for param in params:
        tmp = savgol_filter(signal_vals, window_length=param,
                            polyorder=3, mode=mode)
        filtered_signals_window.append(tmp)

    params, filtered_signals_poly = poly_order_list, []
    for param in params:
        tmp = savgol_filter(signal_vals, window_length=15,
                            polyorder=param, mode=mode)
        filtered_signals_poly.append(tmp)

    plt.close("all")
    fig, ax_objs = plt.subplots(2, 1, figsize=(14, 8))
    ax_objs[0].plot(signal_vals, color="k", 
        linewidth=1.5, label="Original Signal")
    for ind, item in enumerate(window_length_list):
        label_name = "window_length_{}".format(item)
        ax_objs[0].plot(filtered_signals_window[ind], color=colors[ind],
               linewidth=1.5, label=label_name)
    ax_objs[0].tick_params(axis="both", labelsize=12)
    ax_objs[0].set_xlim(0, len(signal_vals))
    ax_objs[0].legend(fontsize=12)
    ax_objs[0].grid(True)

    ax_objs[1].plot(signal_vals, color="k", 
        linewidth=1.5, label="Original Signal")
    for ind, item in enumerate(poly_order_list):
        label_name = "poly_order_{}".format(item)
        ax_objs[1].plot(filtered_signals_poly[ind], color=colors[ind],
               linewidth=1.5, label=label_name)
    ax_objs[1].tick_params(axis="both", labelsize=12)
    ax_objs[1].set_xlim(0, len(signal_vals))
    ax_objs[1].legend(fontsize=12)
    ax_objs[1].grid(True)
    plt.tight_layout()

    if save_fig:
        plt.savefig(file_path, bbox_inches="tight", dpi=700)


def wavelets_filter(signal_vals=None, file_path=None, save_fig=False,
                    level_list=None):
    """Signal noise filtering."""
    plt.close("all")
    if file_path is None:
        file_path = "..//plots//s_g_smoothed_signal.png"

    if level_list is None:
        level_list = [1, 2, 3]

    params, filtered_signals = level_list, []
    for param in params:
        tmp = denoise_signal(signal_vals, level=param)
        filtered_signals.append(tmp)

    plt.close("all")
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(signal_vals, color="k", 
        linewidth=1.5, label="Original Signal")
    for ind, item in enumerate(level_list):
        label_name = "window_length_{}".format(item)
        ax.plot(filtered_signals[ind], color=colors[ind],
               linewidth=1.5, label=label_name)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_xlim(0, len(signal_vals))
    ax.legend(fontsize=12)
    ax.grid(True)
    plt.tight_layout()

    if save_fig:
        plt.savefig(file_path, bbox_inches="tight", dpi=700)


if __name__ == "__main__":
    ts = load_data(path="..//data//", nrows=15000)
    ts["sg_0"] = savgol_filter(ts["signal"].values, window_length=39,
        polyorder=2, mode="mirror")
    ts["sg_1"] = savgol_filter(ts["signal"].values, window_length=89,
        polyorder=2, mode="mirror")

    # Prediction 1: Exponential Weighted Function.
    ts["sg_0_ewm"] = ts["signal"].ewm(span=9, adjust=False).mean()
    
#    model = AR(ts["signal"].values)
#    model_fit = model.fit(maxlag=10)
#    pred_res = model_fit.predict(len(ts), len(ts))
    feat_name = "sg_0_ewm"
    # Simple visualizing
    start, end = 1000, 8000
    f, ax = plt.subplots(figsize=(14, 8))
    ax.plot(ts["signal"].values[start:end], color="k", 
            linewidth=1.5, label="Original Signal")
    ax.plot(ts["sg_0"].values[start:end], color="red", 
            linewidth=1.5, label="Savgol filter 0")
    ax.plot(ts["sg_1"].values[start:end], color="green", 
            linewidth=1.5, label="Savgol filter 1")
#    ax.plot(ts[feat_name].values[start:end], color="blue", 
#            linewidth=1.5, label="predict")
    ax.tick_params(axis="both", labelsize=12)
    ax.set_xlim(0, end - start)
    ax.legend(fontsize=12)
    ax.grid(True)
    plt.tight_layout()
