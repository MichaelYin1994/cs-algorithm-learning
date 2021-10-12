# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 22:28:31 2020

@author: XPS13
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from utils import LoadSave, signal_savgol_filter, signal_mean_filter

colors = ["red", "green", "blue", "yellow"]
sns.set(style="ticks", font_scale=1.2, palette='deep', color_codes=True)
###############################################################################
def parse_curve_val(curve=None):
    """Str curve value to numeric curve value."""
    return list(map(float, curve.split(",")))


def plot_signal_weighting(save_fig=True, file_path=None):
    """Plot some typical weighting functions."""
    plt.close("all")
    if file_path is None:
        file_path = "..//Plots//weighting_function.png"

    M = 37
    hanning_vals = np.hanning(M=M)
    hamming_vals = np.hamming(M=M)
    bartlett_vals = np.bartlett(M=M)
    blackman_vals = np.blackman(M=M)

    lw_val = 2
    f, ax = plt.subplots(figsize=(8, 6))
    ax.plot(hanning_vals, linewidth=lw_val, color="k",
            label="Hanning weights")
    ax.plot(hamming_vals, linewidth=lw_val, color="red",
            label="Hamming weights")
    ax.plot(bartlett_vals, linewidth=lw_val, color="blue",
            label="Bartlett weights")
    ax.plot(blackman_vals, linewidth=lw_val, color="green",
            label="Blackman weights")
    ax.set_title("M = {}".format(M))
    ax.tick_params(axis="both", labelsize=10)
    ax.set_xlim(0, len(hanning_vals)-1)
    ax.legend(fontsize=10)
    ax.grid(True)
    plt.tight_layout()

    if save_fig:
        plt.savefig(file_path, bbox_inches="tight", dpi=700)

if __name__ == "__main__":
    df_curve = pd.read_csv("..//Data//current_signal.csv")
    
    # 170
    plot_list = [170, 125, 348, 222, 110]
#    plot_list = [222]
    
#    signal_savgol_filter(signal, save_fig=True,
#                         window_length_list=[37, 87, 137],
#                         poly_order_list=[1, 2, 4],
#                         file_path="..//Plots//signal_filtered_{}.png".format(ind))

    for i in plot_list:
        signal = np.array(parse_curve_val(curve=df_curve["curve"].iloc[i]))
#        signal_mean_filter(signal, save_fig=True,
#                           window_length_list=[19, 133],
#                           mode="bartlett",
#                           file_path="..//Plots//signal_mean_filtered_{}.png".format(i))

        signal_savgol_filter(signal, save_fig=True,
                             window_length_list=[27, 57, 127],
                             poly_order_list=[1, 2, 4],
                             file_path="..//Plots//s_g_filtered_{}.png".format(i))


#    plot_signal_weighting()
