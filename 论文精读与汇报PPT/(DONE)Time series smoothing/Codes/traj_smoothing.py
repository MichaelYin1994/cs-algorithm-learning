# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 22:28:03 2020

@author: XPS13
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from utils import LoadSave, signal_savgol_filter, signal_mean_filter

colors = ["red", "green", "blue", "yellow"]
sns.set(style="ticks", font_scale=1.2, palette='deep', color_codes=True)
###############################################################################
def load_traj_data():
    """Loading trajectory sample data from the local path."""
    file_processor = LoadSave()
    return file_processor.load_data("..//Data//traj_data.pkl")


def plot_traj_seq(traj=None, feature_name=None,
                  file_path=None):
    """Plot a sigle sequence for a signal."""
    plt.close("all")
    if feature_name is None:
        feature_name = "speed"
    if file_path is None:
        file_path = "..//Plots//traj_speed.png"
    target, boat_id = traj["type"].unique()[0], traj["boat_id"].unique()[0]

    f, ax = plt.subplots(figsize=(12, 6))
    ax.plot(traj[feature_name], color="k", linewidth=2, label=feature_name)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(fontsize=12)
    ax.set_title
    ax.set_xlim(0, len(traj))
    ax.grid(True)
    ax.set_title("Target: {}, boat_id: {}".format(target, boat_id))
    plt.tight_layout()
    plt.savefig(file_path, bbox_inches="tight", dpi=700)


def plot_traj_coord(traj=None, file_path=None):
    """Plot a single trajectory."""
    plt.close("all")
    if file_path is None:
        file_path = "..//Plots//traj_coord.png"

    fig, ax = plt.subplots(figsize=(7, 5))
    target, boat_id = traj["type"].unique()[0], traj["boat_id"].unique()[0]
    # Plot the raw trajectory
    traj_coord = traj[["x", "y"]].values / 10000
    ax.plot(traj_coord[:, 0], traj_coord[:, 1], marker="o",
        markersize=2, color="k", linewidth=1)
    ax.plot(traj_coord[0, 0], traj_coord[0, 1],
        marker="s", markersize=5, color="blue", label="Start")
    ax.plot(traj_coord[-1, 0], traj_coord[-1, 1],
        marker="^", markersize=5, color="red", label="end")
    ax.tick_params(axis="both", labelsize=8)
    ax.legend(fontsize=10)
    ax.grid(True)
    ax.set_title("Target: {}, boat_id: {}".format(target, boat_id))
    plt.tight_layout()
    plt.savefig(file_path, bbox_inches="tight", dpi=700)


def plot_traj_speed_direction_sequence(traj=None, file_path=None):
    plt.close("all")
    if file_path is None:
        file_path = "..//Plots//traj_speed_direction_sequence.png"

    fig, ax_objs = plt.subplots(2, 1, figsize=(12, 8))
    ax_objs = ax_objs.ravel()
    ax_objs[0].plot(traj["speed"], color="g", linewidth=1.8, label="Speed")
    ax_objs[1].plot(traj["direction"], color="g", linewidth=1.8, label="Direction")

    for ax in ax_objs:
        ax.tick_params(axis="both", labelsize=12)
        ax.legend(fontsize=12)
        ax.set_xlim(0, len(traj))
    plt.tight_layout()
    plt.savefig(file_path, bbox_inches="tight", dpi=500)


if __name__ == "__main__":
    traj_data = load_traj_data()
    ind = 1
    traj = traj_data[ind]

    plot_traj_speed_direction_sequence(traj)
#    plot_traj_seq(traj, feature_name="speed")
#    plot_traj_seq(traj, feature_name="direction",
#                  file_path="..//Plots//traj_direction.png")
#    plot_traj_coord(traj)
#    signal_savgol_filter(signal_vals=traj["speed"].values, 
#                         file_path="..//Plots//speed_filtered.png")
#    plot_list = [0, 1, 2, 3, 5]
#    for i in plot_list:
#        signal = traj_data[i]["speed"].values
##        signal_mean_filter(signal, save_fig=True,
##                           window_length_list=[5, 11, 17],
##                           file_path="..//Plots//speed_mean_filtered_{}.png".format(i))
#        signal_savgol_filter(signal, save_fig=True,
#                             window_length_list=[11, 17, 27],
#                             poly_order_list=[1, 2, 4],
#                             file_path="..//Plots//s_g_filtered_{}.png".format(i))
