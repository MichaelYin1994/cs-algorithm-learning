#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 20:29:34 2019

@author: yinzhuo
"""

import time
import pickle
from functools import wraps
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn.metrics import mean_absolute_error
from numpy import iinfo, finfo, int8, int16, int32, int64, float32, float64

colors = ["red", "green", "blue", "yellow"]
sns.set(style="ticks", font_scale=1.2, palette='deep', color_codes=True)
###############################################################################
def signal_mean_filter(signal_vals=None, file_path=None,
                       window_length_list=None,
                       mode="flat", save_fig=False):
    """Smooth the data using a window with requested size."""
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
        file_path = "..//Plots//signal_filtered.png"

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


def timefn(fcn):
    """Decorator for efficency analysis. """
    @wraps(fcn)
    def measure_time(*args, **kwargs):
        start = time.time()
        result = fcn(*args, **kwargs)
        end = time.time()
        print("@timefn: " + fcn.__name__ + " took {:.5f}".format(end-start)
            + " seconds.")
        return result
    return measure_time


@timefn
def basic_feature_report(data_table=None, precent=None):
    """Reporting basic characteristics of the tabular data data_table."""
    precent = precent or [0.01, 0.25, 0.5, 0.75, 0.95, 0.99]
    if data_table is None:
        return None
    num_samples = len(data_table)

    # Basic statistics
    basic_report = data_table.isnull().sum()
    basic_report = pd.DataFrame(basic_report, columns=["#missing"])
    basic_report["missing_precent"] = basic_report["#missing"]/num_samples
    basic_report["#uniques"] = data_table.nunique(dropna=False).values
    basic_report["types"] = data_table.dtypes.values
    basic_report.reset_index(inplace=True)
    basic_report.rename(columns={"index":"feature_name"}, inplace=True)

    # Basic quantile of data
    data_description = data_table.describe(precent).transpose()
    data_description.reset_index(inplace=True)
    data_description.rename(columns={"index":"feature_name"}, inplace=True)
    basic_report = pd.merge(basic_report, data_description,
        on='feature_name', how='left')
    return basic_report


class LoadSave():
    """Class for loading and saving object in .pkl format."""
    def __init__(self, file_name=None):
        self._file_name = file_name

    def save_data(self, data=None, path=None):
        """Save data to path."""
        if path is None:
            assert self._file_name is not None, "Invaild file path !"
        else:
            self._file_name = path
        self.__save_data(data)

    def load_data(self, path=None):
        """Load data from path."""
        if path is None:
            assert self._file_name is not None, "Invaild file path !"
        else:
            self._file_name = path
        return self.__load_data()

    def __save_data(self, data=None):
        """Save data to path."""
        print("--------------Start saving--------------")
        print("@SAVING DATA TO {}.".format(self._file_name))
        with open(self._file_name, 'wb') as file:
            pickle.dump(data, file)
        print("@SAVING SUCESSED !")
        print("----------------------------------------\n")

    def __load_data(self):
        """Load data from path."""
        if not self._file_name:
            raise ValueError("Invaild file path !")
        print("--------------Start loading--------------")
        print("@LOADING DATA FROM {}.".format(self._file_name))
        with open(self._file_name, 'rb') as file:
            data = pickle.load(file)
        print("@LOADING SUCCESSED !")
        print("-----------------------------------------\n")
        return data


class ReduceMemoryUsage():
    """
    ----------
    Author: Michael Yin
    E-Mail: zhuoyin94@163.com
    ----------
    
    @Description:
    ----------
    Reduce the memory usage of pandas dataframe.
    
    @Parameters:
    ----------
    data: pandas DataFrame-like
        The dataframe that need to be reduced memory usage.
    verbose: bool
        Whether to print the memory reduction information or not.
        
    @Return:
    ----------
    Memory-reduced dataframe.
    """
    def __init__(self, data_table=None, verbose=True):
        self._data_table = data_table
        self._verbose = verbose

    def type_report(self, data_table):
        """Reporting basic characteristics of the tabular data data_table."""
        data_types = list(map(str, data_table.dtypes.values))
        basic_report = pd.DataFrame(data_types, columns=["types"])
        basic_report["feature_name"] = list(data_table.columns)
        return basic_report

    @timefn
    def reduce_memory_usage(self):
        memory_reduced_data = self.__reduce_memory()
        return memory_reduced_data

    def __reduce_memory(self):
        print("\nReduce memory process:")
        print("-------------------------------------------")
        memory_before_reduced = self._data_table.memory_usage(
            deep=True).sum() / 1024**2
        types = self.type_report(self._data_table)
        if self._verbose is True:
            print("@Memory usage of data is {:.5f} MB.".format(
                memory_before_reduced))

        # Scan each feature in data_table, reduce the memory usage for features
        for ind, name in enumerate(types["feature_name"].values):
            # ToBeFixed: Unstable query.
            feature_type = str(
                types[types["feature_name"] == name]["types"].iloc[0])

            if (feature_type in "object") and (feature_type in "datetime64[ns]"):
                try:
                    feature_min = self._data_table[name].min()
                    feature_max = self._data_table[name].max()

                    # np.iinfo for reference:
                    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.iinfo.html
                    # numpy data types reference:
                    # https://wizardforcel.gitbooks.io/ts-numpy-tut/content/3.html
                    if "int" in feature_type:
                        if feature_min > iinfo(int8).min and feature_max < iinfo(int8).max:
                            self._data_table[name] = self._data_table[name].astype(int8)
                        elif feature_min > iinfo(int16).min and feature_max < iinfo(int16).max:
                            self._data_table[name] = self._data_table[name].astype(int16)
                        elif feature_min > iinfo(int32).min and feature_max < iinfo(int32).max:
                            self._data_table[name] = self._data_table[name].astype(int32)
                        else:
                            self._data_table[name] = self._data_table[name].astype(int64)
                    else:
                        if feature_min > finfo(float32).min and feature_max < finfo(float32).max:
                            self._data_table[name] = self._data_table[name].astype(float32)
                        else:
                            self._data_table[name] = self._data_table[name].astype(float64)
                except Exception as error_msg:
                    print("\n--------ERROR INFORMATION---------")
                    print(error_msg)
                    print("Error on the {}".format(name))
                    print("--------ERROR INFORMATION---------\n")
            if self._verbose is True:
                print("Processed {} feature({}), total is {}.".format(
                    ind + 1, name, len(types)))

        memory_after_reduced = self._data_table.memory_usage(
            deep=True).sum() / 1024**2
        if self._verbose is True:
            print("@Memory usage after optimization: {:.5f} MB.".format(
                memory_after_reduced))
            print("@Decreased by {:.5f}%.".format(
                100 * (memory_before_reduced - memory_after_reduced) / memory_before_reduced))
        print("-------------------------------------------")
        return self._data_table
