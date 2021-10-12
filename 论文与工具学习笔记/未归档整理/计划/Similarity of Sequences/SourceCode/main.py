# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 20:16:50 2018

@author: XPS13
"""
import pandas as pd
import numpy as np
import pickle
from WeaponLibrary import LoadSave
from WeaponLibrary import timefn

if __name__ == "__main__":
    dataPath = "..//Data//TrajectoryDataComplete.pkl"
    featurePath = "..//Data//TrajectoryDataFeaturesComplete.pkl"
    
    ls = LoadSave(dataPath)
    trajData = ls.load_data()
    ls._fileName = featurePath
    trajDataFeature = ls.load_data()
