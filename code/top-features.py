# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 11:30:56 2021

@author: katha
"""
import pandas as pd
import numpy as np

# models = ["DT","AB", "RF"]
# metrics = ["Mean_RTT", "Max_RTT", "Min_RTT", "Mean_SyncTraffic", "Max_SyncTraffic", "Min_SyncTraffic", "Mean_ControlPlaneTraffic", "Max_ControlPlaneTraffic", "Min_ControlPlaneTraffic"]

arch = "KN"
model = "RF"
scenario = "N"
metric = "Max_RTT"
features = pd.read_csv("../data/results/"+arch+"_"+model+"_"+ scenario+"_baseline/features_"+metric+".csv",index_col=0)
scores = pd.read_csv("../data/results/"+arch+"_"+model+"_"+ scenario+"_baseline/featureimportances_"+metric+".csv",index_col=0)
for i in range (0,5):
    top_indices = scores.iloc[i][np.logical_not(np.isnan(scores.iloc[i]))].argsort()[-30:][::-1].reset_index(drop=True)
    features_fold = features.iloc[i]
    print(features_fold[top_indices.array])